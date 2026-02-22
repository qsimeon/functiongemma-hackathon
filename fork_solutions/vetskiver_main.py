# Source: https://github.com/vetskiver/functiongemma-hackathon

import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = cactus_init(functiongemma_path)

    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]

    start_time = time.time()

    gemini_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(tools=gemini_tools),
    )

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    for candidate in gemini_response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                })

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """Smart hybrid inference strategy: route to cloud for known weak cases, on-device otherwise."""
    tool_names = {t["name"] for t in tools}
    tool_map = {t["name"]: t for t in tools}
    user_text_raw = " ".join(m["content"] for m in messages if m["role"] == "user")
    strict_reask = "Return all tool calls required by the user request; no omissions."
    import re

    def strip_trailing_punct(s):
        return s.rstrip(" .!?,:;")

    def strip_leading_article(s):
        s_strip = s.lstrip()
        for art in ("the ", "a ", "an "):
            if s_strip.lower().startswith(art):
                return s_strip[len(art):]
        return s

    def string_candidates(value):
        candidates = []
        for v in (value, strip_trailing_punct(value), strip_leading_article(strip_trailing_punct(value))):
            v = v.strip()
            if v and v not in candidates:
                candidates.append(v)
        return candidates

    def align_string_to_prompt(value, prompt_text):
        if not isinstance(value, str) or not value or not prompt_text:
            return value
        text_lower = prompt_text.lower()
        candidates = string_candidates(value)

        candidates.sort(key=len, reverse=True)

        for cand in candidates:
            cand_lower = cand.lower()
            idx = text_lower.find(cand_lower)
            if idx != -1:
                return prompt_text[idx : idx + len(cand_lower)]

        return value

    def string_supported_by_prompt(value, prompt_text):
        if not isinstance(value, str) or not value or not prompt_text:
            return True
        text_lower = prompt_text.lower()
        for cand in string_candidates(value):
            if cand.lower() in text_lower:
                return True
        return False


    def is_valid_call_set(result):
        calls = result.get("function_calls") or []
        if not calls:
            return False
        for call in calls:
            name = call.get("name")
            if name not in tool_map:
                return False
            args = call.get("arguments")
            if not isinstance(args, dict):
                return False
            required = (tool_map[name].get("parameters") or {}).get("required", []) or []
            for key in required:
                if key not in args or args[key] in (None, ""):
                    return False
            for key, val in args.items():
                if isinstance(val, str) and not string_supported_by_prompt(val, user_text_raw):
                    return False
        return True

    def normalize_result_strings(result):
        calls = result.get("function_calls") or []
        if not calls:
            return result
        for call in calls:
            args = call.get("arguments")
            if not isinstance(args, dict):
                continue
            for k, v in list(args.items()):
                if isinstance(v, str):
                    aligned = align_string_to_prompt(v, user_text_raw)
                    key_lower = k.lower()
                    if any(tag in key_lower for tag in ("title", "task", "subject", "reminder")):
                        aligned_lower = aligned.lower()
                        prompt_lower = user_text_raw.lower()
                        if not any(f"{art}{aligned_lower}" in prompt_lower for art in ("the ", "a ", "an ")):
                            aligned = strip_leading_article(aligned)
                    if any(tag in key_lower for tag in ("message", "text", "content")):
                        aligned = strip_trailing_punct(aligned)
                    args[k] = aligned
        return result

    # Count how many distinct tool actions are likely requested
    # Hard cases need multiple tool calls — send to cloud
    def tokenize(text, min_len=4):
        return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) >= min_len}

    user_text_lower = user_text_raw.lower()
    user_tokens = tokenize(user_text_raw, min_len=3)

    def tool_tokens(tool):
        tokens = set()
        tokens |= tokenize(tool.get("name", ""))
        tokens |= tokenize(tool.get("description", ""))
        params = (tool.get("parameters") or {}).get("properties", {}) or {}
        for key in params.keys():
            tokens |= tokenize(key)
        return tokens

    scored_tools = []
    for t in tools:
        tokens = tool_tokens(t)
        score = sum(1 for tok in tokens if tok in user_tokens)
        if score > 0:
            scored_tools.append((score, t.get("name")))

    scored_tools.sort(key=lambda x: x[0], reverse=True)
    top1 = scored_tools[0][0] if scored_tools else 0
    top2 = scored_tools[1][0] if len(scored_tools) > 1 else 0
    clarity = (top1 - top2) / max(top1, 1)
    overlap_count = sum(1 for score, _ in scored_tools if score >= 1)
    multi_intent_hint = bool(re.search(r"\b(and|then|also|plus)\b", user_text_lower) or "," in user_text_raw)

    def expected_tools_for_user():
        if not scored_tools:
            return []
        score_cutoff = max(1, top1 * 0.6)
        if multi_intent_hint and overlap_count >= 2:
            score_cutoff = 1
        expected = [name for score, name in scored_tools if score >= score_cutoff and name in tool_names]
        expected = list(dict.fromkeys(expected))  # preserve order, dedupe
        return expected[:3]

    expected_tools = expected_tools_for_user()
    likely_actions = len(expected_tools)

    dynamic_threshold = confidence_threshold
    if top1 > 0:
        # Lower threshold when intent is clear and likely single-tool
        dynamic_threshold = confidence_threshold - (0.25 * clarity)
        # Additional small relaxation for very clear single-tool prompts
        if likely_actions == 1 and clarity >= 0.7:
            dynamic_threshold -= 0.10
        # Raise threshold slightly for multi-tool intent
        dynamic_threshold += 0.10 * max(0, likely_actions - 1)
        dynamic_threshold = max(0.50, min(confidence_threshold, dynamic_threshold))

    def completeness_score(result, expected_tools):
        if not expected_tools:
            return 0
        calls = result.get("function_calls") or []
        called = {c.get("name") for c in calls if isinstance(c, dict)}
        return sum(1 for t in expected_tools if t in called)

    def strict_reask_content(expected):
        if expected:
            return f"Return tool calls for: {', '.join(expected)}. {strict_reask}"
        return strict_reask

    def should_reask(cloud_result, expected):
        if not expected:
            return False
        # Avoid extra cloud calls unless multi-tool is expected.
        if len(expected) < 2:
            return False
        # Re-ask when cloud missed some expected tools.
        return completeness_score(cloud_result, expected) < len(expected)

    # Tools the on-device model handles poorly
    WEAK_TOOLS = {"set_timer", "create_reminder"}

    # Always go to cloud if any weak tool is present
    if tool_names & WEAK_TOOLS:
        cloud = generate_cloud(messages, tools)
        if should_reask(cloud, expected_tools):
            strict_msg = {
                "role": "user",
                "content": strict_reask_content(expected_tools),
            }
            cloud_strict = generate_cloud(messages + [strict_msg], tools)
            if completeness_score(cloud_strict, expected_tools) > completeness_score(cloud, expected_tools):
                cloud = cloud_strict
        cloud["source"] = "cloud (fallback)"
        cloud["local_confidence"] = 0.0
        return normalize_result_strings(cloud)


    # If multiple actions detected, go straight to cloud to save time.
    if likely_actions >= 2:
        cloud = generate_cloud(messages, tools)
        if should_reask(cloud, expected_tools):
            strict_msg = {
                "role": "user",
                "content": strict_reask_content(expected_tools),
            }
            cloud_strict = generate_cloud(messages + [strict_msg], tools)
            if completeness_score(cloud_strict, expected_tools) > completeness_score(cloud, expected_tools):
                cloud = cloud_strict
        cloud["source"] = "cloud (fallback)"
        cloud["local_confidence"] = 0.0
        return normalize_result_strings(cloud)

    # Single-tool case: try on-device first
    local = generate_cactus(messages, tools)

    valid_local = is_valid_call_set(local)

    if local["confidence"] >= dynamic_threshold and valid_local:
        local["source"] = "on-device"
        return normalize_result_strings(local)

    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local["confidence"]
    cloud["total_time_ms"] += local["total_time_ms"]
    return normalize_result_strings(cloud)


def print_result(label, result):
    """Pretty-print a generation result."""
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result["function_calls"]:
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


############## Example usage ##############

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)
