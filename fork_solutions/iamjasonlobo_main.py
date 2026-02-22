# Source: https://github.com/iamjasonlobo/functiongemma-hackathon

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
        [{"role": "system", "content": "You are a function calling assistant. Call ALL required functions for the user's request. Use exact values from the user's message."}] + messages,
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
        config=types.GenerateContentConfig(
            tools=gemini_tools,
            temperature=0,
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode='ANY')
            ),
            system_instruction=(
                "You are a precise function-calling router. "
                "Your job is to translate user requests into function calls. "
                "Rules: "
                "1) Identify EVERY action the user wants and call the corresponding function for EACH one. Never skip any action. "
                "2) Extract argument values as concisely as possible — use the minimal keyword from the user's input, not full phrases. Drop articles and filler words. "
                "3) Always call functions. Never respond with text."
            ),
        ),
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

    print(f"  [DBG-CLOUD] calls={json.dumps(function_calls)}")

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


def _validate_calls(function_calls, tools, messages):
    """
    Validate function calls against tool schemas and input signals.
    Returns True if calls look structurally sound, False if something's off.
    Generic checks only — no benchmark-specific logic.
    """
    if not function_calls:
        return False

    # Build a lookup of valid tool names → schema
    tool_map = {t["name"]: t for t in tools}

    for call in function_calls:
        name = call.get("name", "")
        args = call.get("arguments", {})

        # 1. Function name must exist in provided tools
        if name not in tool_map:
            return False

        schema = tool_map[name]
        props = schema.get("parameters", {}).get("properties", {})
        required = schema.get("parameters", {}).get("required", [])

        # 2. All required args must be present and non-empty
        for req in required:
            if req not in args:
                return False
            val = args[req]
            if val is None or (isinstance(val, str) and val.strip() == ""):
                return False
            # Numeric sanity: reject negative numbers for integer fields
            # and unreasonably large values (likely hallucinated)
            if isinstance(val, (int, float)):
                if val < 0:
                    return False
                if props.get(req, {}).get("type") == "integer" and val > 1440:
                    # 1440 minutes = 24 hours, reasonable upper bound for time-related ints
                    return False

        # 3. Argument keys must be valid for this tool
        for key in args:
            if key not in props:
                return False

    # 4. Multi-action intent check: reject when user clearly wants multiple
    #    actions but local only returned 1 call
    user_text = " ".join(m["content"] for m in messages if m["role"] == "user").lower()
    separators = [" and ", " then ", " also ", " plus "]
    sep_count = sum(1 for s in separators if s in user_text)

    if sep_count >= 1 and len(function_calls) == 1 and len(tools) >= 2:
        return False

    return True


def _calls_match(a, b):
    """Check if two lists of function calls are equivalent (same names + args, order-independent)."""
    if len(a) != len(b):
        return False
    used = set()
    for call_a in a:
        found = False
        for i, call_b in enumerate(b):
            if i in used:
                continue
            if call_a.get("name") != call_b.get("name"):
                continue
            args_a = call_a.get("arguments", {})
            args_b = call_b.get("arguments", {})
            if all(
                str(args_a.get(k, "")).strip().lower() == str(v).strip().lower()
                for k, v in args_b.items()
            ) and all(
                str(args_b.get(k, "")).strip().lower() == str(v).strip().lower()
                for k, v in args_a.items()
            ):
                used.add(i)
                found = True
                break
        if not found:
            return False
    return True


def _estimate_complexity(messages, tools):
    """Estimate task complexity from input signals. Returns 'easy', 'medium', or 'hard'."""
    user_text = " ".join(m["content"] for m in messages if m["role"] == "user")
    num_tools = len(tools)

    # Count conjunctions/commas that suggest multi-call intent
    multi_signals = sum(1 for w in [" and ", " then ", ", and ", ","] if w in user_text.lower())

    if multi_signals >= 2 or (multi_signals >= 1 and num_tools >= 4):
        return "hard"
    elif num_tools >= 3 or multi_signals >= 1:
        return "medium"
    return "easy"


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """
    Self-consistency voting with adaptive thresholds.

    - High confidence on first run → trust immediately (fast path)
    - Gray zone → run again, check consensus
    - Low confidence or no consensus → fall back to cloud
    - Hard tasks (multi-call) get stricter thresholds
    """
    complexity = _estimate_complexity(messages, tools)

    # Adaptive thresholds based on complexity
    if complexity == "easy":
        high_thresh = 0.85
        low_thresh = 0.35
    elif complexity == "medium":
        high_thresh = 0.88
        low_thresh = 0.40
    else:  # hard
        high_thresh = 0.90
        low_thresh = 0.45

    # First local run
    local1 = generate_cactus(messages, tools)
    total_time = local1["total_time_ms"]

    # DEBUG: see what local actually returns
    user_text = messages[0]["content"] if messages else ""
    print(f"\n  [DBG] Q: {user_text[:80]}")
    print(f"  [DBG] complexity={complexity} conf={local1['confidence']:.4f} calls={json.dumps(local1['function_calls'])}")
    print(f"  [DBG] valid={_validate_calls(local1['function_calls'], tools, messages)}")

    # Fast path: very high confidence → trust it IF it validates
    if local1["confidence"] >= high_thresh and local1["function_calls"]:
        if _validate_calls(local1["function_calls"], tools, messages):
            local1["source"] = "on-device"
            return local1
        # High confidence but invalid structure → cloud
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback)"
        cloud["local_confidence"] = local1["confidence"]
        cloud["total_time_ms"] += total_time
        return cloud

    # Low confidence → skip consensus, go to cloud
    if local1["confidence"] < low_thresh or not local1["function_calls"]:
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback)"
        cloud["local_confidence"] = local1["confidence"]
        cloud["total_time_ms"] += total_time
        return cloud

    # Gray zone: run a second time for consensus
    local2 = generate_cactus(messages, tools)
    total_time += local2["total_time_ms"]

    if local2["function_calls"] and _calls_match(local1["function_calls"], local2["function_calls"]):
        best = local1 if local1["confidence"] >= local2["confidence"] else local2
        # Consensus reached — but still validate structure
        if _validate_calls(best["function_calls"], tools, messages):
            best["source"] = "on-device"
            best["total_time_ms"] = total_time
            return best
        # Consensus but invalid → cloud
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback)"
        cloud["local_confidence"] = best["confidence"]
        cloud["total_time_ms"] += total_time
        return cloud

    # No consensus — try a third run as tiebreaker
    local3 = generate_cactus(messages, tools)
    total_time += local3["total_time_ms"]

    # Check if any two of three agree
    candidates = [local1, local2, local3]
    for i in range(3):
        for j in range(i + 1, 3):
            if (candidates[i]["function_calls"] and candidates[j]["function_calls"]
                    and _calls_match(candidates[i]["function_calls"], candidates[j]["function_calls"])):
                best = candidates[i] if candidates[i]["confidence"] >= candidates[j]["confidence"] else candidates[j]
                if _validate_calls(best["function_calls"], tools, messages):
                    best["source"] = "on-device"
                    best["total_time_ms"] = total_time
                    return best

    # No valid agreement among 3 runs → cloud fallback
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = max(c["confidence"] for c in candidates)
    cloud["total_time_ms"] += total_time
    return cloud


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
