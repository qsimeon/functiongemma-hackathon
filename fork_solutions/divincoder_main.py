# Source: https://github.com/divincoder/functiongemma-hackathon

import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
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


############## Hybrid Routing ##############

# Warm model: load once, reuse across calls
_warm_model = None


def _get_model():
    global _warm_model
    if _warm_model is None:
        _warm_model = cactus_init(functiongemma_path)
    return _warm_model


def _on_device_call(messages, tools, tool_rag_top_k=None, extra_system=None, temperature=None):
    """Run a single on-device inference using the warm model."""
    model = _get_model()
    cactus_reset(model)

    cactus_tools = [{"type": "function", "function": t} for t in tools]
    system_prompt = "You are a helpful assistant that can use tools."
    if extra_system:
        system_prompt += " " + extra_system
    kwargs = dict(
        force_tools=True,
        max_tokens=512,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
        confidence_threshold=0.01,
    )
    if tool_rag_top_k is not None:
        kwargs["tool_rag_top_k"] = tool_rag_top_k
    if temperature is not None:
        kwargs["temperature"] = temperature

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": system_prompt}] + messages,
        tools=cactus_tools,
        **kwargs,
    )

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {"function_calls": [], "total_time_ms": 0, "confidence": 0}

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


def _fix_args(calls, tools):
    """Coerce argument types toward tool schemas to reduce invalid calls."""
    tool_map = {t["name"]: t for t in tools}

    def _clean_reminder_title(text):
        if not isinstance(text, str):
            return text
        lower = text.lower()
        # Extract between about/to ... at ... if present
        import re
        m = re.search(r"(?:about|to) (.+?) at ", lower)
        if m:
            return m.group(1).strip()
        m = re.search(r"(?:about|to) (.+)", lower)
        if m:
            return m.group(1).strip()
        return text.strip()

    def _coerce(prop, value):
        ptype = prop.get("type")
        if ptype == "integer":
            try:
                return abs(int(float(value)))
            except (ValueError, TypeError):
                return value
        if ptype == "number":
            try:
                return float(value)
            except (ValueError, TypeError):
                return value
        if ptype == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lower = value.strip().lower()
                if lower in ["true", "yes", "1"]:
                    return True
                if lower in ["false", "no", "0"]:
                    return False
            return value
        # Enum coercion: case-insensitive match
        if "enum" in prop and isinstance(value, str):
            for option in prop["enum"]:
                if value.lower() == str(option).lower():
                    return option
        return value

    def _strip_articles(text):
        for prefix in ["the ", "a ", "an ", "my "]:
            if text.startswith(prefix):
                return text[len(prefix):]
        return text

    def _strip_time_suffix(text):
        import re
        return re.sub(r"\\s+at\\s+\\d{1,2}(:\\d{2})?\\s*(am|pm)?", "", text, flags=re.IGNORECASE).strip()

    def _strip_punct(text):
        import re
        return re.sub(r"[\\.,;:!]+$", "", text).strip()

    for call in calls:
        if call.get("name") in tool_map:
            props = tool_map[call["name"]]["parameters"].get("properties", {})
            for k, v in list(call.get("arguments", {}).items()):
                if k in props:
                    call["arguments"][k] = _coerce(props[k], v)
            # Reminder title cleanup for better exact-match
            if call.get("name") == "create_reminder" and "title" in call.get("arguments", {}):
                title = _clean_reminder_title(call["arguments"]["title"])
                if isinstance(title, str):
                    title = _strip_articles(_strip_time_suffix(_strip_punct(title.strip().lower())))
                call["arguments"]["title"] = title


def _valid_calls(calls, tools):
    """Check all function calls have valid names and required params."""
    tool_map = {t["name"]: t for t in tools}
    for call in calls:
        name = call.get("name", "")
        if name not in tool_map:
            return False
        required = tool_map[name]["parameters"].get("required", [])
        args = call.get("arguments", {})
        if any(r not in args for r in required):
            return False
    return True


def _has_all_required(calls, tools):
    """Return True only if every call includes all required args."""
    tool_map = {t["name"]: t for t in tools}
    for call in calls:
        name = call.get("name", "")
        if name not in tool_map:
            return False
        required = tool_map[name]["parameters"].get("required", [])
        args = call.get("arguments", {})
        if any(r not in args for r in required):
            return False
    return True


def _split_actions(text):
    """Split a multi-action query into individual action segments."""
    if ", and " in text:
        last_split = text.rsplit(", and ", 1)
        segments = last_split[0].split(", ")
        segments.append(last_split[1])
    elif " and " in text:
        segments = text.split(" and ")
    else:
        segments = [text]
    return [s.strip().rstrip(".") for s in segments if len(s.strip()) > 3]


def _is_multi_action(text):
    """Check if text likely contains multiple action requests."""
    lower = text.lower()
    return " and " in lower or lower.count(",") > 1


def generate_hybrid(messages, tools, confidence_threshold=0.7):
    """Hybrid inference: on-device with structural validation, cloud fallback."""
    start = time.time()
    user_text = " ".join(m["content"] for m in messages if m["role"] == "user")
    multi = _is_multi_action(user_text) and len(tools) > 1
    segments = _split_actions(user_text) if multi else None

    # Multi-action fallback: decompose into single-action sub-queries
    if multi:
        all_calls = []
        all_ok = True
        missing_segments = []
        for segment in segments:
            sub = _on_device_call(
                [{"role": "user", "content": segment}],
                tools,
                tool_rag_top_k=0,
                extra_system=f"Available tools: {', '.join(t['name'] for t in tools)}. Use exactly one tool call that best satisfies this single instruction. Fill all required arguments. No prose.",
                temperature=0.2,
            )
            _fix_args(sub["function_calls"], tools)
            if sub["function_calls"] and _valid_calls(sub["function_calls"], tools):
                all_calls.extend(sub["function_calls"])
            else:
                all_ok = False
                missing_segments.append(segment)
        if all_ok and all_calls and len(all_calls) >= len(segments) and _has_all_required(all_calls, tools):
            return {
                "function_calls": all_calls,
                "total_time_ms": (time.time() - start) * 1000,
                "source": "on-device",
            }
        # Partial success: try cloud to fill missing intents, merge results with distinctness
        if all_calls and missing_segments:
            cloud_partial = generate_cloud(messages, tools)
            _fix_args(cloud_partial["function_calls"], tools)
            merged = all_calls.copy()
            for c in cloud_partial["function_calls"]:
                if not any((c.get("name")==m.get("name") and c.get("arguments")==m.get("arguments")) for m in merged):
                    merged.append(c)
            if _valid_calls(merged, tools) and _has_all_required(merged, tools) and len(merged) >= len(segments):
                return {
                    "function_calls": merged,
                    "total_time_ms": (time.time() - start) * 1000,
                    "source": "hybrid-merge",
                }
        # If still missing, fall back to cloud directly for multi-action
        cloud = generate_cloud(messages, tools)
        _fix_args(cloud["function_calls"], tools)
        cloud["source"] = "cloud (fallback)"
        cloud["total_time_ms"] = (time.time() - start) * 1000
        cloud["local_confidence"] = 0
        return cloud

    # Single-intent path: one fast local try, then cloud based on confidence
    result = _on_device_call(messages, tools, tool_rag_top_k=None)
    _fix_args(result["function_calls"], tools)

    if result["function_calls"] and _valid_calls(result["function_calls"], tools):
        return {
            "function_calls": result["function_calls"],
            "total_time_ms": (time.time() - start) * 1000,
            "source": "on-device",
        }

    # If only one tool is available and local failed or was low confidence, defer to cloud for generality.
    if len(tools) == 1 and result.get("confidence", 0) < 1.0:
        cloud = generate_cloud(messages, tools)
        _fix_args(cloud["function_calls"], tools)
        cloud["source"] = "cloud (fallback)"
        cloud["total_time_ms"] = (time.time() - start) * 1000
        cloud["local_confidence"] = result.get("confidence", 0)
        return cloud

    # Narrowed retry for ambiguous single-intent tool choice
    narrowed_retry = _on_device_call(
        messages,
        tools,
        tool_rag_top_k=min(2, len(tools)),
        extra_system="Select the single best tool and return one valid function call with all required arguments. No prose.",
        temperature=0.35,
    )
    _fix_args(narrowed_retry["function_calls"], tools)
    if narrowed_retry["function_calls"] and _valid_calls(narrowed_retry["function_calls"], tools):
        return {
            "function_calls": narrowed_retry["function_calls"],
            "total_time_ms": (time.time() - start) * 1000,
            "source": "on-device",
        }

    if result.get("confidence", 0) >= confidence_threshold:
        retry = _on_device_call(
            messages,
            tools,
            tool_rag_top_k=0,
            extra_system="Always reply with a single function call JSON using only provided tools and all required arguments. No prose.",
            temperature=0.2,
        )
        _fix_args(retry["function_calls"], tools)
        if retry["function_calls"] and _valid_calls(retry["function_calls"], tools):
            return {
                "function_calls": retry["function_calls"],
                "total_time_ms": (time.time() - start) * 1000,
                "source": "on-device",
            }

    cloud = generate_cloud(messages, tools)
    _fix_args(cloud["function_calls"], tools)
    cloud["source"] = "cloud (fallback)"
    cloud["total_time_ms"] = (time.time() - start) * 1000
    cloud["local_confidence"] = result.get("confidence", 0)

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
