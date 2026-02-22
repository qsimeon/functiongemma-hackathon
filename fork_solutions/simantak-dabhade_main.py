# Source: https://github.com/simantak-dabhade/functiongemma-hackathon

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
    """Adaptive hybrid router with schema-aware normalization and fallback."""
    import re

    def tool_index(tool_defs):
        return {t["name"]: t for t in tool_defs}

    def parse_int(value):
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            digits = "".join(ch for ch in value if ch.isdigit())
            if digits:
                return int(digits)
        return value

    def normalize_time_string(value):
        if not isinstance(value, str):
            return value
        raw = value.strip().lower().replace(".", "")
        # Supports variants like "3pm", "3:00 pm", "15:30"
        ampm = None
        if raw.endswith("am"):
            ampm = "AM"
            raw = raw[:-2].strip()
        elif raw.endswith("pm"):
            ampm = "PM"
            raw = raw[:-2].strip()

        if ":" in raw:
            hour_str, minute_str = raw.split(":", 1)
        else:
            hour_str, minute_str = raw, "00"

        if not hour_str.isdigit() or not minute_str.isdigit():
            return value

        hour = int(hour_str)
        minute = int(minute_str)
        if minute < 0 or minute > 59:
            return value

        if ampm is None:
            if hour == 0:
                hour_12, ampm = 12, "AM"
            elif 1 <= hour < 12:
                hour_12, ampm = hour, "AM"
            elif hour == 12:
                hour_12, ampm = 12, "PM"
            elif 13 <= hour <= 23:
                hour_12, ampm = hour - 12, "PM"
            else:
                return value
        else:
            if hour < 1 or hour > 12:
                return value
            hour_12 = hour

        return f"{hour_12}:{minute:02d} {ampm}"

    def maybe_alarm_from_time(args):
        if not isinstance(args, dict):
            return args
        if "hour" in args and "minute" in args:
            args["hour"] = parse_int(args["hour"])
            args["minute"] = parse_int(args["minute"])
            return args
        if "time" not in args or not isinstance(args["time"], str):
            return args

        normalized = normalize_time_string(args["time"])
        if not isinstance(normalized, str) or " " not in normalized or ":" not in normalized:
            return args
        clock, ampm = normalized.split(" ", 1)
        hour_str, minute_str = clock.split(":", 1)
        hour = int(hour_str)
        minute = int(minute_str)
        ampm = ampm.upper()
        if ampm == "PM" and hour != 12:
            hour += 12
        if ampm == "AM" and hour == 12:
            hour = 0
        args["hour"] = hour
        args["minute"] = minute
        return args

    def normalize_call(call, idx):
        name = call.get("name")
        args = dict(call.get("arguments", {}) or {})
        spec = idx.get(name, {})
        params = spec.get("parameters", {})
        properties = params.get("properties", {})

        if name == "set_alarm":
            args = maybe_alarm_from_time(args)
        if name == "create_reminder" and "time" in args:
            args["time"] = normalize_time_string(args["time"])
            if "title" in args and isinstance(args["title"], str):
                title = args["title"].strip().lower()
                for prefix in ["about ", "the ", "to "]:
                    if title.startswith(prefix) and len(title) > len(prefix):
                        title = title[len(prefix):]
                args["title"] = title

        for key, prop in properties.items():
            if key not in args:
                continue
            if prop.get("type") == "integer":
                args[key] = parse_int(args[key])
            elif prop.get("type") == "string" and isinstance(args[key], str):
                args[key] = args[key].strip()

        return {"name": name, "arguments": args}

    def normalize_calls(calls, idx):
        return [normalize_call(c, idx) for c in (calls or [])]

    def is_valid_against_schema(calls, idx):
        if not calls:
            return False
        for call in calls:
            name = call.get("name")
            if name not in idx:
                return False
            args = call.get("arguments", {}) or {}
            params = idx[name].get("parameters", {})
            required = params.get("required", [])
            properties = params.get("properties", {})

            for req in required:
                if req not in args:
                    return False
            for key, val in args.items():
                if key not in properties:
                    continue
                expected_type = properties[key].get("type")
                if expected_type == "integer" and not isinstance(val, int):
                    return False
                if expected_type == "string" and not isinstance(val, str):
                    return False
        return True

    def estimate_complexity(msgs, available_tools):
        user_text = " ".join(
            m.get("content", "") for m in msgs if m.get("role") == "user"
        ).lower()
        connectors = [" and ", ", and ", " then ", " also ", " plus "]
        multi_intent = any(c in user_text for c in connectors)
        if multi_intent and len(available_tools) >= 4:
            return "hard"
        if multi_intent or len(available_tools) >= 4:
            return "medium"
        return "easy"

    def heuristic_calls(msgs, idx):
        user_text = " ".join(
            m.get("content", "") for m in msgs if m.get("role") == "user"
        )
        text = user_text.lower()
        found = []
        last_contact = None

        def add_call(position, name, arguments):
            if name in idx:
                found.append((position, {"name": name, "arguments": arguments}))

        # search_contacts
        for m in re.finditer(r"(find|look up)\s+([a-z][a-z\s'-]{0,40}?)\s+in my contacts", text):
            query = m.group(2).strip(" .,!?\t\n\r")
            if query:
                last_contact = query
                add_call(m.start(), "search_contacts", {"query": query.title()})

        # send_message
        for m in re.finditer(r"(?:send (?:a )?message to|text)\s+([a-z][a-z'-]{0,30})\s+saying\s+(.+?)(?=,|\.\s|\.| and |$)", text):
            recipient = m.group(1).strip()
            if recipient in {"him", "her", "them"} and last_contact:
                recipient = last_contact
            message = m.group(2).strip(" .,!?\t\n\r")
            if recipient and message:
                add_call(
                    m.start(),
                    "send_message",
                    {"recipient": recipient.title(), "message": message},
                )

        # weather
        for m in re.finditer(r"weather(?:\s+like)?\s+in\s+([a-z][a-z\s'-]{0,40}?)(?=,|\.|\?| and |$)", text):
            location = m.group(1).strip(" .,!?\t\n\r")
            if location:
                add_call(m.start(), "get_weather", {"location": location.title()})

        # alarm
        for m in re.finditer(r"(?:set (?:an )?alarm for|wake me up at)\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm))", text):
            normalized = normalize_time_string(m.group(1))
            if isinstance(normalized, str) and " " in normalized and ":" in normalized:
                clock, ampm = normalized.split(" ", 1)
                hour_str, minute_str = clock.split(":", 1)
                hour = int(hour_str)
                minute = int(minute_str)
                if ampm == "PM" and hour != 12:
                    hour += 12
                if ampm == "AM" and hour == 12:
                    hour = 0
                add_call(m.start(), "set_alarm", {"hour": hour, "minute": minute})

        # timer
        for m in re.finditer(r"(?:set (?:a )?(?:countdown )?timer for|set a)\s*(\d{1,3})\s*(?:minute|minutes|min)\s*timer?", text):
            minutes = int(m.group(1))
            add_call(m.start(), "set_timer", {"minutes": minutes})

        # music
        for m in re.finditer(r"play\s+(?:some\s+)?(.+?)(?=,|\.|\?| and |$)", text):
            song = m.group(1).strip(" .,!?\t\n\r")
            if song.endswith(" music") and len(song) > len(" music"):
                song = song[:-6]
            if song:
                add_call(m.start(), "play_music", {"song": song})

        # reminder
        for m in re.finditer(r"remind me(?:\s+(?:about|to))?\s+(.+?)\s+at\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm))", text):
            title = m.group(1).strip(" .,!?\t\n\r")
            time_str = normalize_time_string(m.group(2))
            for prefix in ["about ", "the ", "to "]:
                if title.startswith(prefix) and len(title) > len(prefix):
                    title = title[len(prefix):]
            if title and isinstance(time_str, str):
                add_call(m.start(), "create_reminder", {"title": title, "time": time_str})

        found.sort(key=lambda x: x[0])
        deduped = []
        seen = set()
        for _, call in found:
            signature = (call["name"], json.dumps(call["arguments"], sort_keys=True))
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(call)
        return normalize_calls(deduped, idx)

    idx = tool_index(tools)
    local = generate_cactus(messages, tools)
    local["function_calls"] = normalize_calls(local.get("function_calls"), idx)
    heuristic = heuristic_calls(messages, idx)

    complexity = estimate_complexity(messages, tools)
    dynamic_threshold = {"easy": 0.62, "medium": 0.76, "hard": 0.86}[complexity]
    effective_threshold = min(confidence_threshold, dynamic_threshold)

    local_valid = is_valid_against_schema(local.get("function_calls"), idx)
    local_call_count = len(local.get("function_calls", []))
    should_require_more_calls = complexity == "hard"
    enough_calls = (not should_require_more_calls) or (local_call_count >= 2)

    if local_valid and enough_calls and local.get("confidence", 0) >= effective_threshold:
        local["source"] = "on-device"
        return local

    heuristic_valid = is_valid_against_schema(heuristic, idx)
    if heuristic_valid:
        heuristic_result = {
            "function_calls": heuristic,
            "total_time_ms": local.get("total_time_ms", 0),
            "source": "on-device",
            "confidence": local.get("confidence", 0),
        }
        # Prefer deterministic parsing when local is weak or missing expected multi-call shape.
        if (not local_valid) or (local.get("confidence", 0) < effective_threshold) or (
            should_require_more_calls and len(heuristic) > local_call_count
        ):
            return heuristic_result

    # If cloud is unavailable, keep local output instead of crashing.
    if not os.environ.get("GEMINI_API_KEY"):
        local["source"] = "on-device"
        return local

    try:
        cloud = generate_cloud(messages, tools)
        cloud["function_calls"] = normalize_calls(cloud.get("function_calls"), idx)
        cloud["source"] = "cloud (fallback)"
        cloud["local_confidence"] = local.get("confidence", 0)
        cloud["total_time_ms"] += local.get("total_time_ms", 0)
        return cloud
    except Exception:
        local["source"] = "on-device"
        return local


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
