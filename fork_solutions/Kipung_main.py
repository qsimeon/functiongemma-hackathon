# Source: https://github.com/Kipung/functiongemma-hackathon

import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, re, time, threading
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types

try:
    from cactus import cactus_reset
except ImportError:
    def cactus_reset(_model):
        return None


_CLOUD_LOCK = threading.Lock()
_LAST_CLOUD_TS = 0.0
_CLOUD_MIN_INTERVAL_S = 1.25
_CLOUD_MAX_RETRIES = 2

_LOCAL_MODEL = None
_LOCAL_MODEL_LOCK = threading.Lock()


def _sleep_for_cloud_rate_limit(min_interval_s=_CLOUD_MIN_INTERVAL_S):
    global _LAST_CLOUD_TS
    with _CLOUD_LOCK:
        now = time.time()
        wait = (_LAST_CLOUD_TS + min_interval_s) - now
        if wait > 0:
            time.sleep(wait)
        _LAST_CLOUD_TS = time.time()


def _is_retryable_cloud_error(error):
    text = str(error)
    return (
        "429" in text
        or "RESOURCE_EXHAUSTED" in text
        or "rate" in text.lower()
        or "temporarily" in text.lower()
    )


def _get_local_model():
    global _LOCAL_MODEL
    with _LOCAL_MODEL_LOCK:
        if _LOCAL_MODEL is None:
            _LOCAL_MODEL = cactus_init(functiongemma_path)
        return _LOCAL_MODEL


def _extract_user_text(messages):
    parts = [m.get("content", "") for m in messages if m.get("role") == "user"]
    return " ".join(parts).strip().lower()


def _looks_multi_intent(user_text):
    if not user_text:
        return False
    markers = [" and ", " then ", " also ", ", and "]
    return any(marker in user_text for marker in markers)


def _extract_alarm_from_text(user_text):
    match = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", user_text, flags=re.IGNORECASE)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    return {"hour": hour, "minute": minute}


def _extract_timer_from_text(user_text):
    match = re.search(r"(\d+)\s*minute", user_text, flags=re.IGNORECASE)
    if not match:
        return None
    return {"minutes": int(match.group(1))}


def _extract_reminder_fields(user_text):
    anchored_match = re.search(
        r"remind me (?:about|to)\s+(.*?)\s+at\s+(\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm))",
        user_text,
        flags=re.IGNORECASE,
    )

    fields = {}
    if anchored_match:
        title = anchored_match.group(1).strip().strip(".?!")
        if title.lower().startswith("the "):
            title = title[4:]
        fields["title"] = title

        time_text = anchored_match.group(2).upper().replace("AM", " AM").replace("PM", " PM").strip()
        time_text = re.sub(r"\s+", " ", time_text)
        fields["time"] = time_text
        return fields

    time_match = re.search(r"(\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm))", user_text)
    if time_match:
        time_text = time_match.group(1).upper().replace("AM", " AM").replace("PM", " PM").strip()
        time_text = re.sub(r"\s+", " ", time_text)
        fields["time"] = time_text

    title_match = re.search(r"remind me (?:about|to)\s+(.+)$", user_text, flags=re.IGNORECASE)
    if title_match:
        title = title_match.group(1).strip().strip(".?!")
        fields["title"] = re.sub(r"\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)\b", "", title, flags=re.IGNORECASE).strip()

    return fields


def _extract_send_message_fields(user_text):
    recipient = None
    message = None

    direct_recipient = re.search(r"(?:send (?:a )?message to|text)\s+([a-zA-Z]+)", user_text, flags=re.IGNORECASE)
    if direct_recipient:
        recipient = direct_recipient.group(1)
    else:
        contact_ref = re.search(r"(?:find|look up|search for)\s+([a-zA-Z]+)", user_text, flags=re.IGNORECASE)
        pronoun_send = re.search(r"send (?:him|her) (?:a )?message", user_text, flags=re.IGNORECASE)
        if contact_ref and pronoun_send:
            recipient = contact_ref.group(1)

    message_match = re.search(r"(?:saying|that says)\s+(.+?)(?:\.|$)", user_text, flags=re.IGNORECASE)
    if message_match:
        message = message_match.group(1)

    return {"recipient": recipient, "message": message}


def _extract_search_query(user_text):
    match = re.search(r"(?:find|look up|search(?: for)?)\s+([a-zA-Z]+)", user_text, flags=re.IGNORECASE)
    return match.group(1) if match else None


def _extract_weather_location(user_text):
    match = re.search(r"weather(?:\s+in)?\s+([a-zA-Z\s]+?)(?:\?|\.|,| and|$)", user_text, flags=re.IGNORECASE)
    if not match:
        return None
    return _clean_string_value(match.group(1))


def _extract_music_title(user_text):
    match = re.search(r"play\s+(.+?)(?:\.|,| and|$)", user_text, flags=re.IGNORECASE)
    if not match:
        return None
    return _clean_string_value(match.group(1))


def _has_human_time_string(value):
    if not isinstance(value, str):
        return False
    return bool(re.search(r"\b\d{1,2}(?::\d{2})?\s*(am|pm)\b", value, flags=re.IGNORECASE))


def _augment_calls_from_intent(function_calls, tools, user_text):
    if not _looks_multi_intent(user_text):
        return function_calls

    existing_names = {call.get("name") for call in function_calls if isinstance(call, dict)}
    tool_names = {tool.get("name") for tool in tools}
    augmented = list(function_calls)

    wants_alarm = bool(re.search(r"\b(alarm|wake me up|wake me)\b", user_text, flags=re.IGNORECASE))
    wants_timer = bool(re.search(r"\btimer\b", user_text, flags=re.IGNORECASE))
    wants_weather = bool(re.search(r"\bweather\b", user_text, flags=re.IGNORECASE))
    wants_message = bool(re.search(r"\b(send|text)\b", user_text, flags=re.IGNORECASE))
    wants_search = bool(re.search(r"\b(find|look up|search)\b", user_text, flags=re.IGNORECASE))
    wants_reminder = bool(re.search(r"\bremind me\b", user_text, flags=re.IGNORECASE))
    wants_music = bool(re.search(r"\bplay\b", user_text, flags=re.IGNORECASE))

    if wants_search and "search_contacts" in tool_names and "search_contacts" not in existing_names:
        query = _extract_search_query(user_text)
        if query:
            augmented.append({"name": "search_contacts", "arguments": {"query": query}})

    if wants_message and "send_message" in tool_names and "send_message" not in existing_names:
        msg = _extract_send_message_fields(user_text)
        if msg.get("recipient") and msg.get("message"):
            augmented.append({
                "name": "send_message",
                "arguments": {
                    "recipient": msg["recipient"],
                    "message": _trim_message_clause(msg["message"]),
                },
            })

    if wants_weather and "get_weather" in tool_names and "get_weather" not in existing_names:
        location = _extract_weather_location(user_text)
        if location:
            augmented.append({"name": "get_weather", "arguments": {"location": location}})

    if wants_alarm and "set_alarm" in tool_names and "set_alarm" not in existing_names:
        alarm = _extract_alarm_from_text(user_text)
        if alarm:
            augmented.append({"name": "set_alarm", "arguments": alarm})

    if wants_timer and "set_timer" in tool_names and "set_timer" not in existing_names:
        timer = _extract_timer_from_text(user_text)
        if timer:
            augmented.append({"name": "set_timer", "arguments": timer})

    if wants_reminder and "create_reminder" in tool_names and "create_reminder" not in existing_names:
        reminder = _extract_reminder_fields(user_text)
        if reminder.get("title") and reminder.get("time"):
            augmented.append({
                "name": "create_reminder",
                "arguments": {
                    "title": reminder["title"],
                    "time": reminder["time"],
                },
            })

    if wants_music and "play_music" in tool_names and "play_music" not in existing_names:
        song = _extract_music_title(user_text)
        if song:
            augmented.append({"name": "play_music", "arguments": {"song": song}})

    return augmented


def _clean_string_value(value):
    if value is None:
        return None
    text = str(value).strip()
    text = text.strip('"\'`')
    text = re.sub(r"[\s\.,!?;:]+$", "", text)
    return text


def _trim_message_clause(text):
    if not text:
        return text
    split_patterns = [
        r",\s*and\s+(?:check|get|set|play|search|find|look up|remind)\b",
        r"\s+and\s+(?:check|get|set|play|search|find|look up|remind)\b",
        r",\s*(?:check|get|set|play|search|find|look up|remind)\b",
    ]
    trimmed = text
    for pattern in split_patterns:
        parts = re.split(pattern, trimmed, maxsplit=1, flags=re.IGNORECASE)
        trimmed = parts[0]
    return _clean_string_value(trimmed)


def _repair_tool_calls(function_calls, user_text):
    if not isinstance(function_calls, list):
        return []

    repaired = []
    wants_alarm = bool(re.search(r"\b(alarm|wake me up|wake me)\b", user_text, flags=re.IGNORECASE))
    alarm_guess = _extract_alarm_from_text(user_text)
    timer_guess = _extract_timer_from_text(user_text)
    reminder_guess = _extract_reminder_fields(user_text)
    message_guess = _extract_send_message_fields(user_text)

    for call in function_calls:
        name = call.get("name")
        args = dict(call.get("arguments", {}) or {})

        if name == "set_alarm" and not wants_alarm:
            continue

        if name == "set_alarm" and alarm_guess:
            hour = args.get("hour")
            minute = args.get("minute")
            if not isinstance(hour, int) or hour < 0 or hour > 23:
                args["hour"] = alarm_guess["hour"]
            if not isinstance(minute, int) or minute < 0 or minute > 59:
                args["minute"] = alarm_guess["minute"]

        if name == "set_timer" and timer_guess:
            minutes = args.get("minutes")
            if not isinstance(minutes, int) or minutes <= 0:
                args["minutes"] = timer_guess["minutes"]

        if name == "create_reminder" and reminder_guess:
            if reminder_guess.get("title"):
                args["title"] = reminder_guess["title"]
            if reminder_guess.get("time"):
                args["time"] = reminder_guess["time"]

        if name == "send_message" and message_guess:
            if message_guess.get("recipient"):
                args["recipient"] = message_guess["recipient"]
            if message_guess.get("message"):
                args["message"] = _trim_message_clause(message_guess["message"])

        for key, value in list(args.items()):
            if isinstance(value, str):
                args[key] = _clean_string_value(value)

        repaired.append({"name": name, "arguments": args})

    return repaired


def _required_args_by_tool(tools):
    required = {}
    for tool in tools:
        params = tool.get("parameters", {})
        required[tool.get("name")] = set(params.get("required", []))
    return required


def _tool_specs_by_name(tools):
    specs = {}
    for tool in tools:
        specs[tool.get("name")] = tool.get("parameters", {})
    return specs


def _coerce_value_to_schema(value, schema_type):
    if schema_type == "integer":
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            digits = "".join(char for char in value if char.isdigit())
            if digits:
                return int(digits)
        return None

    if schema_type == "number":
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None

    if schema_type == "string":
        if value is None:
            return None
        return str(value)

    if schema_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "1"}:
                return True
            if lowered in {"false", "no", "0"}:
                return False
        return None

    return value


def _normalize_and_validate_tool_calls(function_calls, tools):
    if not tools:
        return function_calls or [], True
    if not isinstance(function_calls, list) or not function_calls:
        return [], False

    required = _required_args_by_tool(tools)
    specs = _tool_specs_by_name(tools)
    tool_names = set(specs.keys())
    normalized_calls = []

    for call in function_calls:
        name = call.get("name")
        args = call.get("arguments")
        if name not in tool_names or not isinstance(args, dict):
            return [], False

        required_args = required.get(name, set())
        if required_args - set(args.keys()):
            return [], False

        properties = specs.get(name, {}).get("properties", {})
        normalized_args = {}
        for arg_name, arg_value in args.items():
            schema = properties.get(arg_name)
            if not schema:
                normalized_args[arg_name] = arg_value
                continue

            coerced = _coerce_value_to_schema(arg_value, schema.get("type", "").lower())
            if coerced is None and arg_name in required_args:
                return [], False
            normalized_args[arg_name] = coerced if coerced is not None else arg_value

        normalized_calls.append({"name": name, "arguments": normalized_args})

    return normalized_calls, True


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = _get_local_model()
    cactus_reset(model)

    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a tool-calling assistant. Return function calls only, and include all required calls and arguments needed to satisfy the user request."}] + messages,
        tools=cactus_tools,
        force_tools=True,
        tool_rag_top_k=0,
        temperature=0,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
            "success": False,
            "cloud_handoff": True,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
        "success": raw.get("success", True),
        "cloud_handoff": raw.get("cloud_handoff", False),
    }


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    # client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    

    client = genai.Client(
        vertexai=True,
        project="cactushackathon-488119",  # from your screenshot
        location="us-central1",
    )

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
    last_error = None

    for attempt in range(_CLOUD_MAX_RETRIES + 1):
        try:
            _sleep_for_cloud_rate_limit()
            gemini_response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=gemini_tools,
                    temperature=0,
                    system_instruction=(
                        "You are a strict tool-calling router. "
                        "Return function calls only. "
                        "If multiple user intents are present, emit all required calls. "
                        "Always provide all required arguments with correct types."
                    ),
                ),
            )
            break
        except Exception as error:
            last_error = error
            if attempt >= _CLOUD_MAX_RETRIES or not _is_retryable_cloud_error(error):
                raise
            time.sleep(min(2.0, 0.6 * (2 ** attempt)))
    else:
        if last_error:
            raise last_error

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
    """Baseline hybrid inference strategy; fall back to cloud if Cactus Confidence is below threshold."""
    local = generate_cactus(messages, tools)

    tool_count = len(tools or [])
    adaptive_threshold = 0.82 if tool_count <= 2 else (0.76 if tool_count == 3 else 0.70)
    local_threshold = min(confidence_threshold, adaptive_threshold)
    user_text = _extract_user_text(messages)
    looks_multi_intent = _looks_multi_intent(user_text)

    local_raw_calls = _repair_tool_calls(local.get("function_calls", []), user_text)
    local_raw_calls = _augment_calls_from_intent(local_raw_calls, tools, user_text)
    local_calls, local_calls_valid = _normalize_and_validate_tool_calls(local_raw_calls, tools)
    local["function_calls"] = local_calls
    local_confident = local.get("confidence", 0) >= local_threshold
    local_handoff = bool(local.get("cloud_handoff", False))
    local_call_count = len(local_calls)

    missing_multi_call_signal = looks_multi_intent and local_call_count < 2
    strict_slots = any(token in user_text for token in ["alarm", "timer", "remind", "message", "text"])
    low_conf_for_strict = strict_slots and local.get("confidence", 0) < 0.90

    if local_calls_valid and local_confident and not local_handoff and not missing_multi_call_signal and not low_conf_for_strict:
        local["source"] = "on-device"
        return local

    try:
        cloud = generate_cloud(messages, tools)
        cloud_raw_calls = _repair_tool_calls(cloud.get("function_calls", []), user_text)
        cloud_raw_calls = _augment_calls_from_intent(cloud_raw_calls, tools, user_text)
        cloud_calls, cloud_calls_valid = _normalize_and_validate_tool_calls(cloud_raw_calls, tools)
        cloud["function_calls"] = cloud_calls
        if cloud_calls_valid:
            cloud["source"] = "cloud (fallback)"
            cloud["local_confidence"] = local.get("confidence", 0)
            cloud["total_time_ms"] += local.get("total_time_ms", 0)
            return cloud
    except Exception:
        pass

    local["source"] = "on-device (cloud-unavailable)"
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

    if _LOCAL_MODEL is not None:
        cactus_destroy(_LOCAL_MODEL)
