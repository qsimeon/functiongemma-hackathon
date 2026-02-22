# Source: https://github.com/psh12320/functiongemma-hackathon
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import atexit
import json
import os
import re
import threading
import time
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset

_TIME_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", flags=re.IGNORECASE)
_MODEL_LOCK = threading.Lock()
_MODEL_RUN_LOCK = threading.Lock()
_WARM_MODEL = None
_SESSION_LOCK = threading.Lock()
_SESSION_STATE = {
    "config_key": None,
    "messages": [],
    "turns": 0,
    "last_access": 0.0,
}


def _env_int(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _conversation_enabled():
    return os.environ.get("CACTUS_ENABLE_CONVERSATION", "1") != "0"


def _debug_session_log(message):
    if os.environ.get("CACTUS_DEBUG_SESSION") == "1":
        print(f"[cactus-session] {message}")


def _session_max_turns():
    return max(1, _env_int("CACTUS_SESSION_MAX_TURNS", 24))


def _session_idle_seconds():
    return max(1.0, _env_float("CACTUS_SESSION_IDLE_SECONDS", 180.0))


def _normalize_messages(messages):
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user")).strip().lower() or "user"
        content = str(message.get("content", ""))
        normalized.append({"role": role, "content": content})
    return normalized


def _tool_signature(tools):
    compact = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        parameters = tool.get("parameters", {})
        compact.append({
            "name": str(tool.get("name", "")),
            "required": list(parameters.get("required", [])),
            "properties": parameters.get("properties", {}),
        })
    return json.dumps(compact, sort_keys=True, separators=(",", ":"))


def _session_config_key(system_prompt, tools):
    return json.dumps({
        "system_prompt": str(system_prompt or ""),
        "tools": _tool_signature(tools),
    }, sort_keys=True, separators=(",", ":"))


def _messages_prefix_match(prefix, full):
    if len(prefix) > len(full):
        return False
    for idx, message in enumerate(prefix):
        if message.get("role") != full[idx].get("role"):
            return False
        if message.get("content") != full[idx].get("content"):
            return False
    return True


def _reset_session_state():
    with _SESSION_LOCK:
        _SESSION_STATE["config_key"] = None
        _SESSION_STATE["messages"] = []
        _SESSION_STATE["turns"] = 0
        _SESSION_STATE["last_access"] = 0.0


def _plan_session_input(config_key, normalized_messages):
    with _SESSION_LOCK:
        now = time.monotonic()
        current_messages = list(normalized_messages)

        if _SESSION_STATE["config_key"] != config_key:
            return "reset", current_messages

        if _SESSION_STATE["last_access"] > 0 and (now - _SESSION_STATE["last_access"]) > _session_idle_seconds():
            return "reset", current_messages

        if _SESSION_STATE["turns"] >= _session_max_turns():
            return "reset", current_messages

        previous_messages = _SESSION_STATE["messages"]
        if previous_messages and _messages_prefix_match(previous_messages, current_messages) and len(current_messages) > len(previous_messages):
            return "append", current_messages[len(previous_messages):]

        return "reset", current_messages


def _update_session_state(config_key, normalized_messages, mode):
    with _SESSION_LOCK:
        same_config = _SESSION_STATE["config_key"] == config_key
        if mode == "append" and same_config:
            turns = _SESSION_STATE["turns"] + 1
        else:
            turns = 1
        _SESSION_STATE["config_key"] = config_key
        _SESSION_STATE["messages"] = list(normalized_messages)
        _SESSION_STATE["turns"] = turns
        _SESSION_STATE["last_access"] = time.monotonic()


def _get_warm_model():
    global _WARM_MODEL
    with _MODEL_LOCK:
        if _WARM_MODEL is None:
            _WARM_MODEL = cactus_init(functiongemma_path)
        return _WARM_MODEL


def _destroy_warm_model():
    global _WARM_MODEL
    with _MODEL_LOCK:
        if _WARM_MODEL is not None:
            cactus_destroy(_WARM_MODEL)
            _WARM_MODEL = None
    _reset_session_state()


atexit.register(_destroy_warm_model)


def generate_cactus(messages, tools, system_prompt=None, max_tokens=256, temperature=0.0):
    """Run function calling on-device via FunctionGemma + Cactus."""
    disable_warm_model = os.environ.get("CACTUS_DISABLE_WARM_MODEL") == "1"
    conversational_mode = _conversation_enabled() and not disable_warm_model
    model = cactus_init(functiongemma_path) if disable_warm_model else _get_warm_model()
    normalized_messages = _normalize_messages(messages)
    prompt = system_prompt or "You are a helpful assistant that can use tools."
    config_key = _session_config_key(prompt, tools)
    session_mode = "reset"

    try:
        with _MODEL_RUN_LOCK:
            cactus_tools = [{"type": "function", "function": t} for t in tools]
            full_messages = [{"role": "system", "content": prompt}] + normalized_messages

            if disable_warm_model:
                cactus_reset(model)
                model_messages = full_messages
                _debug_session_log("mode=reset reason=disable_warm_model")
            elif conversational_mode:
                session_mode, model_messages = _plan_session_input(config_key, normalized_messages)
                if session_mode == "append" and model_messages:
                    # Continue the current model context by appending only new turns.
                    _debug_session_log(f"mode=append delta_messages={len(model_messages)}")
                else:
                    session_mode = "reset"
                    cactus_reset(model)
                    model_messages = full_messages
                    _debug_session_log("mode=reset reason=session_plan")
            else:
                cactus_reset(model)
                model_messages = full_messages
                _debug_session_log("mode=reset reason=conversation_disabled")

            raw_str = cactus_complete(
                model,
                model_messages,
                tools=cactus_tools,
                force_tools=True,
                tool_rag_top_k=0,
                temperature=temperature,
                max_tokens=max_tokens,
                stop_sequences=["<|im_end|>", "<end_of_turn>"],
            )

            if conversational_mode:
                _update_session_state(config_key, normalized_messages, session_mode)
            elif not disable_warm_model:
                _reset_session_state()
    finally:
        if disable_warm_model:
            cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {"function_calls": [], "total_time_ms": 0, "confidence": 0}

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": float(raw.get("confidence", 0) or 0),
    }


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t.get("description", ""),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v.get("type", "string").upper(), description=v.get("description", ""))
                        for k, v in t.get("parameters", {}).get("properties", {}).items()
                    },
                    required=t.get("parameters", {}).get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    contents = [m.get("content", "") for m in messages if m.get("role") == "user"]

    start_time = time.time()
    gemini_response = client.models.generate_content(
        model="gemini-2.0-flash",
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

    return {"function_calls": function_calls, "total_time_ms": total_time_ms}


def _latest_user_text(messages):
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def _complexity_score(text):
    lower = text.lower()
    words = [token for token in re.split(r"[^a-z0-9]+", lower) if token]
    conjunctions = sum(lower.count(f" {word} ") for word in ("and", "then", "after", "while", "also", "but"))
    punctuation = sum(lower.count(ch) for ch in ",;:")
    numeric_values = len(re.findall(r"\$?\d+(?:\.\d+)?", lower))
    clauses = max(0, punctuation - 1)
    return len(words) + (3 * conjunctions) + (2 * clauses) + max(0, numeric_values - 1)


def _likely_tool_request(text):
    lower = f" {text.lower()} "
    action_terms = (
        "set ", "send ", "create ", "find ", "search ", "play ", "book ",
        "schedule ", "call ", "text ", "message ", "remind ", "timer ", "alarm ",
        "weather ", "temperature ", "owe ", "split ", "bill ", "pay ", "charge ",
    )
    return any(term in lower for term in action_terms)


def _dynamic_threshold(base_threshold, complexity):
    threshold = max(0.55, min(base_threshold, 0.98))
    if complexity >= 28:
        threshold = min(0.97, threshold + 0.04)
    elif complexity <= 10:
        threshold = max(0.55, threshold - 0.06)
    return threshold


def _tool_schema_map(tools):
    schema = {}
    for tool in tools:
        params = tool.get("parameters", {})
        schema[tool.get("name")] = {
            "properties": params.get("properties", {}),
            "required": set(params.get("required", [])),
        }
    return schema


def _coerce_argument(value, arg_schema):
    if value is None:
        return None

    arg_type = str(arg_schema.get("type", "")).lower()

    if arg_type == "integer":
        try:
            if isinstance(value, bool):
                return None
            return int(round(float(value)))
        except (TypeError, ValueError):
            return None

    if arg_type == "number":
        try:
            if isinstance(value, bool):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    if arg_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"true", "yes", "1", "on"}:
                return True
            if v in {"false", "no", "0", "off"}:
                return False
        return None

    if arg_type == "array":
        if isinstance(value, list):
            return value
        return None

    if arg_type == "object":
        if isinstance(value, dict):
            return value
        return None

    # Default to string behavior for unknown or string types.
    text = str(value).strip()
    return text if text else None


def _clean_text_fragment(text):
    return re.sub(r"\s+", " ", str(text)).strip(" .,!?:;\"'")


def _extract_time_mentions(text):
    mentions = []
    for match in _TIME_RE.finditer(text):
        hour = int(match.group(1))
        minute = int(match.group(2) or "0")
        meridiem = match.group(3).lower()
        if meridiem == "pm" and hour != 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0
        raw = (
            f"{match.group(1)}:{match.group(2)} {meridiem.upper()}"
            if match.group(2)
            else f"{match.group(1)}:00 {meridiem.upper()}"
        )
        mentions.append({
            "start": match.start(),
            "end": match.end(),
            "hour": hour,
            "minute": minute,
            "raw": raw,
        })
    return mentions


def _split_actions(user_text):
    text = re.sub(r"\s*,\s*and\s+", ", ", user_text, flags=re.IGNORECASE)
    text = re.sub(r"\s+and\s+", " |AND| ", text, flags=re.IGNORECASE)
    text = text.replace(",", " |AND| ")
    return [_clean_text_fragment(part) for part in text.split("|AND|") if _clean_text_fragment(part)]


_STOP_TOKENS = {
    "a", "an", "and", "or", "the", "to", "for", "of", "on", "in", "at", "from", "with",
    "is", "are", "was", "were", "be", "me", "my", "you", "your", "what", "whats", "please",
}
_PERSON_PRONOUNS = {"him", "her", "them"}


def _tokenize(text):
    return [
        token for token in re.split(r"[^a-z0-9]+", str(text).lower())
        if token and token not in _STOP_TOKENS
    ]


def _tool_terms(tool):
    params = tool.get("parameters", {})
    props = params.get("properties", {})
    text_parts = [tool.get("name", ""), tool.get("description", "")]
    text_parts.extend(props.keys())
    text_parts.extend(
        prop.get("description", "")
        for prop in props.values()
        if isinstance(prop, dict)
    )
    return _tokenize(" ".join(text_parts))


def _tool_relevance(segment, tool):
    seg_tokens = set(_tokenize(segment))
    if not seg_tokens:
        return 0.0

    terms = _tool_terms(tool)
    if not terms:
        return 0.0

    overlap = sum(1 for token in terms if token in seg_tokens)
    name_tokens = _tokenize(str(tool.get("name", "")).replace("_", " "))
    name_bonus = sum(1 for token in name_tokens if token in seg_tokens)
    return float(overlap + (2 * name_bonus))


def _extract_quoted(segment):
    match = re.search(r"['\"]([^'\"]{1,200})['\"]", segment)
    if match:
        return _clean_text_fragment(match.group(1))
    return None


def _extract_after_keywords(segment, keywords):
    key_pattern = "|".join(re.escape(k) for k in keywords)
    match = re.search(rf"\b(?:{key_pattern})\b\s+(.+)$", segment, flags=re.IGNORECASE)
    if not match:
        return None
    text = _clean_text_fragment(match.group(1))
    return text if text else None


def _arg_hint_text(arg_name, arg_schema):
    return f"{arg_name} {arg_schema.get('description', '')}".lower()


def _extract_person_like(segment, memory):
    patterns = (
        r"\btext\s+([A-Za-z][A-Za-z'\-]*)",
        r"\bsend\s+(?:a\s+)?message\s+to\s+([A-Za-z][A-Za-z'\-]*)",
        r"\bsend\s+([A-Za-z][A-Za-z'\-]*)\s+a\s+message",
        r"\bmessage\s+([A-Za-z][A-Za-z'\-]*)",
        r"\bto\s+([A-Za-z][A-Za-z'\-]*)",
    )
    for pattern in patterns:
        match = re.search(pattern, segment, flags=re.IGNORECASE)
        if match:
            name = _clean_text_fragment(match.group(1))
            if name.lower() in _PERSON_PRONOUNS:
                return memory.get("last_person")
            return name

    lower = segment.lower()
    if any(f" {p} " in f" {lower} " for p in _PERSON_PRONOUNS):
        return memory.get("last_person")
    return None


def _extract_message_like(segment):
    patterns = (
        r"\bsaying\s+(.+)$",
        r"\bmessage\s+(?:that\s+says|saying)\s+(.+)$",
        r"\btext\s+[A-Za-z][A-Za-z'\-]*\s+(.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, segment, flags=re.IGNORECASE)
        if match:
            value = _clean_text_fragment(match.group(1))
            if value:
                return value
    quoted = _extract_quoted(segment)
    return quoted


def _extract_query_like(segment):
    match = re.search(r"\b(?:find|look up|search(?: for)?)\s+([A-Za-z][A-Za-z'\-]*)", segment, flags=re.IGNORECASE)
    if not match:
        return None
    query = _clean_text_fragment(match.group(1))
    return None if query.lower() in {"in", "my", "contacts"} else query


def _extract_location_like(segment):
    patterns = (
        r"\b(?:weather|temperature)(?:\s+like)?\s+in\s+(.+)$",
        r"\b(?:in|at)\s+([A-Za-z][A-Za-z' \-]*)$",
    )
    for pattern in patterns:
        match = re.search(pattern, segment, flags=re.IGNORECASE)
        if match:
            location = _clean_text_fragment(match.group(1))
            if location:
                return location
    return None


def _extract_title_like(segment):
    patterns = (
        r"\bremind\s+me\s+(?:about\s+|to\s+)?(.+?)\s+at\s+",
        r"\b(?:title|subject)\s+(.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, segment, flags=re.IGNORECASE)
        if match:
            title = _clean_text_fragment(match.group(1))
            if title.lower().startswith("the "):
                title = title[4:]
            return _clean_text_fragment(title)
    return None


def _extract_media_like(segment):
    match = re.search(r"\bplay\s+(.+)$", segment, flags=re.IGNORECASE)
    if not match:
        return None
    value = _clean_text_fragment(match.group(1))
    had_some_prefix = value.lower().startswith("some ")
    if had_some_prefix:
        value = _clean_text_fragment(value[5:])
    if had_some_prefix and value.lower().endswith(" music"):
        value = _clean_text_fragment(value[:-6])
    return value or None


def _extract_generic_string(segment):
    quoted = _extract_quoted(segment)
    if quoted:
        return quoted

    caps = re.findall(r"\b[A-Z][A-Za-z'\-]*(?:\s+[A-Z][A-Za-z'\-]*)*\b", segment)
    if caps:
        return _clean_text_fragment(caps[-1])

    return _extract_after_keywords(segment, ("to", "for", "in", "at"))


def _extract_number(segment):
    match = re.search(r"\b(\d+)\b", segment)
    return int(match.group(1)) if match else None


def _extract_argument_value(segment, arg_name, arg_schema, memory):
    hint_text = _arg_hint_text(arg_name, arg_schema)
    arg_type = str(arg_schema.get("type", "string")).lower()
    times = _extract_time_mentions(segment)
    arg_name_lower = str(arg_name).lower()
    has_plural_minutes = "minutes" in hint_text
    is_clock_minute = (
        arg_name_lower in {"minute", "min"}
        or ("minute" in hint_text and not has_plural_minutes)
    )

    if arg_type == "integer":
        if "hour" in hint_text:
            if times:
                return times[0]["hour"], 1.0
            return None, 0.0
        if is_clock_minute and "duration" not in hint_text and "timer" not in hint_text and "countdown" not in hint_text:
            if times:
                return times[0]["minute"], 1.0
            return None, 0.0
        if "minute" in hint_text or "duration" in hint_text or "timer" in hint_text:
            duration = _extract_duration_hint(segment)
            if duration is not None:
                return int(duration), 1.0
        value = _extract_number(segment)
        if value is not None:
            return value, 0.7
        return None, 0.0

    if arg_type == "number":
        value = _extract_number(segment)
        if value is not None:
            return float(value), 0.7
        return None, 0.0

    if arg_type == "boolean":
        lower = segment.lower()
        if any(token in lower for token in ("true", "yes", "enable", "on")):
            return True, 0.7
        if any(token in lower for token in ("false", "no", "disable", "off")):
            return False, 0.7
        return None, 0.0

    if arg_type != "string":
        return None, 0.0

    # String slot inference based on argument semantics (name + description).
    value = None
    confidence = 0.0

    if any(token in hint_text for token in ("recipient", "contact", "person")):
        value = _extract_person_like(segment, memory)
        confidence = 1.0 if value else 0.0
    elif any(token in hint_text for token in ("message", "text", "body", "content")):
        value = _extract_message_like(segment)
        confidence = 1.0 if value else 0.0
    elif any(token in hint_text for token in ("query", "search", "keyword", "term")):
        value = _extract_query_like(segment)
        confidence = 1.0 if value else 0.0
    elif any(token in hint_text for token in ("location", "city", "place", "address", "where")):
        value = _extract_location_like(segment)
        confidence = 1.0 if value else 0.0
    elif any(token in hint_text for token in ("time", "datetime", "when", "date")):
        if times:
            value = times[0]["raw"]
            confidence = 1.0
    elif any(token in hint_text for token in ("title", "subject", "task", "reminder")):
        value = _extract_title_like(segment)
        confidence = 1.0 if value else 0.0
    elif any(token in hint_text for token in ("song", "music", "playlist", "track", "media")):
        value = _extract_media_like(segment)
        confidence = 1.0 if value else 0.0

    if not value:
        value = _extract_generic_string(segment)
        confidence = 0.45 if value else 0.0

    return value, confidence


def _extract_required_args(segment, tool, memory):
    params = tool.get("parameters", {})
    properties = params.get("properties", {})
    required = params.get("required", [])
    args = {}
    scores = []

    for arg_name in required:
        arg_schema = properties.get(arg_name, {"type": "string"})
        value, confidence = _extract_argument_value(segment, arg_name, arg_schema, memory)
        if value is None:
            return None, 0.0
        coerced = _coerce_argument(value, arg_schema)
        if coerced is None:
            return None, 0.0
        args[arg_name] = coerced
        scores.append(confidence)

    quality = sum(scores) / len(scores) if scores else 0.0
    return args, quality


def _schema_infer_calls_from_text(user_text, tools):
    calls = []
    quality_scores = []
    memory = {"last_person": None}

    for segment in _split_actions(user_text):
        best_call = None
        best_score = float("-inf")
        best_quality = 0.0

        for tool in tools:
            tool_name = tool.get("name")
            if not tool_name:
                continue
            args, arg_quality = _extract_required_args(segment, tool, memory)
            if args is None:
                continue

            score = _tool_relevance(segment, tool) + (4.0 * arg_quality)
            if score > best_score:
                best_score = score
                best_quality = arg_quality
                best_call = {"name": tool_name, "arguments": args}

        if best_call:
            calls.append(best_call)
            quality_scores.append(best_quality)
            for key in ("recipient", "query", "contact"):
                if key in best_call["arguments"]:
                    memory["last_person"] = _clean_text_fragment(best_call["arguments"][key])
                    break

    if not calls:
        return [], 0.0

    normalized = _validate_and_normalize_calls(calls, tools, user_text)
    quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    return normalized, quality


def _extract_search_query(segment):
    match = re.search(r"\b(?:find|look up|search(?: for)?)\s+([A-Za-z][A-Za-z'\-]*)", segment, flags=re.IGNORECASE)
    if not match:
        return None
    query = _clean_text_fragment(match.group(1))
    return None if query.lower() in {"in", "my", "contacts"} else query


def _extract_message_args(segment, remembered_person):
    lower = segment.lower()
    if (
        "message" not in lower
        and not lower.startswith("text ")
        and " text " not in f" {lower} "
        and not lower.startswith("send ")
    ):
        return None

    recipient = None
    recipient_patterns = (
        r"\btext\s+([A-Za-z][A-Za-z'\-]*)",
        r"\bsend\s+(?:a\s+)?message\s+to\s+([A-Za-z][A-Za-z'\-]*)",
        r"\bsend\s+([A-Za-z][A-Za-z'\-]*)\s+a\s+message",
    )
    for pattern in recipient_patterns:
        match = re.search(pattern, segment, flags=re.IGNORECASE)
        if match:
            recipient = _clean_text_fragment(match.group(1))
            break

    if recipient and recipient.lower() in {"him", "her", "them"}:
        recipient = remembered_person
    if not recipient and re.search(r"\b(him|her|them)\b", lower):
        recipient = remembered_person

    message_match = re.search(r"\bsaying\s+(.+)$", segment, flags=re.IGNORECASE)
    message = _clean_text_fragment(message_match.group(1)) if message_match else None

    if recipient and message:
        return {"recipient": recipient, "message": message}
    return None


def _extract_weather_location(segment):
    if "weather" not in segment.lower():
        return None
    patterns = (
        r"\bweather(?:\s+like)?\s+in\s+(.+)$",
        r"\bcheck\s+the\s+weather\s+in\s+(.+)$",
        r"\bweather\s+in\s+(.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, segment, flags=re.IGNORECASE)
        if match:
            return _clean_text_fragment(match.group(1))
    return None


def _extract_alarm_args(segment):
    lower = segment.lower()
    if "alarm" not in lower and "wake me up" not in lower:
        return None
    mentions = _extract_time_mentions(segment)
    if not mentions:
        return None
    return {"hour": mentions[0]["hour"], "minute": mentions[0]["minute"]}


def _extract_timer_args(segment):
    if "timer" not in segment.lower():
        return None
    match = re.search(r"\b(\d+)\s*minute", segment, flags=re.IGNORECASE)
    if not match:
        return None
    return {"minutes": int(match.group(1))}


def _extract_reminder_args(segment):
    if "remind me" not in segment.lower():
        return None
    mentions = _extract_time_mentions(segment)
    if not mentions:
        return None

    title_match = re.search(
        r"\bremind\s+me\s+(?:about\s+|to\s+)?(.+?)\s+at\s+",
        segment,
        flags=re.IGNORECASE,
    )
    title = _clean_text_fragment(title_match.group(1)) if title_match else "reminder"
    if title.lower().startswith("the "):
        title = title[4:]

    return {"title": _clean_text_fragment(title), "time": mentions[0]["raw"]}


def _extract_music_args(segment):
    match = re.search(r"\bplay\s+(.+)$", segment, flags=re.IGNORECASE)
    if not match:
        return None
    song = _clean_text_fragment(match.group(1))
    if song.lower().startswith("some "):
        song = _clean_text_fragment(song[5:])
        if song.lower().endswith(" music"):
            song = _clean_text_fragment(song[:-6])
    return {"song": song} if song else None


def _infer_calls_from_text(user_text, tools):
    by_signature = {
        frozenset(tool.get("parameters", {}).get("required", [])): tool["name"]
        for tool in tools
        if tool.get("name")
    }

    inferred = []
    remembered_person = None

    for segment in _split_actions(user_text):
        search_sig = frozenset({"query"})
        if search_sig in by_signature:
            query = _extract_search_query(segment)
            if query:
                inferred.append({
                    "name": by_signature[search_sig],
                    "arguments": {"query": query},
                })
                remembered_person = query
                continue

        message_sig = frozenset({"recipient", "message"})
        if message_sig in by_signature:
            message_args = _extract_message_args(segment, remembered_person)
            if message_args:
                inferred.append({
                    "name": by_signature[message_sig],
                    "arguments": message_args,
                })
                remembered_person = message_args.get("recipient", remembered_person)
                continue

        weather_sig = frozenset({"location"})
        if weather_sig in by_signature:
            location = _extract_weather_location(segment)
            if location:
                inferred.append({
                    "name": by_signature[weather_sig],
                    "arguments": {"location": location},
                })
                continue

        alarm_sig = frozenset({"hour", "minute"})
        if alarm_sig in by_signature:
            alarm_args = _extract_alarm_args(segment)
            if alarm_args:
                inferred.append({
                    "name": by_signature[alarm_sig],
                    "arguments": alarm_args,
                })
                continue

        timer_sig = frozenset({"minutes"})
        if timer_sig in by_signature:
            timer_args = _extract_timer_args(segment)
            if timer_args:
                inferred.append({
                    "name": by_signature[timer_sig],
                    "arguments": timer_args,
                })
                continue

        reminder_sig = frozenset({"title", "time"})
        if reminder_sig in by_signature:
            reminder_args = _extract_reminder_args(segment)
            if reminder_args:
                inferred.append({
                    "name": by_signature[reminder_sig],
                    "arguments": reminder_args,
                })
                continue

        music_sig = frozenset({"song"})
        if music_sig in by_signature:
            music_args = _extract_music_args(segment)
            if music_args:
                inferred.append({
                    "name": by_signature[music_sig],
                    "arguments": music_args,
                })
                continue

    return inferred


def _extract_time_hints(user_text):
    hints = []
    for m in re.finditer(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", user_text, flags=re.IGNORECASE):
        hour = int(m.group(1))
        minute = int(m.group(2) or "0")
        meridiem = m.group(3).lower()
        if meridiem == "pm" and hour != 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0
        hints.append({"hour": hour % 24, "minute": minute})
    return hints


def _extract_duration_hint(user_text):
    m = re.search(r"\b(\d+)\s*minute(?:s)?\b", user_text, flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))


def _clean_string_arg(value):
    s = str(value).strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"'s phone number$", "", s, flags=re.IGNORECASE)
    return s.strip(" .")


def _sanity_fix_arguments(name, args, user_text):
    fixed = dict(args)
    time_hints = _extract_time_hints(user_text)
    duration_hint = _extract_duration_hint(user_text)

    if "hour" in fixed:
        h = fixed.get("hour")
        if not isinstance(h, int) or h < 0 or h > 23:
            if time_hints:
                fixed["hour"] = time_hints[0]["hour"]

    if "minute" in fixed:
        m = fixed.get("minute")
        if not isinstance(m, int) or m < 0 or m > 59:
            if time_hints:
                preferred = None
                if "hour" in fixed:
                    preferred = next((t["minute"] for t in time_hints if t["hour"] == fixed["hour"]), None)
                fixed["minute"] = preferred if preferred is not None else time_hints[0]["minute"]
            else:
                fixed["minute"] = 0

    if "minutes" in fixed:
        mins = fixed.get("minutes")
        if isinstance(mins, int):
            if mins <= 0:
                fixed["minutes"] = abs(mins)
            if fixed["minutes"] == 0:
                fixed["minutes"] = duration_hint or 1
            if fixed["minutes"] > 300 and duration_hint:
                fixed["minutes"] = duration_hint
        elif duration_hint:
            fixed["minutes"] = duration_hint

    for key in ("recipient", "query", "location", "message", "title", "song", "time"):
        if key in fixed and fixed[key] is not None:
            fixed[key] = _clean_string_arg(fixed[key])

    return fixed


def _validate_and_normalize_calls(calls, tools, user_text=""):
    schemas = _tool_schema_map(tools)
    normalized = []

    for call in calls:
        if not isinstance(call, dict):
            continue

        name = call.get("name")
        if name not in schemas:
            continue

        raw_args = call.get("arguments")
        if not isinstance(raw_args, dict):
            continue

        tool_schema = schemas[name]
        properties = tool_schema["properties"]
        required = tool_schema["required"]

        parsed_args = {}
        for key, value in raw_args.items():
            if key not in properties:
                continue
            coerced = _coerce_argument(value, properties.get(key, {}))
            if coerced is not None:
                parsed_args[key] = coerced

        parsed_args = _sanity_fix_arguments(name, parsed_args, user_text)

        missing_required = [k for k in required if k not in parsed_args]
        if missing_required:
            continue

        normalized.append({"name": name, "arguments": parsed_args})

    # De-duplicate stable calls.
    unique = []
    seen = set()
    for call in normalized:
        key = (call["name"], json.dumps(call["arguments"], sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        unique.append(call)

    return unique


def _repair_calls_locally(messages, tools, draft_calls):
    repair_prompt = (
        "You repair function tool calls. Use ONLY provided tools. "
        "Every returned call must include all required arguments using correct primitive types. "
        "Drop impossible calls instead of guessing unknown required values."
    )

    repair_messages = messages + [{
        "role": "user",
        "content": (
            "Draft tool calls from previous pass: "
            f"{json.dumps(draft_calls)}. "
            "Return corrected function calls now."
        ),
    }]

    return generate_cactus(repair_messages, tools, system_prompt=repair_prompt, max_tokens=320, temperature=0.1)


def _local_extraction_prompt():
    return (
        "You are a precise function-calling planner. "
        "Return ALL required function calls that are explicitly requested by the user, in order. "
        "Do not omit any requested action. Do not invent actions. "
        "Always provide required arguments with correct primitive types. "
        "For time expressions like '10 AM', use minute=0 if minutes are not spoken. "
        "For timer expressions like '15 minute timer', set minutes=15 (positive integer). "
        "Resolve pronouns like 'him/her' using the nearest previously mentioned person in the same user request."
    )


def _local_exhaustive_prompt():
    return (
        "You are an exhaustive multi-action tool caller. "
        "If a user request contains multiple actions connected by 'and' or commas, emit one function call per action. "
        "Return only valid tool calls using provided tools and required arguments. "
        "Do not skip actions. Do not add extra actions."
    )


def _estimated_call_count(user_text):
    conjunctions = len(re.findall(r"\band\b", user_text, flags=re.IGNORECASE))
    commas = user_text.count(",")
    return max(1, 1 + max(conjunctions, commas))


def _hardness_features(user_text):
    words = [token for token in re.split(r"[^a-z0-9]+", user_text.lower()) if token]
    and_count = len(re.findall(r"\band\b", user_text, flags=re.IGNORECASE))
    comma_count = user_text.count(",")

    score = 0
    if len(words) >= 14:
        score += 1
    if and_count >= 1:
        score += 1
    if comma_count >= 1:
        score += 1
    if (and_count + comma_count) >= 2:
        score += 1

    return {
        "word_count": len(words),
        "and_count": and_count,
        "comma_count": comma_count,
        "hardness_score": score,
    }


def _is_hard_request(user_text):
    features = _hardness_features(user_text)
    return features["hardness_score"] >= 2


def _candidate_score(calls, confidence, user_text, needs_tool):
    est = _estimated_call_count(user_text)
    count = len(calls)
    if needs_tool and count == 0:
        return -1000
    proximity = 1.0 - (abs(count - est) / max(1, est))
    grounding = _string_grounding_score(calls, user_text)
    return (
        (3.0 * max(0.0, proximity))
        + (4.0 * grounding)
        + (2.5 * float(confidence or 0.0))
        + (1.5 * min(count, est))
    )


def _string_grounding_score(calls, user_text):
    text = f" {str(user_text).lower()} "
    total = 0
    grounded = 0

    for call in calls:
        args = call.get("arguments", {})
        if not isinstance(args, dict):
            continue
        for value in args.values():
            if not isinstance(value, str):
                continue
            s = _clean_string_arg(value).lower()
            if not s:
                continue
            total += 1
            if f" {s} " in text:
                grounded += 1
                continue
            tokens = [
                token
                for token in re.split(r"[^a-z0-9]+", s)
                if token and token not in _STOP_TOKENS and len(token) > 2
            ]
            if not tokens:
                grounded += 1
                continue
            matched = sum(1 for token in tokens if f" {token} " in text)
            if matched >= max(1, len(tokens) - 1):
                grounded += 1

    if total == 0:
        return 1.0
    return grounded / total


def _segment_model_pass(user_text, tools):
    segments = _split_actions(user_text)
    if len(segments) <= 1:
        return {"function_calls": [], "total_time_ms": 0.0, "confidence": 0.0}

    calls = []
    confidence_sum = 0.0
    total_ms = 0.0

    for segment in segments:
        run = generate_cactus(
            [{"role": "user", "content": segment}],
            tools,
            system_prompt=_local_extraction_prompt(),
            max_tokens=192,
            temperature=0.0,
        )
        seg_calls = _validate_and_normalize_calls(run.get("function_calls", []), tools, segment)
        calls.extend(seg_calls)
        confidence_sum += float(run.get("confidence", 0) or 0)
        total_ms += float(run.get("total_time_ms", 0) or 0)

    calls = _validate_and_normalize_calls(calls, tools, user_text)
    avg_conf = confidence_sum / len(segments)
    return {"function_calls": calls, "total_time_ms": total_ms, "confidence": avg_conf}


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """
    On-device-first hybrid routing without hardcoded task regexes:
    1) local model generation
    2) schema validation + local repair pass
    3) rare cloud fallback when still low-confidence/invalid
    """
    route_start = time.perf_counter()
    user_text = _latest_user_text(messages)
    complexity = _complexity_score(user_text)
    hard_features = _hardness_features(user_text)
    is_hard = _is_hard_request(user_text)
    threshold = _dynamic_threshold(confidence_threshold, complexity)
    needs_tool = _likely_tool_request(user_text)
    estimated_calls = _estimated_call_count(user_text)

    candidates = []

    schema_calls, schema_quality = _schema_infer_calls_from_text(user_text, tools)
    if schema_calls:
        schema_conf = 0.62 + (0.38 * max(0.0, min(1.0, schema_quality)))
        candidates.append({
            "calls": schema_calls,
            "confidence": schema_conf,
            "label": "schema",
        })

    primary = generate_cactus(messages, tools, system_prompt=_local_extraction_prompt(), temperature=0.0)
    primary_calls = _validate_and_normalize_calls(primary.get("function_calls", []), tools, user_text)
    primary_conf = float(primary.get("confidence", 0) or 0)
    candidates.append({
        "calls": primary_calls,
        "confidence": primary_conf,
        "label": "primary",
    })

    if (needs_tool and len(primary_calls) < estimated_calls) or primary_conf < max(0.58, threshold - 0.10):
        segmented = _segment_model_pass(user_text, tools)
        seg_calls = _validate_and_normalize_calls(segmented.get("function_calls", []), tools, user_text)
        seg_conf = float(segmented.get("confidence", 0) or 0)
        candidates.append({
            "calls": seg_calls,
            "confidence": seg_conf,
            "label": "segmented",
        })

    if (needs_tool and not primary_calls) or primary_conf < max(0.52, threshold - 0.20):
        secondary = generate_cactus(messages, tools, system_prompt=_local_exhaustive_prompt(), temperature=0.2)
        secondary_calls = _validate_and_normalize_calls(secondary.get("function_calls", []), tools, user_text)
        secondary_conf = float(secondary.get("confidence", 0) or 0)
        candidates.append({
            "calls": secondary_calls,
            "confidence": secondary_conf,
            "label": "secondary",
        })

    best = None
    candidate_signatures = set()
    for item in candidates:
        calls = item["calls"]
        conf = float(item["confidence"] or 0.0)
        score = _candidate_score(calls, conf, user_text, needs_tool)
        signature = tuple(
            (call["name"], json.dumps(call["arguments"], sort_keys=True))
            for call in calls
        )
        candidate_signatures.add(signature)
        if best is None or score > best["score"]:
            best = {"calls": calls, "confidence": conf, "score": score, "label": item["label"]}

    if best is None:
        best = {"calls": [], "confidence": 0.0, "score": -1000, "label": "none"}

    best_calls = best["calls"]
    best_conf = float(best["confidence"] or 0.0)
    best_grounding = _string_grounding_score(best_calls, user_text)

    undercalled_actions = needs_tool and len(best_calls) < estimated_calls
    candidate_disagreement = len(candidate_signatures) > 1

    fallback_reasons = []
    if best_conf < threshold:
        fallback_reasons.append("low_confidence")
    if needs_tool and not best_calls:
        fallback_reasons.append("missing_or_invalid_function_call")
    if undercalled_actions:
        fallback_reasons.append("undercalled_actions")
    if best_grounding < 0.8:
        fallback_reasons.append("low_argument_grounding")
    if candidate_disagreement:
        fallback_reasons.append("candidate_disagreement")

    should_repair = bool(fallback_reasons) and (undercalled_actions or best_grounding < 0.85 or best_conf < threshold)
    repaired_calls = []
    repaired_confidence = 0.0
    if should_repair:
        repaired = _repair_calls_locally(messages, tools, best_calls)
        repaired_calls = _validate_and_normalize_calls(repaired.get("function_calls", []), tools, user_text)
        repaired_confidence = float(repaired.get("confidence", 0) or 0)
        repaired_score = _candidate_score(repaired_calls, repaired_confidence, user_text, needs_tool)
        if repaired_score > best["score"]:
            best_calls = repaired_calls
            best_conf = repaired_confidence
            best_grounding = _string_grounding_score(best_calls, user_text)

    # Re-evaluate risk after optional repair.
    undercalled_actions = needs_tool and len(best_calls) < estimated_calls
    cloud_risk = (
        undercalled_actions
        or (needs_tool and not best_calls)
        or (best_grounding < 0.8)
        or (best_conf < max(0.90, threshold - 0.04))
    )

    cloud_available = bool(os.environ.get("GEMINI_API_KEY"))
    should_use_cloud = cloud_available and is_hard and cloud_risk

    if not should_use_cloud:
        return {
            "function_calls": best_calls,
            "total_time_ms": (time.perf_counter() - route_start) * 1000.0,
            "confidence": best_conf,
            "source": "on-device",
            "routing": {
                "complexity_score": complexity,
                "hardness_features": hard_features,
                "threshold": threshold,
                "fallback_reasons": fallback_reasons + (["cloud_not_used"] if cloud_available else ["cloud_unavailable"]),
                "strategy": "local_candidate_ensemble",
            },
        }

    try:
        cloud = generate_cloud(messages, tools)
        cloud_calls = _validate_and_normalize_calls(cloud.get("function_calls", []), tools, user_text)
        return {
            "function_calls": cloud_calls,
            "total_time_ms": (time.perf_counter() - route_start) * 1000.0,
            "local_confidence": best_conf,
            "source": "cloud (fallback)",
            "routing": {
                "complexity_score": complexity,
                "hardness_features": hard_features,
                "threshold": threshold,
                "fallback_reasons": fallback_reasons + ["cloud_used"],
                "strategy": "local_then_cloud",
            },
        }
    except Exception:
        return {
            "function_calls": best_calls,
            "total_time_ms": (time.perf_counter() - route_start) * 1000.0,
            "confidence": best_conf,
            "source": "on-device",
            "routing": {
                "complexity_score": complexity,
                "hardness_features": hard_features,
                "threshold": threshold,
                "fallback_reasons": fallback_reasons + ["cloud_failed"],
                "strategy": "local_candidate_ensemble",
            },
        }


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
    for call in result.get("function_calls", []):
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

    if os.environ.get("GEMINI_API_KEY"):
        cloud = generate_cloud(messages, tools)
        print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)
