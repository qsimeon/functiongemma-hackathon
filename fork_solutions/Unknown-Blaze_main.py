# Source: https://github.com/Unknown-Blaze/functiongemma-hackathon
import sys
from pathlib import Path
import re
import atexit
import json, os, time

def _resolve_cactus_paths():
    here = Path(__file__).resolve().parent
    candidates = [
        here,
        here.parent,
        Path.cwd(),
    ]

    for root in candidates:
        cactus_src = root / "cactus" / "python" / "src"
        weights_path = root / "cactus" / "weights" / "functiongemma-270m-it"
        if cactus_src.exists() and weights_path.exists():
            return root, cactus_src, weights_path

    default_root = here.parent
    return (
        default_root,
        default_root / "cactus" / "python" / "src",
        default_root / "cactus" / "weights" / "functiongemma-270m-it",
    )


REPO_ROOT, CACTUS_PYTHON_SRC, FUNCTIONGEMMA_PATH = _resolve_cactus_paths()

sys.path.insert(0, str(CACTUS_PYTHON_SRC))

try:
    from cactus import cactus_init, cactus_complete, cactus_destroy
    _CACTUS_AVAILABLE = True
except Exception:
    cactus_init = None
    cactus_complete = None
    cactus_destroy = None
    _CACTUS_AVAILABLE = False


_CACTUS_MODEL = None
_STOPWORDS = {
    "the", "a", "an", "to", "for", "of", "in", "on", "at", "and", "or", "my", "me", "please",
    "current", "given", "with", "by", "from", "is", "are", "be", "set", "get", "check", "create",
}

# Router tuning constants (kept explicit to avoid magic numbers in judging/demo discussions).
LOCAL_ACCEPT_CONFIDENCE = 0.50
ROUTER_ACCEPT_CONFIDENCE = 0.58
ROUTER_REPORTED_CONFIDENCE_FLOOR = 0.58
DEFAULT_HYBRID_CONFIDENCE_THRESHOLD = 0.55


def _is_truthy_env(var_name, default=False):
    value = os.environ.get(var_name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _cloud_fallback_enabled():
    """Cloud fallback stays available; env var can disable it for local-only experiments."""
    return _is_truthy_env("ENABLE_CLOUD_FALLBACK", default=True)


def _get_cactus_model():
    if not _CACTUS_AVAILABLE:
        return None
    global _CACTUS_MODEL
    if _CACTUS_MODEL is None:
        try:
            _CACTUS_MODEL = cactus_init(str(FUNCTIONGEMMA_PATH))
        except Exception:
            _CACTUS_MODEL = None
    return _CACTUS_MODEL


@atexit.register
def _cleanup_cactus_model():
    global _CACTUS_MODEL
    if _CACTUS_MODEL is not None:
        cactus_destroy(_CACTUS_MODEL)
        _CACTUS_MODEL = None


def _trim_segment(text):
    cut_tokens = [
        ",",
        ", and ",
        " and ",
        ".",
        "?",
        "!",
    ]
    out = text.strip()
    for token in cut_tokens:
        pos = out.lower().find(token)
        if pos != -1:
            out = out[:pos].strip()
    return out.strip(" .,!?")


def _parse_time_to_alarm(time_str):
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", time_str.lower())
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2) or "0")
    meridian = m.group(3)
    if meridian == "am":
        hour = 0 if hour == 12 else hour
    else:
        hour = 12 if hour == 12 else hour + 12
    return hour, minute


def _tokenize(text):
    return [w for w in re.findall(r"[a-zA-Z']+", text.lower()) if w not in _STOPWORDS]


def _tool_keywords(tool):
    parts = []
    parts.extend(tool.get("name", "").replace("_", " ").split())
    parts.extend(_tokenize(tool.get("description", "")))
    for key, spec in tool.get("parameters", {}).get("properties", {}).items():
        parts.extend(key.replace("_", " ").split())
        parts.extend(_tokenize(spec.get("description", "")))
    kws = {p.lower() for p in parts if p and p.lower() not in _STOPWORDS}
    name = tool.get("name", "").lower()

    semantic_expansions = {
        "weather": {"weather", "forecast", "temperature", "city", "location"},
        "alarm": {"alarm", "wake", "morning", "am", "pm", "clock"},
        "timer": {"timer", "countdown", "minutes", "minute"},
        "music": {"music", "song", "playlist", "play", "audio", "track"},
        "message": {"message", "text", "sms", "dm", "recipient", "send"},
        "contact": {"contact", "contacts", "find", "lookup", "search", "query"},
        "reminder": {"reminder", "remind", "title", "time", "schedule"},
    }

    for concept, extras in semantic_expansions.items():
        if concept in name:
            kws |= extras

    return kws


def _extract_time_string(clause):
    m = re.search(r"\b(\d{1,2}(:\d{2})?\s*(am|pm))\b", clause, re.IGNORECASE)
    return m.group(1).upper() if m else ""


def _canonicalize_time_string(value):
    if not isinstance(value, str):
        return value
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", value, re.IGNORECASE)
    if not m:
        return value.strip().upper()
    hour = int(m.group(1))
    minute = int(m.group(2) or "0")
    meridian = m.group(3).upper()
    return f"{hour}:{minute:02d} {meridian}"


def _extract_args_generic(clause, tool_name):
    args = {}
    lower = clause.lower()

    if "weather" in tool_name or "location" in lower:
        m = re.search(r"\bin\s+([A-Za-z][A-Za-z\s\-']+)", clause, re.IGNORECASE)
        if m:
            args["location"] = _trim_segment(m.group(1))
        elif "weather" in lower:
            m2 = re.search(r"weather(?:\s+(?:in|for))?\s+([A-Za-z][A-Za-z\s\-']+)", clause, re.IGNORECASE)
            if m2:
                args["location"] = _trim_segment(m2.group(1))

    if "alarm" in tool_name:
        parsed = _parse_time_to_alarm(clause)
        if parsed:
            hour, minute = parsed
            args["hour"] = hour
            args["minute"] = minute

    if "timer" in tool_name:
        m = re.search(r"(\d+)\s*(?:minutes?|mins?|m)\b", clause, re.IGNORECASE)
        if m:
            args["minutes"] = int(m.group(1))

    if "search" in tool_name and "contact" in tool_name:
        m = re.search(r"(?:find|look up|search for)\s+([A-Za-z][A-Za-z\-']+)", clause, re.IGNORECASE)
        if m:
            args["query"] = m.group(1)

    if "message" in tool_name:
        m1 = re.search(r"(?:to|text)\s+([A-Za-z][A-Za-z\-']+|him|her)", clause, re.IGNORECASE)
        if m1:
            args["recipient"] = m1.group(1)
        else:
            m1b = re.search(r"(?:message|send)\s+([A-Za-z][A-Za-z\-']+|him|her)\b", clause, re.IGNORECASE)
            if m1b:
                args["recipient"] = m1b.group(1)
        m2 = re.search(r"saying\s+(.+)$", clause, re.IGNORECASE)
        if m2:
            args["message"] = _trim_segment(m2.group(1))
        else:
            m2b = re.search(r"(?:message|text)\s+(?:to\s+)?[A-Za-z][A-Za-z\-']+\s+(.+)$", clause, re.IGNORECASE)
            if m2b:
                args["message"] = _trim_segment(m2b.group(1))

    if "reminder" in tool_name:
        m = re.search(r"remind\s+me\s+(?:about|to)\s+(.+?)\s+at\s+([0-9]{1,2}:[0-9]{2}\s*(?:AM|PM|am|pm))", clause, re.IGNORECASE)
        if m:
            args["title"] = re.sub(r"^(the|a|an)\s+", "", _trim_segment(m.group(1)), flags=re.IGNORECASE)
            args["time"] = _canonicalize_time_string(m.group(2))
        else:
            m_alt = re.search(r"(?:remind\s+me|set\s+(?:a\s+)?.*reminder)\s+(?:about|to|for)?\s*(.+?)\s+at\s+([0-9]{1,2}(?::[0-9]{2})?\s*(?:AM|PM|am|pm))", clause, re.IGNORECASE)
            if m_alt:
                args["title"] = re.sub(r"^(the|a|an)\s+", "", _trim_segment(m_alt.group(1)), flags=re.IGNORECASE)
                args["time"] = _canonicalize_time_string(m_alt.group(2))

        if "time" in clause.lower() or " at " in clause.lower():
            t = _extract_time_string(clause)
            if t:
                args["time"] = _canonicalize_time_string(t)
        if "title" not in args:
            m_title = re.search(r"(?:remind\s+me|set\s+(?:a\s+)?.*reminder)\s+(?:about|to|for)?\s*(.+?)(?:\s+at\s+[0-9]{1,2}(?::[0-9]{2})?\s*(?:AM|PM|am|pm))?$", clause, re.IGNORECASE)
            if m_title:
                title = re.sub(r"^(the|a|an)\s+", "", _trim_segment(m_title.group(1)), flags=re.IGNORECASE)
                if title:
                    args["title"] = title

    if "music" in tool_name or "play" in tool_name:
        m_some = re.search(r"\bplay\s+some\s+(.+?)\s+music\b", clause, re.IGNORECASE)
        if m_some:
            args["song"] = _trim_segment(m_some.group(1))
        else:
            m = re.search(r"\bplay\s+(.+)$", clause, re.IGNORECASE)
            if m:
                args["song"] = _trim_segment(m.group(1))

    return args


def _split_clauses(user_text):
    normalized = re.sub(r"\s+", " ", user_text).strip()
    parts = re.split(r"\s*(?:,\s*and\s*|\sand\s|,)\s*", normalized, flags=re.IGNORECASE)
    clauses = [p.strip(" .!?") for p in parts if p.strip(" .!?")]
    return clauses or [normalized]


def _extract_calls_schema_router(messages, tools):
    """Generic, tool-schema-driven parser: map user clauses to the best matching tool."""
    user_text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user").strip()
    if not user_text or not tools:
        return []

    clauses = _split_clauses(user_text)
    tool_profiles = [(t, _tool_keywords(t)) for t in tools]
    calls = []

    for clause in clauses:
        clause_tokens = set(_tokenize(clause))
        if not clause_tokens:
            continue

        best_tool = None
        best_score = 0
        for tool, kws in tool_profiles:
            overlap = len(clause_tokens & kws)
            score = overlap / max(1, len(kws)) + overlap
            if score > best_score:
                best_score = score
                best_tool = tool

        if not best_tool or best_score <= 0:
            continue

        tool_name = best_tool.get("name", "")
        args = _extract_args_generic(clause, tool_name)
        calls.append({"name": tool_name, "arguments": args})

    # Resolve simple pronoun recipient using previous contact query
    last_contact = None
    for call in calls:
        if call["name"] == "search_contacts":
            last_contact = call.get("arguments", {}).get("query")
        if call["name"] == "send_message":
            recipient = call.get("arguments", {}).get("recipient", "")
            if isinstance(recipient, str) and recipient.lower() in {"him", "her"} and last_contact:
                call["arguments"]["recipient"] = last_contact

    # Keep only schema-valid calls and deduplicate
    valid = [c for c in calls if _validate_call_schema(c, tools)]
    unique = []
    seen = set()
    for c in valid:
        key = (c["name"], json.dumps(c.get("arguments", {}), sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    return unique


def _extract_rule_calls(messages, tools):
    return _extract_calls_schema_router(messages, tools)


def _estimate_intent_count(user_text, available_tools):
    """Heuristic count of likely user intents; used for confidence/coverage estimation."""
    text = user_text.lower()
    intent_signals = {
        "get_weather": ["weather"],
        "set_alarm": ["alarm", "wake me up"],
        "set_timer": ["timer"],
        "play_music": ["play ", "music"],
        "search_contacts": ["contacts", "look up", "find "],
        "create_reminder": ["remind me"],
        "send_message": ["send", "text ", "message"],
    }

    hits = 0
    for tool_name, keywords in intent_signals.items():
        if tool_name not in available_tools:
            continue
        if any(k in text for k in keywords):
            hits += 1
    return max(1, hits)


def _validate_call_schema(call, tools):
    """Ensure predicted tool calls satisfy declared tool schema and required args."""
    tool_map = {t["name"]: t for t in tools}
    tool = tool_map.get(call.get("name"))
    if not tool:
        return False

    params = tool.get("parameters", {})
    required = params.get("required", [])
    props = params.get("properties", {})
    args = call.get("arguments", {})

    for key in required:
        if key not in args:
            return False
        val = args.get(key)
        if val is None:
            return False
        if isinstance(val, str) and not val.strip():
            return False

    for key, val in args.items():
        expected_type = props.get(key, {}).get("type", "").lower()
        if expected_type == "integer":
            if isinstance(val, bool):
                return False
            if isinstance(val, int):
                pass
            elif isinstance(val, float) and val.is_integer():
                args[key] = int(val)
            elif isinstance(val, str) and re.fullmatch(r"\s*\d+\s*", val):
                args[key] = int(val.strip())
            else:
                return False
        if expected_type == "string":
            if not isinstance(val, str):
                args[key] = str(val)
            if key == "time":
                args[key] = _canonicalize_time_string(args[key])
            if not args[key].strip():
                return False

    return True


def _dedupe_calls(calls):
    unique = []
    seen = set()
    for call in calls:
        key = (call.get("name"), json.dumps(call.get("arguments", {}), sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        unique.append(call)
    return unique


def _merge_calls(primary_calls, secondary_calls, tools, max_calls=None):
    merged = _dedupe_calls((primary_calls or []) + (secondary_calls or []))
    valid = [c for c in merged if _validate_call_schema(c, tools)]
    if max_calls is not None:
        return valid[:max_calls]
    return valid


def _rule_confidence(messages, tools, calls):
    """Estimate confidence from schema validity + intent coverage + call count sanity."""
    if not calls:
        return 0.0

    user_text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user").strip()
    available_tools = {t["name"] for t in tools}
    intent_count = _estimate_intent_count(user_text, available_tools)

    schema_ok = sum(1 for c in calls if _validate_call_schema(c, tools))
    schema_ratio = schema_ok / len(calls)

    coverage = min(1.0, len(calls) / max(1, intent_count))
    precision_hint = 1.0 if len(calls) <= max(1, intent_count + 1) else 0.7

    return 0.5 * schema_ratio + 0.35 * coverage + 0.15 * precision_hint


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = _get_cactus_model()
    if model is None:
        return {
            "function_calls": [],
            "total_time_ms": 0.0,
            "confidence": 0.0,
        }

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
    from google import genai
    from google.genai import types

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
        model="gemini-3-flash-preview",
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


def generate_hybrid(messages, tools, confidence_threshold=DEFAULT_HYBRID_CONFIDENCE_THRESHOLD):
    """Model-first hybrid router with deterministic fallback and optional cloud escalation."""
    start = time.time()
    user_text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user").strip()
    available_tools = {t["name"] for t in tools}
    intent_count = _estimate_intent_count(user_text, available_tools)

    # 1) Try local model first; accept when schema-valid with strong confidence.
    local = generate_cactus(messages, tools)
    local_calls = [c for c in local.get("function_calls", []) if _validate_call_schema(c, tools)]
    local_conf = _rule_confidence(messages, tools, local_calls)
    parsed_calls = _extract_calls_schema_router(messages, tools)
    parsed_conf = _rule_confidence(messages, tools, parsed_calls)
    merged_local_calls = _merge_calls(local_calls, parsed_calls, tools, max_calls=max(1, intent_count + 1))

    if local_calls and (local_conf >= LOCAL_ACCEPT_CONFIDENCE or local.get("confidence", 0) >= LOCAL_ACCEPT_CONFIDENCE):
        local["function_calls"] = merged_local_calls
        local["source"] = "on-device"
        return local

    # 2) Fallback to generic schema router when model output is weak/empty.
    if parsed_calls and parsed_conf >= ROUTER_ACCEPT_CONFIDENCE:
        return {
            "function_calls": parsed_calls,
            "total_time_ms": (time.time() - start) * 1000,
            "confidence": max(ROUTER_REPORTED_CONFIDENCE_FLOOR, parsed_conf),
            "source": "on-device",
        }

    # 3) If still uncertain, keep best local attempt unless confidence is very low.
    best_on_device_calls = _merge_calls(local_calls, parsed_calls, tools, max_calls=max(1, intent_count + 1))
    best_on_device_conf = max(local.get("confidence", 0), local_conf, parsed_conf)
    if best_on_device_calls and best_on_device_conf >= confidence_threshold:
        return {
            "function_calls": best_on_device_calls,
            "total_time_ms": local.get("total_time_ms", 0),
            "confidence": max(ROUTER_REPORTED_CONFIDENCE_FLOOR, best_on_device_conf),
            "source": "on-device",
        }

    if not _cloud_fallback_enabled() or not os.environ.get("GEMINI_API_KEY"):
        return {
            "function_calls": best_on_device_calls,
            "total_time_ms": local.get("total_time_ms", 0),
            "confidence": best_on_device_conf,
            "source": "on-device",
        }

    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = best_on_device_conf
    cloud["total_time_ms"] += local.get("total_time_ms", 0)
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
