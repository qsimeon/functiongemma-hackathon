# Source: https://github.com/SahilSaxena007/functiongemma-hackathon
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import atexit
import json, os, re, time
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
from google import genai
from google.genai import types

DEBUG = True

INTENT_HINTS = {
    "weather": "get_weather",
    "alarm": "set_alarm",
    "wake": "set_alarm",
    "timer": "set_timer",
    "play": "play_music",
    "music": "play_music",
    "remind": "create_reminder",
    "message": "send_message",
    "text": "send_message",
    "find": "search_contacts",
    "search": "search_contacts",
    "look up": "search_contacts",
    "calendar": "create_calendar_event",
    "schedule": "create_calendar_event",
    "book": "create_calendar_event",
    "appointment": "create_calendar_event",
    "meeting": "create_calendar_event",
    "event": "create_calendar_event",
    "reschedule": "reschedule_calendar_event",
    "move": "reschedule_calendar_event",
    "update": "reschedule_calendar_event",
    "cancel": "delete_calendar_event",
    "delete": "delete_calendar_event",
    "remove": "delete_calendar_event",
    "agenda": "list_calendar_events",
    "list": "list_calendar_events",
    "show": "list_calendar_events",
}

ACTION_LEADERS = [
    "get",
    "check",
    "what",
    "set",
    "wake",
    "play",
    "remind",
    "send",
    "text",
    "find",
    "search",
    "look",
    "schedule",
    "book",
    "create",
    "add",
    "move",
    "update",
    "reschedule",
    "cancel",
    "delete",
    "remove",
    "list",
    "show",
]


def _debug(*args):
    if DEBUG:
        print("[DBG]", *args, flush=True)


def _looks_garbled(text):
    if not isinstance(text, str) or not text:
        return False
    if "<0x" in text:
        return True
    printable = sum(1 for c in text if c.isprintable()) / max(1, len(text))
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(1, len(text))
    return printable < 0.85 or ascii_ratio < 0.55


# =====================================================================
#  GLOBAL MODEL CACHE (large latency win)
# =====================================================================

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        _debug("Initializing FunctionGemma...")
        _MODEL = cactus_init(functiongemma_path)
    return _MODEL


def _destroy_model():
    global _MODEL
    if _MODEL is not None:
        try:
            cactus_destroy(_MODEL)
            _debug("Destroyed FunctionGemma model")
        except Exception as exc:
            _debug("Model destroy error:", str(exc))
        finally:
            _MODEL = None


atexit.register(_destroy_model)


# =====================================================================
#  Generation - ON DEVICE
# =====================================================================

def _run_cactus_once(messages, tools, system_prompt):
    model = _get_model()
    try:
        cactus_reset(model)
    except Exception:
        pass

    cactus_tools = [{"type": "function", "function": t} for t in tools]
    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": system_prompt}] + messages,
        tools=cactus_tools,
        force_tools=True,
        tool_rag_top_k=0,
        confidence_threshold=0.0,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    try:
        raw = json.loads(raw_str)
    except Exception:
        _debug("CACTUS JSON FAIL:", raw_str[:240])
        return {"function_calls": [], "confidence": 0, "total_time_ms": 0, "cloud_handoff": False, "response": raw_str}

    _debug("CACTUS RAW PAYLOAD:", raw_str[:1200])
    if raw.get("function_calls"):
        _debug("CACTUS STRUCTURED CALLS:", json.dumps(raw.get("function_calls"), ensure_ascii=False))
    if raw.get("response") is not None:
        _debug("CACTUS RESPONSE TEXT:", str(raw.get("response"))[:600])

    if not raw.get("function_calls") and raw.get("response"):
        extracted = _extract_calls_from_response(raw.get("response"), tools)
        if extracted:
            raw["function_calls"] = extracted
            _debug("Recovered calls from response text:", extracted)
        else:
            _debug("No structured calls. response_snippet:", raw.get("response", "")[:220])

    return raw


def generate_cactus(messages, tools, system_prompt):
    raw = _run_cactus_once(messages, tools, system_prompt)

    # If output looks corrupted, recreate model and retry once.
    if (not raw.get("function_calls")) and _looks_garbled(raw.get("response", "")):
        _debug("Detected garbled local output. Reinitializing model and retrying once.")
        _destroy_model()
        raw = _run_cactus_once(messages, tools, system_prompt)

    _debug(
        f"cactus -> handoff={raw.get('cloud_handoff')} "
        f"calls={raw.get('function_calls')} "
        f"conf={raw.get('confidence', 0):.3f} "
        f"time={raw.get('total_time_ms', 0):.0f}ms"
    )

    return raw


# =====================================================================
#  Generation - CLOUD
# =====================================================================

def generate_cloud(messages, tools):
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper())
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]
    start = time.time()

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(tools=gemini_tools),
    )

    total_time_ms = (time.time() - start) * 1000
    calls = []
    for cand in resp.candidates:
        for part in cand.content.parts:
            if part.function_call:
                calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                })

    _debug(f"cloud -> calls={calls} time={total_time_ms:.0f}ms")
    return {"function_calls": calls, "total_time_ms": total_time_ms}


# =====================================================================
#  Prompts
# =====================================================================

def _strict_prompt(tools):
    names = ", ".join(t["name"] for t in tools)
    return (
        "You are a tool-calling assistant.\\n"
        f"Available functions: {names}\\n"
        "Use the provided functions when appropriate.\\n"
        "If multiple actions are requested, return multiple function calls in order.\\n"
        "Do not output random text."
    )


def _repair_prompt(tools):
    names = ", ".join(t["name"] for t in tools)
    return (
        "You are a function-calling assistant.\\n"
        f"Allowed tools: {names}\\n"
        "Return only tool calls. Include all requested actions and required arguments."
    )


def _split_instructions(text):
    if not isinstance(text, str):
        return []
    text = text.strip()
    if not text:
        return []
    normalized = re.sub(r"\s+", " ", text)
    normalized = re.sub(r"\b(?:then|after that|also)\b", ",", normalized, flags=re.IGNORECASE)
    comma_parts = [p.strip(" .") for p in re.split(r"\s*,\s*", normalized) if p.strip(" .")]
    clauses = []
    split_re = re.compile(
        r"\s+and\s+(?=(?:"
        + "|".join(ACTION_LEADERS)
        + r")\b)",
        flags=re.IGNORECASE,
    )
    for part in comma_parts:
        subparts = []
        for s in re.split(split_re, part):
            cleaned = re.sub(r"^(and|then)\s+", "", s.strip(" ."), flags=re.IGNORECASE)
            if cleaned:
                subparts.append(cleaned)
        clauses.extend(subparts)
    return clauses or [text.strip(" .")]


def _find_tool(tools, tool_name):
    for tool in tools:
        if tool.get("name") == tool_name:
            return tool
    return None


def _intent_tool_names(clause, tools):
    available = {t["name"] for t in tools}
    text = (clause or "").lower()
    names = []
    for hint, tool_name in INTENT_HINTS.items():
        if tool_name not in available:
            continue
        if " " in hint:
            if hint in text:
                names.append(tool_name)
        else:
            if re.search(rf"\b{re.escape(hint)}\b", text):
                names.append(tool_name)
    if names:
        deduped = []
        for n in names:
            if n not in deduped:
                deduped.append(n)
        return deduped

    picked = _pick_tool_for_clause(tools, clause)
    if picked:
        return [picked["name"]]
    return []


def _parse_clock_time(text):
    if not isinstance(text, str):
        return None
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", text, flags=re.IGNORECASE)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    meridiem = (m.group(3) or "").upper()
    return {
        "hour": hour,
        "minute": minute,
        "meridiem": meridiem,
    }


def _extract_weather_location(clause):
    m = re.search(r"\bweather(?:\s+like)?\s+in\s+([A-Za-z][A-Za-z .'-]+)", clause, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\bin\s+([A-Za-z][A-Za-z .'-]+)", clause, flags=re.IGNORECASE)
        if not m:
            return None
    location = m.group(1).strip(" .!?\"'")
    location = re.sub(r"^(the)\s+", "", location, flags=re.IGNORECASE)
    return location if location else None


def _extract_timer_minutes(clause):
    m = re.search(r"\b(\d+)\s*(?:minutes?|mins?)\b", clause, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\bset\s+(?:a\s+)?timer\s+for\s+(\d+)\b", clause, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _extract_search_name(clause):
    m = re.search(r"\b(?:find|search(?:\s+for)?|look\s+up)\s+([A-Za-z][A-Za-z'-]*)\b", clause, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip(" .!?\"'")


def _extract_message_recipient(clause, context):
    m = re.search(r"\b(?:to|text)\s+([A-Za-z][A-Za-z'-]*)\b", clause, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(" .!?\"'")
    m = re.search(r"\bsend\s+(him|her|them)\b", clause, flags=re.IGNORECASE)
    if m and context.get("last_contact"):
        return context["last_contact"]
    return None


def _extract_message_body(clause):
    m = re.search(r"\b(?:saying|says|that says)\s+(.+)$", clause, flags=re.IGNORECASE)
    if not m:
        return None
    body = m.group(1).strip(" .!?\"'")
    return body if body else None


def _extract_music_target(clause):
    m = re.search(r"\bplay\s+(.+)$", clause, flags=re.IGNORECASE)
    if not m:
        return None
    song = m.group(1).strip(" .!?\"'")
    if re.match(r"^some\s+.+\s+music$", song, flags=re.IGNORECASE):
        song = re.sub(r"^some\s+", "", song, flags=re.IGNORECASE)
        song = re.sub(r"\s+music$", "", song, flags=re.IGNORECASE)
    return song if song else None


def _extract_reminder_title_and_time(clause):
    time_info = _parse_clock_time(clause)
    if not time_info:
        return None, None
    title_match = re.search(r"\bremind\s+me\s+(?:about|to)\s+(.+?)(?:\s+at\b|$)", clause, flags=re.IGNORECASE)
    if not title_match:
        return None, None
    title = title_match.group(1).strip(" .!?\"'")
    title = re.sub(r"^the\s+", "", title, flags=re.IGNORECASE)
    minute = f"{time_info['minute']:02d}"
    meridiem = f" {time_info['meridiem']}" if time_info["meridiem"] else ""
    time_text = f"{time_info['hour']}:{minute}{meridiem}"
    return title, time_text


def _extract_date_phrase(text):
    if not isinstance(text, str):
        return None
    patterns = [
        r"\bday after tomorrow\b",
        r"\btomorrow\b",
        r"\btoday\b",
        r"\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            return m.group(0)
    return None


def _extract_time_phrase(text):
    if not isinstance(text, str):
        return None
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", text, flags=re.IGNORECASE)
    if m:
        hour = m.group(1)
        minute = m.group(2) or "00"
        meridiem = m.group(3).upper()
        return f"{hour}:{minute} {meridiem}"
    m = re.search(r"\b(\d{1,2}):(\d{2})\b", text, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1)}:{m.group(2)}"
    return None


def _extract_duration_minutes(text):
    if not isinstance(text, str):
        return None
    m = re.search(r"\b(\d+)\s*(minutes?|mins?)\b", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(\d+)\s*(hours?|hrs?|hr)\b", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1)) * 60
    return None


def _extract_calendar_title(clause):
    if not isinstance(clause, str):
        return None
    original = clause.strip()
    text = original
    m = re.search(
        r"\b(?:schedule|add|book|create|put)\b\s+(?:an?\s+)?(?:calendar\s+)?(?:event|meeting|appointment)?\s*(?:for\s+)?(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        text = m.group(1)

    text = re.sub(r"\bon\s+my\s+calendar\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:today|tomorrow|day after tomorrow)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bat\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bon\s+\d{4}-\d{2}-\d{2}\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bon\s+\d{1,2}/\d{1,2}(?:/\d{2,4})?\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bfor\s+\d+\s*(?:minutes?|mins?|hours?|hrs?|hr)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" .,!?:;\"'")
    if text:
        return text
    return original.strip(" .,!?:;\"'") or "New event"


def _extract_reschedule_query(clause):
    if not isinstance(clause, str):
        return None
    before = re.split(r"\bto\b", clause, maxsplit=1, flags=re.IGNORECASE)[0]
    before = re.sub(r"\b(reschedule|move|update)\b", "", before, flags=re.IGNORECASE)
    before = re.sub(r"\b(my|calendar|event)\b", "", before, flags=re.IGNORECASE)
    before = re.sub(r"\s+", " ", before).strip(" .,!?:;\"'")
    return before if before else None


def _extract_delete_query(clause):
    if not isinstance(clause, str):
        return None
    text = re.sub(r"\b(cancel|delete|remove)\b", "", clause, flags=re.IGNORECASE)
    text = re.sub(r"\b(my|calendar|event)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(on|for)\s+\d{4}-\d{2}-\d{2}\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(on|for)\s+\d{1,2}/\d{1,2}(?:/\d{2,4})?\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(today|tomorrow|day after tomorrow)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" .,!?:;\"'")
    return text if text else None


def _deterministic_call_for_tool(clause, tool_name, context):
    clause = (clause or "").strip()
    if not clause:
        return None

    if tool_name == "get_weather":
        location = _extract_weather_location(clause)
        if location:
            return {"name": "get_weather", "arguments": {"location": location}}
        return None

    if tool_name == "set_alarm":
        time_info = _parse_clock_time(clause)
        if time_info:
            return {
                "name": "set_alarm",
                "arguments": {"hour": time_info["hour"], "minute": time_info["minute"]},
            }
        return None

    if tool_name == "set_timer":
        minutes = _extract_timer_minutes(clause)
        if minutes is not None:
            return {"name": "set_timer", "arguments": {"minutes": minutes}}
        return None

    if tool_name == "play_music":
        song = _extract_music_target(clause)
        if song:
            return {"name": "play_music", "arguments": {"song": song}}
        return None

    if tool_name == "create_reminder":
        title, time_text = _extract_reminder_title_and_time(clause)
        if title and time_text:
            return {"name": "create_reminder", "arguments": {"title": title, "time": time_text}}
        return None

    if tool_name == "search_contacts":
        query = _extract_search_name(clause)
        if query:
            return {"name": "search_contacts", "arguments": {"query": query}}
        return None

    if tool_name == "send_message":
        recipient = _extract_message_recipient(clause, context)
        message = _extract_message_body(clause)
        if recipient and message:
            return {"name": "send_message", "arguments": {"recipient": recipient, "message": message}}
        return None

    if tool_name == "create_calendar_event":
        title = _extract_calendar_title(clause)
        date_text = _extract_date_phrase(clause) or "today"
        time_text = _extract_time_phrase(clause) or "9:00 AM"
        duration = _extract_duration_minutes(clause) or 60
        return {
            "name": "create_calendar_event",
            "arguments": {
                "title": title,
                "date": date_text,
                "time": time_text,
                "duration_minutes": duration,
            },
        }

    if tool_name == "reschedule_calendar_event":
        query = _extract_reschedule_query(clause)
        split_parts = re.split(r"\bto\b", clause, maxsplit=1, flags=re.IGNORECASE)
        right = split_parts[1] if len(split_parts) == 2 else clause
        new_date = _extract_date_phrase(right) or _extract_date_phrase(clause)
        new_time = _extract_time_phrase(right) or _extract_time_phrase(clause)
        duration = _extract_duration_minutes(right or clause)
        if query and new_date and new_time:
            args = {"query": query, "new_date": new_date, "new_time": new_time}
            if duration:
                args["duration_minutes"] = duration
            return {"name": "reschedule_calendar_event", "arguments": args}
        return None

    if tool_name == "delete_calendar_event":
        query = _extract_delete_query(clause)
        date_text = _extract_date_phrase(clause)
        if query:
            args = {"query": query}
            if date_text:
                args["date"] = date_text
            return {"name": "delete_calendar_event", "arguments": args}
        return None

    if tool_name == "list_calendar_events":
        date_text = _extract_date_phrase(clause) or "today"
        return {"name": "list_calendar_events", "arguments": {"date": date_text}}

    return None


def _update_clause_context(call, context):
    if not call:
        return
    if call.get("name") == "search_contacts":
        query = call.get("arguments", {}).get("query")
        if isinstance(query, str) and query:
            context["last_contact"] = query
    if call.get("name") == "send_message":
        recipient = call.get("arguments", {}).get("recipient")
        if isinstance(recipient, str) and recipient:
            context["last_contact"] = recipient


def _route_clause_deterministically(clause, tools, context):
    for name in _intent_tool_names(clause, tools):
        call = _deterministic_call_for_tool(clause, name, context)
        if call and _validate_call(call, tools):
            return call
    return None


def _tokenize(text):
    return set(re.findall(r"[a-zA-Z_]+", (text or "").lower()))


def _score_tool_for_clause(tool, clause):
    clause_tokens = _tokenize(clause)
    if not clause_tokens:
        return 0

    name_tokens = _tokenize(tool.get("name", "").replace("_", " "))
    desc_tokens = _tokenize(tool.get("description", ""))
    param_tokens = set()
    for p in tool.get("parameters", {}).get("properties", {}).keys():
        param_tokens |= _tokenize(p.replace("_", " "))

    # weighted lexical overlap
    return (
        3 * len(clause_tokens & name_tokens)
        + 2 * len(clause_tokens & desc_tokens)
        + 1 * len(clause_tokens & param_tokens)
    )


def _pick_tool_for_clause(tools, clause):
    scored = sorted(
        (( _score_tool_for_clause(t, clause), t) for t in tools),
        key=lambda x: x[0],
        reverse=True,
    )
    if not scored:
        return None
    # If everything scores 0, still return first tool as fallback.
    return scored[0][1]


def _extract_calls_from_response(response_text, tools):
    if not isinstance(response_text, str) or not response_text.strip():
        return []

    tool_names = {t["name"] for t in tools}

    try:
        data = json.loads(response_text)
        if isinstance(data, dict):
            if isinstance(data.get("function_calls"), list):
                calls = [c for c in data["function_calls"] if isinstance(c, dict)]
                return [c for c in calls if c.get("name") in tool_names]
            if data.get("name") in tool_names:
                return [data]
        elif isinstance(data, list):
            calls = [c for c in data if isinstance(c, dict) and c.get("name") in tool_names]
            if calls:
                return calls
    except Exception:
        pass

    # Fallback: find simple JSON object patterns containing "name" and "arguments".
    matches = re.findall(r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}\s*\}', response_text)
    calls = []
    for m in matches:
        try:
            c = json.loads(m)
            if c.get("name") in tool_names:
                calls.append(c)
        except Exception:
            pass
    return calls


def _candidate_tools_for_clause(tools, clause):
    names = _intent_tool_names(clause, tools)
    ordered = []
    for name in names:
        tool = _find_tool(tools, name)
        if tool and tool not in ordered:
            ordered.append(tool)
    for tool in tools:
        if tool not in ordered:
            ordered.append(tool)
    return ordered


def _dedupe_calls(calls):
    seen = set()
    out = []
    for call in calls:
        key = json.dumps(call, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(call)
    return out


def _align_calls_to_clauses(calls, clauses, tools):
    if not calls:
        return []
    if len(clauses) <= 1:
        return [calls[0]]
    available = list(enumerate(calls))
    ordered = []
    for clause in clauses:
        hint_names = _intent_tool_names(clause, tools)
        chosen_idx = None

        for idx, call in available:
            if call.get("name") in hint_names:
                chosen_idx = idx
                break

        if chosen_idx is None:
            best_score = -1
            best_idx = None
            for idx, call in available:
                tool = _find_tool(tools, call.get("name"))
                if not tool:
                    continue
                score = _score_tool_for_clause(tool, clause)
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is not None and best_score > 0:
                chosen_idx = best_idx

        if chosen_idx is None:
            return []

        chosen = calls[chosen_idx]
        ordered.append(chosen)
        available = [(idx, c) for idx, c in available if idx != chosen_idx]

    return ordered


# =====================================================================
#  Validation / normalization
# =====================================================================

def _clean_string_arg(value):
    if not isinstance(value, str):
        return value
    value = re.sub(r"\\s+", " ", value.strip())
    value = value.strip("\"'")
    value = re.sub(r"[.!?]+$", "", value)
    return value


def _normalize_calls(calls, tools):
    tool_map = {t["name"]: t for t in tools}
    out = []
    for call in calls:
        name = call.get("name")
        args = call.get("arguments", {})
        td = tool_map.get(name)
        if not td:
            continue
        props = td["parameters"].get("properties", {})
        fixed = {}
        for k, v in args.items():
            if k not in props:
                continue
            typ = props[k].get("type")
            if typ == "integer" and not isinstance(v, int):
                try:
                    fixed[k] = int(float(str(v)))
                except (TypeError, ValueError):
                    fixed[k] = v
            elif typ == "string":
                fixed[k] = _clean_string_arg(str(v))
            else:
                fixed[k] = v
        out.append({"name": name, "arguments": fixed})
    return out


def _validate_call(call, tools):
    if not isinstance(call, dict) or "name" not in call:
        return False
    for t in tools:
        if t["name"] == call["name"]:
            required = t["parameters"].get("required", [])
            args = call.get("arguments", {})
            return all((r in args and args[r] not in (None, "", "unknown")) for r in required)
    return False


# =====================================================================
#  HYBRID STRATEGY
# =====================================================================

def generate_hybrid(messages, tools, confidence_threshold=0.99):
    hybrid_start = time.perf_counter()
    user_msg = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")

    _debug("\\n" + "=" * 70)
    _debug("USER:", user_msg)
    _debug("TOOLS:", [t["name"] for t in tools])
    _debug("THRESHOLD_ARG:", confidence_threshold)

    clauses = _split_instructions(user_msg)
    desired_calls = max(1, len(clauses))
    _debug("CLAUSES:", clauses)

    # Stage 1: deterministic intent-hint routing (zero inference latency).
    context = {}
    deterministic_calls = []
    unresolved_clause = False
    for clause in clauses:
        call = _route_clause_deterministically(clause, tools, context)
        if call and _validate_call(call, tools):
            deterministic_calls.append(call)
            _update_clause_context(call, context)
        else:
            unresolved_clause = True

    deterministic_calls = _dedupe_calls(_normalize_calls(deterministic_calls, tools))
    if deterministic_calls and not unresolved_clause and len(deterministic_calls) >= desired_calls:
        _debug("ACCEPT DETERMINISTIC LOCAL")
        return {
            "function_calls": deterministic_calls[:desired_calls],
            "total_time_ms": (time.perf_counter() - hybrid_start) * 1000,
            "source": "on-device",
            "confidence": 1.0,
        }

    # Stage 2: full-query local attempt.
    local = generate_cactus(messages, tools, _strict_prompt(tools))

    calls = _normalize_calls(local.get("function_calls", []), tools)
    conf = local.get("confidence", 0)
    handoff = local.get("cloud_handoff", False)
    local_time = local.get("total_time_ms", 0)

    valid = _dedupe_calls([c for c in calls if _validate_call(c, tools)])
    aligned = _align_calls_to_clauses(valid, clauses, tools)

    _debug(
        "LOCAL CHECK:",
        {
            "raw_calls": len(local.get("function_calls", [])),
            "valid_calls": len(valid),
            "aligned_calls": len(aligned),
            "confidence": round(conf, 4),
            "handoff": handoff,
        },
    )

    if aligned and len(aligned) >= desired_calls:
        _debug("ACCEPT FULL-QUERY LOCAL")
        return {
            "function_calls": aligned[:desired_calls],
            "total_time_ms": local_time,
            "source": "on-device",
            "confidence": conf,
        }

    # Stage 3: per-clause local routing with deterministic parse + narrowed tools.
    _debug("CLAUSE ROUTER LOCAL")
    clause_context = {}
    clause_calls = []
    clause_time_ms = 0.0
    for clause in clauses:
        deterministic = _route_clause_deterministically(clause, tools, clause_context)
        if deterministic and _validate_call(deterministic, tools):
            clause_calls.append(deterministic)
            _update_clause_context(deterministic, clause_context)
            continue

        candidate_tools = _candidate_tools_for_clause(tools, clause)
        picked = candidate_tools[:2] if len(candidate_tools) > 2 else candidate_tools
        clause_local = generate_cactus(
            [{"role": "user", "content": clause}],
            picked,
            _repair_prompt(picked),
        )
        clause_time_ms += clause_local.get("total_time_ms", 0)
        ccalls = _normalize_calls(clause_local.get("function_calls", []), picked)
        cvalid = [c for c in ccalls if _validate_call(c, tools)]
        if not cvalid and picked != tools:
            fallback_local = generate_cactus(
                [{"role": "user", "content": clause}],
                tools,
                _repair_prompt(tools),
            )
            clause_time_ms += fallback_local.get("total_time_ms", 0)
            fcalls = _normalize_calls(fallback_local.get("function_calls", []), tools)
            cvalid = [c for c in fcalls if _validate_call(c, tools)]
        if not cvalid:
            unresolved_clause = True
            continue

        best = _align_calls_to_clauses(cvalid, [clause], tools)
        chosen_call = best[0] if best else cvalid[0]
        clause_calls.append(chosen_call)
        _update_clause_context(chosen_call, clause_context)

    clause_calls = _dedupe_calls(_normalize_calls(clause_calls, tools))
    aligned_clause_calls = _align_calls_to_clauses(clause_calls, clauses, tools)
    if aligned_clause_calls and len(aligned_clause_calls) >= desired_calls:
        _debug("ACCEPT CLAUSE-ROUTED LOCAL")
        return {
            "function_calls": aligned_clause_calls[:desired_calls],
            "total_time_ms": local_time + clause_time_ms,
            "source": "on-device",
            "confidence": max(conf, 0.8),
        }

    if handoff:
        _debug("CACTUS SUGGESTED HANDOFF")

    # Stage 4: cloud fallback only if key exists.
    if not os.environ.get("GEMINI_API_KEY"):
        _debug("NO GEMINI_API_KEY; RETURN BEST LOCAL")
        best_local = aligned if aligned else valid
        return {
            "function_calls": best_local[:desired_calls] if best_local else [],
            "total_time_ms": local_time + clause_time_ms,
            "source": "on-device",
            "confidence": conf,
            "local_confidence": conf,
        }

    _debug("FALLBACK -> CLOUD")
    cloud = generate_cloud(messages, tools)

    return {
        "function_calls": cloud["function_calls"],
        "total_time_ms": cloud["total_time_ms"] + local_time + clause_time_ms,
        "source": "cloud (fallback)",
        "confidence": conf,
        "local_confidence": conf,
    }


# =====================================================================

def print_result(label, result):
    print(f"\\n=== {label} ===")
    print("Source:", result.get("source"))
    print("Confidence:", round(result.get("confidence", 0), 4))
    print("Time:", round(result.get("total_time_ms", 0), 2), "ms")
    for call in result.get("function_calls", []):
        print(call)


if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"],
        },
    }]

    messages = [{"role": "user", "content": "What is the weather in San Francisco?"}]

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid", hybrid)
