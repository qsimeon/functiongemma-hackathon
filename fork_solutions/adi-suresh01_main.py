# Source: https://github.com/adi-suresh01/functiongemma-hackathon
import sys
from pathlib import Path
import json, os, re, time
from difflib import SequenceMatcher


def _configure_cactus_import_path():
    root = Path(__file__).resolve().parent
    candidates = []

    env_path = os.environ.get("CACTUS_PYTHON_SRC")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.extend([
        root / "cactus" / "python" / "src",
        root.parent / "cactus" / "python" / "src",
    ])

    for path in candidates:
        if (path / "cactus.py").exists() or (path / "cactus").is_dir():
            sys.path.insert(0, str(path))
            return


def _resolve_functiongemma_path():
    root = Path(__file__).resolve().parent
    candidates = []

    env_path = os.environ.get("CACTUS_FUNCTIONGEMMA_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.extend([
        root / "cactus" / "weights" / "functiongemma-270m-it",
        root.parent / "cactus" / "weights" / "functiongemma-270m-it",
    ])

    for path in candidates:
        if path.is_dir():
            return str(path)

    # Keep backward compatibility with original relative path.
    return "cactus/weights/functiongemma-270m-it"


_configure_cactus_import_path()
functiongemma_path = _resolve_functiongemma_path()

try:
    from cactus import cactus_init, cactus_complete, cactus_destroy
except ModuleNotFoundError as exc:
    if exc.name != "cactus":
        raise
    raise ModuleNotFoundError(
        "Could not import 'cactus'. Set CACTUS_PYTHON_SRC to your cactus/python/src path "
        "(for example: /Users/adi/Desktop/cactus/python/src), or clone cactus into "
        "./cactus inside this repo."
    ) from exc


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
    try:
        from google import genai
        from google.genai import types
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Cloud fallback requires the `google-genai` package. "
            "Install it with: pip install google-genai"
        ) from exc

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Cloud fallback requires GEMINI_API_KEY to be set.")

    client = genai.Client(api_key=api_key)

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


_STOPWORDS = {
    "a", "an", "and", "at", "be", "by", "for", "from", "get", "in", "is", "it",
    "me", "my", "of", "on", "or", "the", "to", "up", "with", "what", "whats",
    "what's", "please", "current", "like", "can", "you",
}

_ACTION_HINTS = {
    "weather": ["weather", "forecast", "temperature", "rain", "sunny", "city", "location"],
    "alarm": ["alarm", "wake", "wake up", "morning", "hour", "minute", "clock"],
    "timer": ["timer", "countdown", "minutes", "seconds", "duration"],
    "message": ["message", "text", "sms", "send", "recipient", "contact"],
    "reminder": ["remind", "reminder", "remember", "title", "time"],
    "search": ["search", "find", "lookup", "look up", "contact", "query"],
    "contact": ["contact", "contacts", "person", "name", "find", "lookup"],
    "play": ["play", "music", "song", "playlist", "listen", "track"],
}


def _extract_user_text(messages):
    parts = []
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str):
                parts.append(content.strip())
    return " ".join(p for p in parts if p).strip()


def _norm_key(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _split_words(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _clean_span(text):
    return str(text).strip().strip("\"'").strip(".,!?;:").strip()


def _tool_schema(tool):
    params = tool.get("parameters", {})
    if not isinstance(params, dict):
        return {}, []
    properties = params.get("properties", {})
    required = params.get("required", [])
    if not isinstance(properties, dict):
        properties = {}
    if not isinstance(required, list):
        required = []
    return properties, required


def _match_tool_name(name, tools_by_name):
    if name in tools_by_name:
        return name
    if not name:
        return None
    target = _norm_key(name)
    for candidate in tools_by_name:
        if _norm_key(candidate) == target:
            return candidate
    best_name, best_ratio = None, 0.0
    for candidate in tools_by_name:
        ratio = SequenceMatcher(None, target, _norm_key(candidate)).ratio()
        if ratio > best_ratio:
            best_name, best_ratio = candidate, ratio
    if best_ratio >= 0.78:
        return best_name
    return None


def _match_arg_key(key, properties):
    if key in properties:
        return key
    target = _norm_key(key)
    if not target:
        return None

    for candidate in properties:
        if _norm_key(candidate) == target:
            return candidate

    best_key, best_ratio = None, 0.0
    for candidate in properties:
        ratio = SequenceMatcher(None, target, _norm_key(candidate)).ratio()
        if ratio > best_ratio:
            best_key, best_ratio = candidate, ratio
    if best_ratio >= 0.75:
        return best_key
    return None


def _extract_time_tuple(text):
    match = re.search(r"\b(\d{1,2})(?::([0-5]\d))?\s*(am|pm)\b", text, flags=re.IGNORECASE)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        am_pm = match.group(3).lower()
        if am_pm == "am" and hour == 12:
            hour = 0
        return hour, minute

    match = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)
    if match:
        return int(match.group(1)), int(match.group(2))

    return None


def _extract_time_text(text):
    match = re.search(r"\b(\d{1,2})(?::([0-5]\d))?\s*(AM|PM)\b", text, flags=re.IGNORECASE)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        am_pm = match.group(3).upper()
        return f"{hour}:{minute:02d} {am_pm}"

    match = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)
    if match:
        return f"{int(match.group(1))}:{int(match.group(2)):02d}"

    return None


def _extract_all_time_texts(text):
    times = []
    for match in re.finditer(r"\b(\d{1,2})(?::([0-5]\d))?\s*(AM|PM)\b", text, flags=re.IGNORECASE):
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        am_pm = match.group(3).upper()
        times.append(f"{hour}:{minute:02d} {am_pm}")
    for match in re.finditer(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text):
        times.append(f"{int(match.group(1))}:{int(match.group(2)):02d}")
    return times


def _extract_int_for_key(key, text):
    key_l = key.lower()
    time_tuple = _extract_time_tuple(text)

    if key_l == "hour" and time_tuple:
        return time_tuple[0]
    if key_l == "minute" and time_tuple:
        return time_tuple[1]

    if "minute" in key_l:
        match = re.search(r"(-?\d+)\s*(?:minutes?|mins?)\b", text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

    if "second" in key_l:
        match = re.search(r"(-?\d+)\s*(?:seconds?|secs?)\b", text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

    match = re.search(r"-?\d+", text)
    if match:
        return int(match.group(0))
    return None


def _extract_string_for_key(key, call_name, text):
    key_l = key.lower()
    call_l = str(call_name).lower()

    if key_l in {"location", "city", "place"}:
        patterns = [
            r"(?:weather|forecast|temperature)\s+(?:in|at|for)\s+([a-z0-9' \-]+?)(?:[,.!?]|$|\band\b)",
            r"(?:in|at|for)\s+([a-z0-9' \-]+?)(?:[,.!?]|$|\band\b)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return _clean_span(match.group(1)).title()

    if key_l in {"recipient", "contact", "query", "name", "person"}:
        patterns = [
            r"(?:message|text|send(?:\s+a)?\s+message)\s+to\s+([a-z][a-z' \-]+?)(?:\s+saying\b|[,.!?]|$|\band\b)",
            r"\btext\s+([a-z][a-z' \-]+?)(?:\s+saying\b|[,.!?]|$|\band\b)",
            r"(?:find|look up|search(?: for)?)\s+([a-z][a-z' \-]+?)(?:\s+in\b|[,.!?]|$|\band\b)",
            r"\bto\s+([a-z][a-z' \-]+?)(?:\s+saying\b|[,.!?]|$|\band\b)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                candidate = _clean_span(match.group(1))
                if candidate and candidate.lower() not in {"him", "her", "them", "me", "you"}:
                    return candidate.title()

    if key_l in {"message", "text", "body", "content"}:
        patterns = [
            r"(?:saying|that says|saying that)\s+(.+?)(?:[,.!?]|$|\band\b)",
            r"(?:message|text)\s+(.+?)(?:[,.!?]|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return _clean_span(match.group(1))

    if key_l in {"song", "music", "track", "playlist"}:
        patterns = [
            r"\bplay\s+some\s+(.+?)\s+music(?:[,.!?]|$|\band\b)",
            r"\bplay\s+(?:some\s+)?(.+?)(?:[,.!?]|$|\band\b)",
            r"\blisten to\s+(.+?)(?:[,.!?]|$|\band\b)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return _clean_span(match.group(1))

    if key_l == "title" or ("reminder" in call_l and key_l in {"task", "subject"}):
        pattern = r"\bremind me(?:\s+to|\s+about)?\s+(.+?)(?:\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)\b|[,.!?]|$|\band\b)"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            title = _clean_span(match.group(1))
            title = re.sub(r"^\b(the|a|an)\b\s+", "", title, flags=re.IGNORECASE)
            return _clean_span(title)

    if key_l in {"time", "when", "datetime"}:
        return _extract_time_text(text)

    quote_match = re.search(r"\"([^\"]+)\"|'([^']+)'", text)
    if quote_match:
        return _clean_span(quote_match.group(1) or quote_match.group(2))

    return None


def _infer_argument(key, arg_type, text, call_name):
    arg_type_l = str(arg_type).lower()
    if arg_type_l in {"integer", "number"}:
        return _extract_int_for_key(key, text)
    if arg_type_l == "boolean":
        lower = text.lower()
        if re.search(r"\b(true|yes|on|enable)\b", lower):
            return True
        if re.search(r"\b(false|no|off|disable)\b", lower):
            return False
        return None
    return _extract_string_for_key(key, call_name, text)


def _coerce_argument(value, arg_type, key, user_text):
    arg_type_l = str(arg_type).lower()
    key_l = key.lower()

    if arg_type_l == "integer":
        if isinstance(value, bool):
            parsed = int(value)
        elif isinstance(value, int):
            parsed = value
        elif isinstance(value, float):
            parsed = int(round(value))
        elif isinstance(value, str):
            match = re.search(r"-?\d+", value)
            parsed = int(match.group(0)) if match else None
        else:
            parsed = None

        if parsed is None:
            parsed = _extract_int_for_key(key_l, user_text)
        if parsed is None:
            return None

        if key_l in {"minutes", "minute", "hour"} and parsed < 0:
            parsed = abs(parsed)

        if key_l == "minutes" and parsed >= 300 and re.search(r"\bminute", user_text, flags=re.IGNORECASE):
            if parsed % 60 == 0:
                parsed = parsed // 60

        if key_l == "minute" and parsed > 59:
            parsed = parsed % 60

        return parsed

    if arg_type_l == "number":
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            match = re.search(r"-?\d+(?:\.\d+)?", value)
            if match:
                return float(match.group(0))
        inferred = _extract_int_for_key(key_l, user_text)
        if inferred is not None:
            return float(inferred)
        return None

    if arg_type_l == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lower = value.strip().lower()
            if lower in {"true", "yes", "1", "on"}:
                return True
            if lower in {"false", "no", "0", "off"}:
                return False
        return None

    if value is None:
        inferred = _extract_string_for_key(key_l, "", user_text)
        return inferred if inferred is not None else ""

    if key_l == "time":
        value_s = str(value).strip()
        normalized = _extract_time_text(value_s)
        if normalized:
            return normalized
        # Reject malformed clock values (for example: 29:00) and recover from user text when possible.
        if re.search(r"\b\d{1,2}:\d{2}\b", value_s):
            inferred = _extract_time_text(user_text)
            return inferred if inferred is not None else None
        return value_s

    return str(value).strip()


def _sanitize_calls(calls, tools, user_text):
    tools_by_name = {t["name"]: t for t in tools}
    repaired = []

    for call in calls or []:
        if not isinstance(call, dict):
            continue

        name = _match_tool_name(call.get("name"), tools_by_name)
        if not name:
            continue

        tool = tools_by_name[name]
        properties, required = _tool_schema(tool)
        raw_args = call.get("arguments", {})
        if not isinstance(raw_args, dict):
            raw_args = {}

        args = {}
        for raw_key, raw_value in raw_args.items():
            key = _match_arg_key(raw_key, properties)
            if not key:
                continue
            expected_type = properties.get(key, {}).get("type", "string")
            coerced = _coerce_argument(raw_value, expected_type, key, user_text)
            if coerced is not None and coerced != "":
                args[key] = coerced

        for req_key in required:
            if req_key in args and args[req_key] not in ("", None):
                continue
            req_type = properties.get(req_key, {}).get("type", "string")
            inferred = _infer_argument(req_key, req_type, user_text, name)
            if inferred is None:
                continue
            coerced = _coerce_argument(inferred, req_type, req_key, user_text)
            if coerced is not None and coerced != "":
                args[req_key] = coerced

        if "hour" in args and "minute" in args:
            parsed_time = _extract_time_tuple(user_text)
            if parsed_time:
                args["hour"], args["minute"] = parsed_time

        # Prefer user-grounded values for common ambiguous slots.
        for key, schema in properties.items():
            if key not in args:
                continue
            key_l = key.lower()
            expected_type = schema.get("type", "string") if isinstance(schema, dict) else "string"
            if expected_type == "integer" and key_l in {"hour", "minute", "minutes"}:
                inferred_int = _extract_int_for_key(key_l, user_text)
                if inferred_int is not None:
                    args[key] = _coerce_argument(inferred_int, "integer", key_l, user_text)
            if expected_type == "string" and key_l in {"location", "recipient", "query", "song", "title", "message", "time"}:
                inferred_str = _extract_string_for_key(key_l, name, user_text)
                if inferred_str:
                    current_val = str(args.get(key, ""))
                    inferred_clean = _clean_span(inferred_str)
                    if key_l == "time":
                        current_clean = _clean_span(current_val)
                        suspicious_time = (
                            not current_clean or
                            re.search(r"\d{4}-\d{2}-\d{2}", current_clean) is not None or
                            re.search(r"\d{1,2}h\d{2}", current_clean.lower()) is not None
                        )
                        time_mentions = _extract_all_time_texts(user_text)
                        if suspicious_time or len(time_mentions) == 1:
                            args[key] = inferred_clean
                        continue
                    # Override suspicious generated values with grounded values from user text.
                    suspicious = (
                        not current_val or
                        "@" in current_val or
                        "t" in current_val.lower() and re.search(r"\d{4}-\d{2}-\d{2}", current_val) is not None or
                        (key_l == "title" and inferred_clean.lower() in user_text.lower()) or
                        inferred_clean.lower() not in user_text.lower()
                    )
                    if suspicious:
                        args[key] = inferred_clean

        missing = [k for k in required if k not in args or args[k] in ("", None)]
        if missing:
            continue

        repaired.append({
            "name": name,
            "arguments": args,
        })

    unique = []
    seen = set()
    for call in repaired:
        key = (call["name"], json.dumps(call["arguments"], sort_keys=True, ensure_ascii=False))
        if key in seen:
            continue
        seen.add(key)
        unique.append(call)
    return unique


def _contains_refusal(text):
    lower = str(text).lower()
    markers = [
        "i apologize",
        "i cannot",
        "i can't",
        "unable to",
        "not able to",
        "my capabilities are limited",
    ]
    return any(marker in lower for marker in markers)


def _split_clauses(user_text):
    if not user_text:
        return []
    clauses = re.split(r"\s*(?:,|;|\band then\b|\bthen\b|\band\b)\s*", user_text, flags=re.IGNORECASE)
    cleaned = [_clean_span(c) for c in clauses if _clean_span(c)]
    return cleaned or [_clean_span(user_text)]


def _tool_keywords(tool):
    words = set()
    words.update(_split_words(tool.get("name", "")))
    words.update(_split_words(tool.get("description", "")))

    properties, _ = _tool_schema(tool)
    for key, schema in properties.items():
        words.update(_split_words(key))
        if isinstance(schema, dict):
            words.update(_split_words(schema.get("description", "")))

    name_l = str(tool.get("name", "")).lower()
    for action, hints in _ACTION_HINTS.items():
        if action in name_l:
            for hint in hints:
                words.update(_split_words(hint))

    return {w for w in words if w and w not in _STOPWORDS}


def _tool_relevance(tool, text):
    text_l = str(text).lower()
    if not text_l.strip():
        return 0
    score = 0
    for kw in _tool_keywords(tool):
        if re.search(rf"\b{re.escape(kw)}\b", text_l):
            score += 2
        elif kw in text_l:
            score += 1
    return score


def _choose_clause_tools(clause, tools, max_tools=6):
    ranked = sorted(((tool, _tool_relevance(tool, clause)) for tool in tools), key=lambda x: x[1], reverse=True)
    if not ranked:
        return tools
    positive = [tool for tool, score in ranked if score > 0]
    if not positive:
        return tools
    return positive[:max_tools]


def _run_local_candidate_with_model(model, messages, tools, system_prompt=None, tool_rag_top_k=0, max_tokens=256):
    cactus_tools = [{"type": "function", "function": t} for t in tools]
    if not system_prompt:
        system_prompt = "You are a helpful assistant that can use tools."

    kwargs = {
        "tools": cactus_tools,
        "force_tools": True,
        "max_tokens": max_tokens,
        "stop_sequences": ["<|im_end|>", "<end_of_turn>"],
        "tool_rag_top_k": tool_rag_top_k,
    }
    try:
        raw_str = cactus_complete(
            model,
            [{"role": "system", "content": system_prompt}] + messages,
            **kwargs,
        )
    except TypeError:
        kwargs.pop("tool_rag_top_k", None)
        raw_str = cactus_complete(
            model,
            [{"role": "system", "content": system_prompt}] + messages,
            **kwargs,
        )

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
            "response": raw_str,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
        "response": raw.get("response", ""),
    }


def _run_local_candidate(messages, tools, system_prompt=None, tool_rag_top_k=0, max_tokens=256):
    model = cactus_init(functiongemma_path)
    try:
        return _run_local_candidate_with_model(
            model,
            messages,
            tools,
            system_prompt=system_prompt,
            tool_rag_top_k=tool_rag_top_k,
            max_tokens=max_tokens,
        )
    finally:
        cactus_destroy(model)


def _merge_calls(primary_calls, extra_calls):
    merged = []
    seen = set()
    for call in (primary_calls or []) + (extra_calls or []):
        key = (call.get("name"), json.dumps(call.get("arguments", {}), sort_keys=True, ensure_ascii=False))
        if key in seen:
            continue
        seen.add(key)
        merged.append(call)
    return merged


def _estimate_intent_count(user_text, tools):
    if not user_text:
        return 1
    clauses = [c for c in _split_clauses(user_text) if len(c.split()) >= 2]
    count = len(clauses) if clauses else 1
    if tools:
        count = min(count, len(tools))
    return max(1, count)


def _score_local_candidate(calls, tools, user_text, confidence=0.0, response=""):
    if not tools:
        return 0.0

    tools_by_name = {t["name"]: t for t in tools}
    valid_calls = 0
    required_total = 0
    required_hit = 0

    for call in calls:
        name = call.get("name")
        if name not in tools_by_name:
            continue
        valid_calls += 1
        properties, required = _tool_schema(tools_by_name[name])
        required_total += len(required)
        args = call.get("arguments", {})
        if not isinstance(args, dict):
            args = {}
        for req in required:
            if req in args and args[req] not in ("", None):
                required_hit += 1

    valid_ratio = (valid_calls / len(calls)) if calls else 0.0
    required_ratio = (required_hit / required_total) if required_total else (1.0 if calls else 0.0)
    expected = _estimate_intent_count(user_text, tools)
    coverage = min(len(calls), expected) / max(1, expected)
    overflow_penalty = max(0, len(calls) - expected) * 0.15
    refusal_penalty = 0.20 if _contains_refusal(response) and not calls else 0.0
    conf = max(0.0, min(1.0, float(confidence or 0.0)))

    score = (
        0.35 * valid_ratio +
        0.25 * required_ratio +
        0.25 * coverage +
        0.15 * conf -
        overflow_penalty -
        refusal_penalty
    )
    return max(0.0, min(1.0, score))


def _likely_tool_names(user_text, tools):
    if not tools:
        return set()
    scored = [(tool["name"], _tool_relevance(tool, user_text)) for tool in tools]
    top = max(score for _, score in scored)
    threshold = max(2, top - 2)
    return {name for name, score in scored if score >= threshold and score > 0}


def _likely_tools_with_clauses(user_text, tools):
    likely = set(_likely_tool_names(user_text, tools))
    for clause in _split_clauses(user_text):
        scored = sorted(((tool["name"], _tool_relevance(tool, clause)) for tool in tools), key=lambda x: x[1], reverse=True)
        if not scored or scored[0][1] <= 0:
            continue
        top_name, top_score = scored[0]
        likely.add(top_name)
        # Keep second-best when it is close to top relevance.
        if len(scored) > 1 and scored[1][1] >= max(1, top_score - 1):
            likely.add(scored[1][0])
    return likely


def _select_relevant_tools(user_text, tools, likely_tools=None, expected_calls=1):
    if not tools:
        return []
    if len(tools) <= 3:
        return tools

    likely_set = set(likely_tools or [])
    ranked = sorted(((tool, _tool_relevance(tool, user_text)) for tool in tools), key=lambda x: x[1], reverse=True)
    top_score = ranked[0][1] if ranked else 0

    keep_names = set()
    for tool, score in ranked:
        name = tool.get("name")
        if name in likely_set:
            keep_names.add(name)
            continue
        if top_score > 0 and score >= max(1, top_score - 2):
            keep_names.add(name)

    min_keep = min(len(tools), max(3, expected_calls + 1))
    max_keep = min(len(tools), max(6, expected_calls + 3))

    for tool, _ in ranked:
        if len(keep_names) >= min_keep:
            break
        keep_names.add(tool.get("name"))

    selected = [tool for tool in tools if tool.get("name") in keep_names]

    if len(selected) > max_keep:
        top_names = set()
        for tool, _ in ranked[:max_keep]:
            top_names.add(tool.get("name"))
        selected = [tool for tool in tools if tool.get("name") in top_names]

    return selected if selected else tools


def _required_arg_ratio(tool, clause_text, user_text):
    properties, required = _tool_schema(tool)
    if not required:
        return 1.0

    hits = 0
    tool_name = tool.get("name", "")
    for key in required:
        arg_type = properties.get(key, {}).get("type", "string")
        value = _infer_argument(key, arg_type, clause_text, tool_name)
        if value is None:
            value = _infer_argument(key, arg_type, user_text, tool_name)
        if value is None:
            continue
        coerced = _coerce_argument(value, arg_type, key, user_text)
        if coerced is not None and coerced != "":
            hits += 1
    return hits / len(required)


def _graph_route_calls(user_text, tools, expected_calls):
    if not user_text or not tools:
        return [], {"ambiguity": 0.0, "avg_margin": 0.0, "low_margin_ratio": 0.0}

    clauses = _split_clauses(user_text)
    if not clauses:
        clauses = [user_text]

    # Keep graph pass lightweight even on long prompts.
    clauses = clauses[: max(2, min(6, expected_calls + 2))]
    tools_by_name = {t["name"]: t for t in tools}

    used_counts = {}
    assignments = []
    margins = []

    for clause in clauses:
        scored = []
        for tool in tools:
            clause_rel = _tool_relevance(tool, clause)
            global_rel = _tool_relevance(tool, user_text)
            req_ratio = _required_arg_ratio(tool, clause, user_text)
            reuse_penalty = 0.9 * used_counts.get(tool.get("name"), 0)
            edge_score = (2.0 * clause_rel) + (0.35 * global_rel) + (2.8 * req_ratio) - reuse_penalty
            scored.append((edge_score, tool))

        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            continue

        best_score, best_tool = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        margin = best_score - second_score
        margins.append(margin)

        if best_score <= 0.6:
            continue

        assignments.append({
            "clause": clause,
            "tool_name": best_tool.get("name"),
            "score": best_score,
            "margin": margin,
        })
        used_counts[best_tool.get("name")] = used_counts.get(best_tool.get("name"), 0) + 1

    candidate_calls = []
    for item in assignments:
        tool_name = item.get("tool_name")
        tool = tools_by_name.get(tool_name)
        if not tool:
            continue
        clause = item.get("clause", "")

        clause_calls = _sanitize_calls(
            _synthesize_calls_from_text(clause, [tool]),
            [tool],
            clause,
        )
        if clause_calls:
            candidate_calls.append(clause_calls[0])
            continue

        # Fallback to global prompt for pronoun-heavy clauses.
        global_calls = _sanitize_calls(
            _synthesize_calls_from_text(user_text, [tool]),
            [tool],
            user_text,
        )
        if global_calls:
            candidate_calls.append(global_calls[0])

    candidate_calls = _sanitize_calls(candidate_calls, tools, user_text)

    avg_margin = (sum(margins) / len(margins)) if margins else 0.0
    low_margin_ratio = (
        sum(1 for m in margins if m < 1.2) / len(margins)
        if margins else 0.0
    )

    ambiguity = 0.0
    if low_margin_ratio >= 0.60:
        ambiguity += 0.35
    elif low_margin_ratio >= 0.35:
        ambiguity += 0.18
    if avg_margin <= 0.8:
        ambiguity += 0.25
    elif avg_margin <= 1.5:
        ambiguity += 0.12
    if len(candidate_calls) < min(expected_calls, len(tools)):
        ambiguity += 0.15
    ambiguity = max(0.0, min(1.0, ambiguity))

    return candidate_calls, {
        "ambiguity": ambiguity,
        "avg_margin": avg_margin,
        "low_margin_ratio": low_margin_ratio,
    }


def _sorted_tool_scores(text, tools):
    return sorted(
        ((tool["name"], _tool_relevance(tool, text)) for tool in tools),
        key=lambda x: x[1],
        reverse=True,
    )


def _call_argument_blob(calls):
    parts = []
    for call in calls or []:
        args = call.get("arguments", {})
        if not isinstance(args, dict):
            continue
        for value in args.values():
            if isinstance(value, str):
                parts.append(value.lower())
    return " ".join(parts)


def _time_values_from_calls(calls):
    values = []
    for call in calls or []:
        args = call.get("arguments", {})
        if not isinstance(args, dict):
            continue
        for key, value in args.items():
            key_l = str(key).lower()
            if key_l in {"time", "when"} and value is not None:
                normalized = _extract_time_text(str(value)) or str(value).strip()
                values.append(str(normalized).lower())
            elif key_l in {"hour", "minute"} and value is not None:
                values.append(str(value).strip().lower())
    return values


def _ambiguity_score(user_text, tools, expected_calls, calls):
    if not user_text or not tools:
        return 0.0

    score = 0.0

    global_scores = _sorted_tool_scores(user_text, tools)
    if len(global_scores) >= 2:
        top_1 = global_scores[0][1]
        top_2 = global_scores[1][1]
        margin = top_1 - top_2
        if top_1 <= 0:
            score += 0.20
        elif margin <= 0:
            score += 0.35
        elif margin <= 1:
            score += 0.28
        elif margin <= 2:
            score += 0.15

    ambiguous_clauses = 0
    clauses = _split_clauses(user_text)
    for clause in clauses:
        clause_scores = _sorted_tool_scores(clause, tools)
        if len(clause_scores) < 2:
            continue
        top_1 = clause_scores[0][1]
        top_2 = clause_scores[1][1]
        if top_1 <= 0:
            continue
        if top_2 >= (top_1 - 1):
            ambiguous_clauses += 1
    score += min(0.30, 0.12 * ambiguous_clauses)

    pronouns = re.findall(r"\b(him|her|them|it|there|that|this)\b", user_text, flags=re.IGNORECASE)
    if pronouns:
        arg_blob = _call_argument_blob(calls)
        unresolved = (not arg_blob) or any(
            re.search(rf"\b{re.escape(pron.lower())}\b", arg_blob) is not None
            for pron in pronouns
        )
        referents = [r.lower() for r in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", user_text)]
        if referents and any(ref in arg_blob for ref in referents):
            unresolved = False
        if unresolved:
            score += 0.15

    user_times = [t.lower() for t in _extract_all_time_texts(user_text)]
    call_times = _time_values_from_calls(calls)
    if len(set(user_times)) >= 2 and call_times:
        matched = any(ct in ut or ut in ct for ct in call_times for ut in user_times)
        if not matched:
            score += 0.15

    if expected_calls >= 2:
        unique_tool_calls = {c.get("name") for c in calls if isinstance(c, dict)}
        if len(unique_tool_calls) < min(expected_calls, 2):
            score += 0.18

    if len(calls) < expected_calls:
        gap = expected_calls - len(calls)
        score += min(0.20, 0.10 * gap)

    return max(0.0, min(1.0, score))


def _call_quality(call, tools_by_name, user_text):
    name = call.get("name")
    if name not in tools_by_name:
        return -1e9

    tool = tools_by_name[name]
    properties, required = _tool_schema(tool)
    args = call.get("arguments", {})
    if not isinstance(args, dict):
        args = {}

    score = float(_tool_relevance(tool, user_text))
    if required:
        present = sum(1 for req in required if req in args and args[req] not in ("", None))
        score += 3.0 * (present / len(required))

    for key, value in args.items():
        value_s = _clean_span(value)
        value_l = value_s.lower()
        if value_s and value_l in user_text.lower():
            score += 1.5
        if key.lower() == "time":
            expected_time = _extract_time_text(user_text)
            if expected_time and value_s.lower() == expected_time.lower():
                score += 2.0
        if "@" in value_s or re.search(r"\d{4}-\d{2}-\d{2}", value_s):
            score -= 2.0

    return score


def _trim_calls(calls, tools, user_text, max_calls, preferred_tools=None):
    if max_calls <= 0 or len(calls) <= max_calls:
        return calls

    tools_by_name = {t["name"]: t for t in tools}
    preferred = set(preferred_tools or [])

    ranked = sorted(calls, key=lambda c: _call_quality(c, tools_by_name, user_text), reverse=True)

    best_per_tool = {}
    for call in ranked:
        name = call.get("name")
        if name not in best_per_tool:
            best_per_tool[name] = call

    selected = []
    selected_keys = set()

    # First, ensure preferred tools are represented when possible (deterministic order).
    preferred_ranked = []
    for tool_name in preferred:
        call = best_per_tool.get(tool_name)
        if call:
            preferred_ranked.append(call)
    preferred_ranked.sort(key=lambda c: _call_quality(c, tools_by_name, user_text), reverse=True)
    for call in preferred_ranked:
        if len(selected) >= max_calls:
            break
        key = (call.get("name"), json.dumps(call.get("arguments", {}), sort_keys=True, ensure_ascii=False))
        if key in selected_keys:
            continue
        selected.append(call)
        selected_keys.add(key)

    # Next, add best distinct tools to improve intent coverage.
    distinct_ranked = sorted(best_per_tool.values(), key=lambda c: _call_quality(c, tools_by_name, user_text), reverse=True)
    for call in distinct_ranked:
        if len(selected) >= max_calls:
            break
        key = (call.get("name"), json.dumps(call.get("arguments", {}), sort_keys=True, ensure_ascii=False))
        if key in selected_keys:
            continue
        selected.append(call)
        selected_keys.add(key)

    # Finally, if still short, allow duplicates by raw quality.
    for call in ranked:
        if len(selected) >= max_calls:
            break
        key = (call.get("name"), json.dumps(call.get("arguments", {}), sort_keys=True, ensure_ascii=False))
        if key in selected_keys:
            continue
        selected.append(call)
        selected_keys.add(key)

    return selected


def _synthesize_calls_from_text(user_text, tools):
    if not user_text or not tools:
        return []

    synthesized = []
    clauses = _split_clauses(user_text)
    last_person = None
    for clause in clauses:
        ranked = sorted(((tool, _tool_relevance(tool, clause)) for tool in tools), key=lambda x: x[1], reverse=True)
        if not ranked or ranked[0][1] <= 0:
            continue
        for tool, score in ranked:
            if score <= 0:
                break
            properties, required = _tool_schema(tool)
            args = {}
            for req in required:
                arg_type = properties.get(req, {}).get("type", "string")
                value = _infer_argument(req, arg_type, clause, tool.get("name", ""))
                if value is None:
                    value = _infer_argument(req, arg_type, user_text, tool.get("name", ""))
                if value is None and req.lower() in {"recipient", "contact", "query", "name", "person"} and last_person:
                    value = last_person
                if value is None:
                    args = None
                    break
                coerced = _coerce_argument(value, arg_type, req, user_text)
                if coerced is None or coerced == "":
                    args = None
                    break
                args[req] = coerced
            if args:
                for req, coerced in args.items():
                    if req.lower() in {"recipient", "contact", "query", "name", "person"}:
                        if isinstance(coerced, str) and coerced.lower() not in {"him", "her", "them"}:
                            last_person = coerced
                synthesized.append({"name": tool["name"], "arguments": args})
                break

    if not synthesized:
        ranked = sorted(((tool, _tool_relevance(tool, user_text)) for tool in tools), key=lambda x: x[1], reverse=True)
        if ranked and ranked[0][1] > 0:
            tool = ranked[0][0]
            properties, required = _tool_schema(tool)
            args = {}
            for req in required:
                arg_type = properties.get(req, {}).get("type", "string")
                value = _infer_argument(req, arg_type, user_text, tool.get("name", ""))
                if value is None and req.lower() in {"recipient", "contact", "query", "name", "person"} and last_person:
                    value = last_person
                if value is None:
                    args = None
                    break
                coerced = _coerce_argument(value, arg_type, req, user_text)
                if coerced is None or coerced == "":
                    args = None
                    break
                args[req] = coerced
            if args:
                synthesized.append({"name": tool["name"], "arguments": args})

    return synthesized


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """
    Hybrid routing with local-first repair/recovery, then selective cloud fallback.
    Keeps interface compatible with benchmark.py.
    """
    user_text = _extract_user_text(messages)
    expected_calls = _estimate_intent_count(user_text, tools)
    likely_tools = _likely_tools_with_clauses(user_text, tools)

    # Tunable routing knobs for hidden-eval iteration.
    recovery_ambiguity_threshold = 0.45
    synthesis_ambiguity_threshold = 0.60
    cloud_ambiguity_low = 0.35
    cloud_ambiguity_high = 0.60
    cloud_ambiguity_extreme = 0.75
    local_time_budget_ms = 700
    min_budget_for_extra_pass_ms = 120
    clause_pass_budget_estimate_ms = 130
    focused_max_tokens = 112
    clause_max_tokens = 72

    total_local_time = 0
    local_confidence = 0.0
    primary_response = ""
    best_calls = []
    best_score = 0.0
    graph_meta = {"ambiguity": 0.0, "avg_margin": 0.0, "low_margin_ratio": 0.0}

    model = cactus_init(functiongemma_path)
    try:
        primary_max_tokens = 176 if expected_calls <= 2 else 224
        primary = _run_local_candidate_with_model(
            model,
            messages,
            tools,
            tool_rag_top_k=0,
            max_tokens=primary_max_tokens,
        )
        total_local_time += primary.get("total_time_ms", 0) or 0
        local_confidence = float(primary.get("confidence", 0) or 0)
        primary_response = primary.get("response", "")

        repaired_primary_calls = _sanitize_calls(primary.get("function_calls", []), tools, user_text)
        best_calls = _trim_calls(repaired_primary_calls, tools, user_text, expected_calls, preferred_tools=likely_tools)
        best_score = _score_local_candidate(best_calls, tools, user_text, local_confidence, primary_response)
        current_ambiguity = _ambiguity_score(user_text, tools, expected_calls, best_calls)
        best_tool_names = {c.get("name") for c in best_calls}
        missing_likely = likely_tools - best_tool_names

        need_recovery_signal = (
            len(best_calls) == 0 or
            len(best_calls) < expected_calls or
            best_score < 0.72 or
            bool(missing_likely) or
            current_ambiguity >= recovery_ambiguity_threshold
        )

        if need_recovery_signal and tools:
            # Run deterministic synthesis before expensive extra local passes.
            quick_synth = _sanitize_calls(_synthesize_calls_from_text(user_text, tools), tools, user_text)
            quick_merged = _sanitize_calls(_merge_calls(best_calls, quick_synth), tools, user_text)
            quick_merged = _trim_calls(quick_merged, tools, user_text, expected_calls, preferred_tools=likely_tools)
            quick_score = _score_local_candidate(quick_merged, tools, user_text, local_confidence, "")
            if quick_score >= best_score:
                best_calls = quick_merged
                best_score = quick_score
                current_ambiguity = _ambiguity_score(user_text, tools, expected_calls, best_calls)
                best_tool_names = {c.get("name") for c in best_calls}
                missing_likely = likely_tools - best_tool_names
                need_recovery_signal = (
                    len(best_calls) == 0 or
                    len(best_calls) < expected_calls or
                    best_score < 0.72 or
                    bool(missing_likely) or
                    current_ambiguity >= recovery_ambiguity_threshold
                )

        if tools:
            graph_calls, graph_meta_candidate = _graph_route_calls(user_text, tools, expected_calls)
            graph_meta = graph_meta_candidate
            if graph_calls:
                graph_merged = _sanitize_calls(_merge_calls(best_calls, graph_calls), tools, user_text)
                graph_merged = _trim_calls(
                    graph_merged,
                    tools,
                    user_text,
                    expected_calls,
                    preferred_tools=likely_tools,
                )
                graph_score = _score_local_candidate(
                    graph_merged,
                    tools,
                    user_text,
                    local_confidence,
                    primary_response,
                )
                graph_tool_names = {c.get("name") for c in graph_merged}
                best_tool_names = {c.get("name") for c in best_calls}
                graph_coverage_gain = len(graph_tool_names) > len(best_tool_names)
                if graph_score > best_score or (graph_coverage_gain and graph_score >= best_score - 0.01):
                    best_calls = graph_merged
                    best_score = graph_score
                    current_ambiguity = max(
                        _ambiguity_score(user_text, tools, expected_calls, best_calls),
                        graph_meta.get("ambiguity", 0.0),
                    )
                    best_tool_names = graph_tool_names
                    missing_likely = likely_tools - best_tool_names
                    need_recovery_signal = (
                        len(best_calls) == 0 or
                        len(best_calls) < expected_calls or
                        best_score < 0.72 or
                        bool(missing_likely) or
                        current_ambiguity >= recovery_ambiguity_threshold
                    )

        can_afford_extra_pass = (total_local_time + min_budget_for_extra_pass_ms) <= local_time_budget_ms
        if need_recovery_signal and tools and can_afford_extra_pass:
            focused_tools = _select_relevant_tools(user_text, tools, likely_tools=likely_tools, expected_calls=expected_calls)
            if len(focused_tools) < len(tools):
                focused_result = _run_local_candidate_with_model(
                    model,
                    messages,
                    focused_tools,
                    system_prompt=(
                        "You are a precise tool-calling assistant. "
                        "Use only the provided tools and return all needed function calls."
                    ),
                    tool_rag_top_k=0,
                    max_tokens=focused_max_tokens,
                )
                total_local_time += focused_result.get("total_time_ms", 0) or 0
                focused_calls = _sanitize_calls(focused_result.get("function_calls", []), focused_tools, user_text)
                focused_merged = _sanitize_calls(_merge_calls(best_calls, focused_calls), tools, user_text)
                focused_merged = _trim_calls(
                    focused_merged,
                    tools,
                    user_text,
                    expected_calls,
                    preferred_tools=likely_tools,
                )
                focused_score = _score_local_candidate(
                    focused_merged,
                    tools,
                    user_text,
                    local_confidence,
                    focused_result.get("response", ""),
                )
                focused_tool_names = {c.get("name") for c in focused_merged}
                best_tool_names = {c.get("name") for c in best_calls}
                improved_coverage = len(focused_tool_names) > len(best_tool_names)
                if focused_score > best_score or (improved_coverage and focused_score >= best_score - 0.02):
                    best_calls = focused_merged
                    best_score = focused_score
                    current_ambiguity = _ambiguity_score(user_text, tools, expected_calls, best_calls)
                    best_tool_names = focused_tool_names
                    missing_likely = likely_tools - best_tool_names
                    need_recovery_signal = (
                        len(best_calls) == 0 or
                        len(best_calls) < expected_calls or
                        best_score < 0.72 or
                        bool(missing_likely) or
                        current_ambiguity >= recovery_ambiguity_threshold
                    )

        can_afford_extra_pass = (total_local_time + min_budget_for_extra_pass_ms) <= local_time_budget_ms
        if need_recovery_signal and tools and can_afford_extra_pass:
            clause_calls = []
            clause_limit = min(3, max(2, expected_calls))
            remaining_budget_ms = max(0, local_time_budget_ms - total_local_time)
            affordable_clause_passes = int(remaining_budget_ms // clause_pass_budget_estimate_ms)
            clause_limit = min(clause_limit, affordable_clause_passes)
            for clause in _split_clauses(user_text)[:clause_limit]:
                if (total_local_time + min_budget_for_extra_pass_ms) > local_time_budget_ms:
                    break
                clause_tools = _choose_clause_tools(clause, tools)
                clause_messages = [{"role": "user", "content": clause}]
                clause_result = _run_local_candidate_with_model(
                    model,
                    clause_messages,
                    clause_tools,
                    tool_rag_top_k=0,
                    max_tokens=clause_max_tokens,
                )
                total_local_time += clause_result.get("total_time_ms", 0) or 0
                clause_fixed = _sanitize_calls(clause_result.get("function_calls", []), clause_tools, clause)
                clause_synth = _sanitize_calls(_synthesize_calls_from_text(clause, clause_tools), clause_tools, clause)
                clause_candidates = _sanitize_calls(_merge_calls(clause_fixed, clause_synth), clause_tools, clause)
                if clause_candidates:
                    # Each clause should usually map to one primary tool call.
                    clause_calls.append(clause_candidates[0])
                    tentative_calls = _sanitize_calls(_merge_calls(best_calls, clause_calls), tools, user_text)
                    tentative_calls = _trim_calls(
                        tentative_calls,
                        tools,
                        user_text,
                        expected_calls,
                        preferred_tools=likely_tools,
                    )
                    tentative_tool_names = {c.get("name") for c in tentative_calls}
                    tentative_missing = likely_tools - tentative_tool_names
                    if len(tentative_calls) >= expected_calls and (
                        not tentative_missing or len(tentative_tool_names) >= expected_calls
                    ):
                        break

            merged_calls = _sanitize_calls(_merge_calls(best_calls, clause_calls), tools, user_text)
            merged_calls = _trim_calls(merged_calls, tools, user_text, expected_calls, preferred_tools=likely_tools)
            merged_score = _score_local_candidate(
                merged_calls,
                tools,
                user_text,
                local_confidence,
                primary_response,
            )
            if merged_score > best_score:
                best_calls = merged_calls
                best_score = merged_score
                current_ambiguity = _ambiguity_score(user_text, tools, expected_calls, best_calls)

        best_tool_names = {c.get("name") for c in best_calls}
        missing_likely = likely_tools - best_tool_names
        if (
            len(best_calls) < expected_calls or
            best_score < 0.55 or
            bool(missing_likely) or
            current_ambiguity >= synthesis_ambiguity_threshold
        ) and tools:
            synthesized = _sanitize_calls(_synthesize_calls_from_text(user_text, tools), tools, user_text)
            merged_synth = _sanitize_calls(_merge_calls(best_calls, synthesized), tools, user_text)
            merged_synth = _trim_calls(merged_synth, tools, user_text, expected_calls, preferred_tools=likely_tools)
            synth_score = _score_local_candidate(merged_synth, tools, user_text, local_confidence, "")
            if synth_score >= best_score:
                best_calls = merged_synth
                best_score = synth_score
                current_ambiguity = _ambiguity_score(user_text, tools, expected_calls, best_calls)
    finally:
        cactus_destroy(model)

    local = {
        "function_calls": best_calls,
        "total_time_ms": total_local_time,
        "confidence": local_confidence,
    }

    best_tool_names = {c.get("name") for c in best_calls}
    missing_likely = likely_tools - best_tool_names
    graph_ambiguity = graph_meta.get("ambiguity", 0.0)
    graph_avg_margin = graph_meta.get("avg_margin", 0.0)
    graph_low_margin_ratio = graph_meta.get("low_margin_ratio", 0.0)
    final_ambiguity = max(
        _ambiguity_score(user_text, tools, expected_calls, best_calls),
        graph_ambiguity,
    )
    risk = 0
    if len(best_calls) == 0:
        risk += 2
    if len(best_calls) < expected_calls:
        risk += 1
    if best_score < 0.45:
        risk += 2
    if best_score < 0.70 and expected_calls >= 2:
        risk += 1
    if len(tools) >= 8 and best_score < 0.80:
        risk += 1
    if bool(missing_likely):
        risk += 1
    if final_ambiguity >= cloud_ambiguity_low:
        risk += 1
    if final_ambiguity >= cloud_ambiguity_high:
        risk += 1
    if graph_low_margin_ratio >= 0.50:
        risk += 1
    if graph_avg_margin <= 0.8 and expected_calls >= 2:
        risk += 1
    if local_confidence < min(0.85, confidence_threshold):
        risk += 1
    if _contains_refusal(primary_response) and not best_calls:
        risk += 1

    cloud_pressure = risk
    if final_ambiguity >= cloud_ambiguity_extreme and best_score < 0.90:
        cloud_pressure += 1
    if expected_calls >= 3 and best_score < 0.85:
        cloud_pressure += 1
    if graph_low_margin_ratio >= 0.67:
        cloud_pressure += 1

    local_bias = 0
    if best_score >= 0.92 and len(best_calls) >= expected_calls:
        local_bias += 2
    elif best_score >= 0.82:
        local_bias += 1
    if local_confidence >= 0.95:
        local_bias += 1
    if total_local_time >= local_time_budget_ms:
        local_bias += 2
    elif total_local_time >= 500:
        # Cloud fallback here is usually net-latency-negative.
        local_bias += 1
    if len(tools) <= 2 and expected_calls == 1 and best_score >= 0.70:
        local_bias += 1
    if graph_avg_margin >= 1.8 and not bool(missing_likely):
        local_bias += 1
    if graph_avg_margin >= 2.6 and best_score >= 0.85:
        local_bias += 1

    route_score = cloud_pressure - local_bias
    cloud_threshold = 3
    if expected_calls >= 3 or len(tools) >= 8:
        cloud_threshold = 2
    if final_ambiguity >= max(cloud_ambiguity_high, 0.70):
        cloud_threshold -= 1
    if graph_avg_margin >= 2.4 and expected_calls <= 2:
        cloud_threshold += 1
    if graph_low_margin_ratio >= 0.75:
        cloud_threshold -= 1
    cloud_threshold = max(1, cloud_threshold)

    should_try_cloud = bool(os.environ.get("GEMINI_API_KEY")) and route_score >= cloud_threshold
    if (
        should_try_cloud and
        total_local_time >= local_time_budget_ms and
        best_score >= 0.78 and
        len(best_calls) >= max(1, expected_calls - 1)
    ):
        # Avoid stacking cloud latency on top of already-expensive local recovery
        # when local quality is likely acceptable.
        should_try_cloud = False
    if should_try_cloud:
        try:
            cloud = generate_cloud(messages, tools)
        except Exception as exc:
            # Keep benchmark execution alive in offline/local-only setups.
            local["source"] = "on-device"
            local["cloud_error"] = str(exc)
            return local

        cloud_calls = _sanitize_calls(cloud.get("function_calls", []), tools, user_text)
        cloud_score = _score_local_candidate(cloud_calls, tools, user_text, 1.0, "")
        if cloud_score >= best_score + 0.02:
            cloud["function_calls"] = cloud_calls
            cloud["source"] = "cloud (fallback)"
            cloud["local_confidence"] = local_confidence
            cloud["total_time_ms"] += total_local_time
            return cloud

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
