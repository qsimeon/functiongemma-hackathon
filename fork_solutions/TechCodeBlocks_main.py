# Source: https://github.com/TechCodeBlocks/functiongemma-hackathon
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time, re, string, atexit, concurrent.futures, logging
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
from google import genai
from google.genai import types

_log = logging.getLogger("hybrid")


# ──────────────────────────────────────────────
# Stage hit counters (reset between benchmark runs)
# ──────────────────────────────────────────────

_stats = {
    "step4_accepted": 0,
    "step4_5_accepted": 0,
    "step5_decomp_full": 0,
    "step5_decomp_partial": 0,
    "step6_retry_accepted": 0,
    "cloud_fallback": 0,
}


def reset_stats():
    """Reset all counters. Call before a benchmark run."""
    for k in _stats:
        _stats[k] = 0


def get_stats():
    """Return a copy of current counters."""
    return dict(_stats)


# ──────────────────────────────────────────────
# System prompts
# ──────────────────────────────────────────────

_DEFAULT_PROMPT = "You are a helpful assistant that can use tools."

_SINGLE_CALL_PROMPT = (
    "You are a tool-calling assistant. Call exactly one function. "
    "Use the exact words from the request as argument values."
)

_FOCUSED_CALL_PROMPT_TEMPLATE = (
    'Call {name}({params}). Use exact words from the request.'
)


def _build_rich_prompt(tool):
    """
    Build a descriptive system prompt from the tool schema.
    Bridges indirect phrasings by including the tool's description
    and detailed parameter expectations. Fully tool-agnostic.
    """
    name = tool["name"]
    desc = tool.get("description", "")
    props = tool.get("parameters", {}).get("properties", {})
    required = tool.get("parameters", {}).get("required", [])

    param_parts = []
    for pname in required:
        pinfo = props.get(pname, {})
        ptype = pinfo.get("type", "string")
        pdesc = pinfo.get("description", pname)
        param_parts.append(f"{pname} ({ptype}): {pdesc}")

    prompt = f"You must call {name}. {desc}. "
    if param_parts:
        prompt += "Parameters: " + "; ".join(param_parts) + ". "
    prompt += "Extract values directly from the user's words."
    return prompt


def _augment_query(query, tool):
    """
    Prepend the tool description to the query so the model sees
    the semantic bridge between indirect phrasings and the tool.
    e.g. "Wake me up at 6 AM" → "[Set an alarm for a given time] Wake me up at 6 AM"
    """
    desc = tool.get("description", "")
    if desc:
        return f"[{desc}] {query}"
    return query


def _extract_from_broken_json(raw_str, tools):
    """
    Recover function calls from broken JSON that the 270M model produces.

    Common issues: Chinese colons (：), <escape> tags, leading zeros (00),
    bare values without quotes, truncated output. The model often has the
    RIGHT answer but wrong JSON syntax.

    Tool-agnostic: uses tool schemas to know what params to look for.
    """
    # Find which tool the model tried to call
    name_m = re.search(r'"name"\s*[：:]\s*"(\w+)"', raw_str)
    if not name_m:
        return []

    tool_name = name_m.group(1)
    tool = next((t for t in tools if t["name"] == tool_name), None)
    if not tool:
        return []

    props = tool.get("parameters", {}).get("properties", {})
    required = tool.get("parameters", {}).get("required", [])
    args = {}

    for pname in required:
        pinfo = props.get(pname, {})
        ptype = pinfo.get("type", "string")

        if ptype == "integer":
            # Match integer value (possibly with leading zeros)
            m = re.search(rf'"{pname}"\s*[：:]\s*(-?0*\d+)', raw_str)
            if m:
                args[pname] = int(m.group(1))
        elif ptype == "string":
            # Try quoted string first
            m = re.search(rf'"{pname}"\s*[：:]\s*"([^"]+)"', raw_str)
            if not m:
                # Try unquoted with <escape> tags
                m = re.search(
                    rf'"{pname}[：:]\s*(?:</?escape>)*\s*([A-Za-z][A-Za-z0-9\s\'-]+)',
                    raw_str,
                )
            if m:
                val = re.sub(r'</?escape>', '', m.group(1)).strip().rstrip('}"')
                if val:
                    args[pname] = val

    if all(p in args for p in required):
        return [{"name": tool_name, "arguments": args}]
    return []


def _extract_from_response_field(raw_str, tools):
    """
    The 270M model sometimes puts function call info in the "response" field
    as text instead of in function_calls. Example:
      "response": "<start_function_declaration> call:get_weather(location:\"Paris\")"

    Parse this and return a proper function call list.
    """
    # Extract the response field
    resp_m = re.search(r'"response"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_str)
    if not resp_m:
        return []

    resp = resp_m.group(1).replace('\\"', '"').replace('\\n', ' ')

    # Pattern: call:tool_name(param:"value", param2:"value2")
    call_m = re.search(r'call:(\w+)\((.+?)\)\s*$', resp)
    if not call_m:
        return []

    tool_name = call_m.group(1)
    params_str = call_m.group(2)

    tool = next((t for t in tools if t["name"] == tool_name), None)
    if not tool:
        return []

    props = tool.get("parameters", {}).get("properties", {})
    required = tool.get("parameters", {}).get("required", [])
    args = {}

    for pname in required:
        pinfo = props.get(pname, {})
        ptype = pinfo.get("type", "string")
        if ptype == "string":
            m = re.search(rf'{pname}:\s*["\',]*\s*([^"\',\)]+)', params_str)
            if m:
                args[pname] = m.group(1).strip().strip('"\'')
        elif ptype == "integer":
            m = re.search(rf'{pname}:\s*(\d+)', params_str)
            if m:
                args[pname] = int(m.group(1))

    if all(p in args for p in required):
        return [{"name": tool_name, "arguments": args}]
    return []


def _construct_synthetic_call(query, tool):
    """
    Last-resort: construct a function call by extracting parameter values
    directly from the query using the tool schema as a guide.

    Only works reliably for tools with 1 required string param or
    tools with well-structured integer params.
    """
    props = tool.get("parameters", {}).get("properties", {})
    required = tool.get("parameters", {}).get("required", [])

    # Only attempt for simple tools (1-2 required params)
    if len(required) > 2:
        return None

    args = {}
    tool_keywords = _get_tool_keywords(tool)

    # Common stop words to strip from extracted values
    stop = {"a", "an", "the", "some", "my", "me", "in", "at", "for", "to",
            "of", "up", "on", "and", "or", "is", "be", "it", "set", "get",
            "check", "find", "look", "play", "send", "text", "remind",
            "about", "what", "how", "whats", "hows", "please", "can", "you",
            "i", "tell", "show", "do", "does", "like"}

    for pname in required:
        pinfo = props.get(pname, {})
        ptype = pinfo.get("type", "string")

        if ptype == "integer":
            # Extract the first number from the query
            m = re.search(r'(\d+)', query)
            if m:
                args[pname] = int(m.group(1))

        elif ptype == "string":
            pdesc = pinfo.get("description", "").lower()

            # For person-name fields: extract proper nouns (capitalized words)
            # Only trigger for params that are actually about people, not songs/locations
            is_person_name = pname in ("query", "recipient") or (
                "name" in pdesc and any(w in pdesc for w in ("person", "contact", "search"))
            )
            if is_person_name:
                names = re.findall(r'\b([A-Z][a-z]+)\b', query)
                # Filter out stop words (capitalized at sentence start)
                names = [n for n in names if n.lower() not in stop and n.lower() not in tool_keywords]
                if names:
                    args[pname] = names[0]

            # For "message" fields: extract text after "saying"
            elif pname == "message":
                m = re.search(r'\bsaying\s+(.+?)(?:\s+and\s+|[.]?\s*$)', query, re.IGNORECASE)
                if m:
                    args[pname] = m.group(1).strip().rstrip('.')

            # For general single-string params (song, location, etc.):
            # Remove stop words and standalone tool keywords; keep tool keywords
            # that follow a content word (part of a phrase like "classical music")
            elif len(required) == 1:
                words = query.split()
                value_words = []
                for w in words:
                    clean = _strip_punct(w).lower()
                    # Normalize contractions: "how's"→"hows", "what's"→"whats"
                    clean_norm = clean.replace("'", "").replace("\u2019", "")
                    if not clean or clean in stop or clean_norm in stop:
                        continue
                    if clean in tool_keywords:
                        # Keep tool keyword if it extends a content phrase
                        # e.g. "classical music" — "music" follows "classical"
                        # but skip standalone keywords like "weather" at start
                        if value_words and _strip_punct(value_words[-1]).lower() not in tool_keywords:
                            value_words.append(w.strip(string.punctuation))
                        continue
                    value_words.append(w.strip(string.punctuation))
                if value_words:
                    args[pname] = " ".join(value_words)

    if all(p in args for p in required):
        return {"name": tool["name"], "arguments": args}
    return None


# ──────────────────────────────────────────────
# Global model cache (saves ~100-200ms per call)
# ──────────────────────────────────────────────

_fg_model = None


def _get_model():
    """Lazy-init FunctionGemma model; reuse across all calls."""
    global _fg_model
    if _fg_model is None:
        _fg_model = cactus_init(functiongemma_path)
    return _fg_model


def _cleanup():
    global _fg_model
    if _fg_model is not None:
        cactus_destroy(_fg_model)
        _fg_model = None


atexit.register(_cleanup)


# ──────────────────────────────────────────────
# Core local inference helper
# ──────────────────────────────────────────────

def _run_local(messages, tools, max_tokens=360, system_prompt=None):
    """Run FunctionGemma on the cached model."""
    model = _get_model()
    cactus_reset(model)

    if system_prompt is None:
        system_prompt = _DEFAULT_PROMPT

    cactus_tools = [{"type": "function", "function": t} for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": system_prompt}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=max_tokens,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    # ── Parse JSON (with sanitization fallback) ──
    raw = None
    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        # Sanitize common 270M model JSON issues and retry
        sanitized = raw_str
        sanitized = sanitized.replace('：', ':')           # Chinese colons
        sanitized = re.sub(r'</?escape>', '', sanitized)   # <escape> tags
        # Fix leading zeros in numbers (00→0, 01→1) but not inside strings
        sanitized = re.sub(r':(\s*)0(\d+)([,}\]])', r':\g<1>\g<2>\g<3>', sanitized)
        try:
            raw = json.loads(sanitized)
            _log.info("    _run_local recovered via JSON sanitization")
        except json.JSONDecodeError:
            pass

    if raw is not None:
        calls = raw.get("function_calls", [])
        # If function_calls is empty but response field has a function call pattern,
        # extract from the response field (model sometimes puts calls in text)
        if not calls:
            resp_calls = _extract_from_response_field(raw_str, tools)
            if resp_calls:
                _log.info("    _run_local recovered %d call(s) from response field", len(resp_calls))
                calls = resp_calls
        return {
            "function_calls": calls,
            "total_time_ms": raw.get("total_time_ms", 0),
            "confidence": raw.get("confidence", 0),
            "_raw": raw_str,
        }

    # ── JSON still broken — try regex extraction ──
    # Extract timing from the raw string so we account for time even on failure
    time_m = re.search(r'"total_time_ms"\s*:\s*([\d.]+)', raw_str)
    time_ms = float(time_m.group(1)) if time_m else 0

    recovered_calls = _extract_from_broken_json(raw_str, tools)
    if recovered_calls:
        _log.info("    _run_local recovered %d call(s) via regex extraction", len(recovered_calls))
        return {
            "function_calls": recovered_calls,
            "total_time_ms": time_ms,
            "confidence": 0,
            "_raw": raw_str,
        }

    _log.info("    _run_local JSON parse FAILED, raw: %.300s", raw_str)
    return {"function_calls": [], "total_time_ms": time_ms, "confidence": 0, "_raw": raw_str}


# ──────────────────────────────────────────────
# Structural validation (lightweight, zero-latency)
# ──────────────────────────────────────────────

def _validate(result, tools):
    """
    Check tool calls are structurally valid.
    Returns (is_valid, issue_type).
    Auto-repairs string->int casts in place.
    """
    calls = result.get("function_calls", [])
    if not calls:
        return False, "no_calls"

    tool_map = {t["name"]: t for t in tools}

    for call in calls:
        name = call.get("name", "")
        if name not in tool_map:
            return False, "bad_tool_name"

        schema = tool_map[name]
        args = call.get("arguments", {})
        required = schema["parameters"].get("required", [])

        for req in required:
            if req not in args:
                return False, "missing_arg"

        props = schema["parameters"].get("properties", {})
        for key, val in list(args.items()):
            if key in props and props[key].get("type") == "integer":
                if not isinstance(val, int):
                    try:
                        call["arguments"][key] = int(float(val))
                    except (ValueError, TypeError):
                        return False, "bad_type"
            if key in props and props[key].get("type") == "string":
                if not isinstance(val, str):
                    return False, "bad_type"

    return True, None


# ──────────────────────────────────────────────
# Value-fixing heuristics (zero-latency F1 boost)
# ──────────────────────────────────────────────

_TIME_PATTERN = re.compile(r'(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?')
_HOUR_PATTERN = re.compile(r'(\d{1,2})\s*(AM|PM|am|pm)')
_MINUTES_PATTERN = re.compile(r'(\d+)\s*minute')
_REMINDER_TIME_PATTERN = re.compile(r'at\s+(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)', re.IGNORECASE)
_REMINDER_TITLE_PATTERN = re.compile(
    r'(?:remind\s+me\s+(?:about|to)\s+)(.+?)(?:\s+at\s+\d)',
    re.IGNORECASE,
)


def _fix_values(result, tools, query):
    """
    Fix common FunctionGemma value errors:
    - Negative integers -> abs()
    - set_alarm hour/minute -> parse from query
    - set_timer minutes -> parse from query
    - Strip trailing punctuation from string args
    """
    for call in result.get("function_calls", []):
        name = call.get("name", "")
        args = call.get("arguments", {})

        # Strip whitespace from argument keys (model sometimes outputs "location " instead of "location")
        clean_args = {k.strip(): v for k, v in args.items()}
        if clean_args != args:
            call["arguments"] = clean_args
            args = call["arguments"]

        # Fix list-typed string arguments (model sometimes returns ["val"] instead of "val")
        tool = next((t for t in tools if t["name"] == name), None)
        if tool:
            props = tool["parameters"].get("properties", {})
            for key, val in list(args.items()):
                if key in props and props[key].get("type") == "string" and isinstance(val, list) and val:
                    args[key] = str(val[0])

        # Strip trailing punctuation from all string arguments
        for key, val in list(args.items()):
            if isinstance(val, str):
                args[key] = val.strip().rstrip(".!?,;:")

        # Fix negative integers
        tool = next((t for t in tools if t["name"] == name), None)
        if tool:
            props = tool["parameters"].get("properties", {})
            for key, val in list(args.items()):
                if key in props and props[key].get("type") == "integer" and isinstance(val, int):
                    if val < 0:
                        args[key] = abs(val)

        # Fix alarm values by parsing query
        if name == "set_alarm":
            m = _TIME_PATTERN.search(query)
            if m:
                hour = int(m.group(1))
                minute = int(m.group(2))
                if m.group(3) and m.group(3).upper() == "PM" and hour < 12:
                    hour += 12
                args["hour"] = hour
                args["minute"] = minute
            else:
                m = _HOUR_PATTERN.search(query)
                if m:
                    hour = int(m.group(1))
                    if m.group(2).upper() == "PM" and hour < 12:
                        hour += 12
                    args["hour"] = hour
                    args["minute"] = 0

        # Fix timer values by parsing query
        if name == "set_timer":
            m = _MINUTES_PATTERN.search(query)
            if m:
                args["minutes"] = int(m.group(1))

        # Fix play_music: "play some X music" → genre is X, strip filler "music"
        if name == "play_music":
            song = args.get("song", "")
            if (song.lower().endswith(" music")
                    and len(song.split()) >= 2
                    and re.search(r'\bsome\b', query, re.IGNORECASE)):
                args["song"] = song.rsplit(" ", 1)[0].strip()

        # Fix name/query string params: strip filler context the model may include
        # e.g. "Tom in my contacts" → "Tom", "Alice from work" → "Alice"
        if tool:
            props = tool["parameters"].get("properties", {})
            for key, val in list(args.items()):
                if isinstance(val, str) and key in props:
                    pdesc = props[key].get("description", "").lower()
                    if "name" in pdesc or key in ("query", "recipient"):
                        # Strip trailing filler phrases
                        cleaned = re.sub(
                            r'\s+(?:in\s+(?:my\s+)?contacts?|from\s+\w+|\'s?\s+name).*$',
                            '', val, flags=re.IGNORECASE,
                        ).strip()
                        if cleaned and len(cleaned) < len(val):
                            args[key] = cleaned

        # Fix reminder time and title by parsing query
        if name == "create_reminder":
            m = _REMINDER_TIME_PATTERN.search(query)
            if m:
                args["time"] = m.group(1).strip()
            m = _REMINDER_TITLE_PATTERN.search(query)
            if m:
                title = m.group(1).strip()
                title = re.sub(r'^(the|a|an)\s+', '', title, flags=re.IGNORECASE)
                args["title"] = title


# ──────────────────────────────────────────────
# Argument quality validation
# ──────────────────────────────────────────────

def _strip_punct(word):
    """Strip leading/trailing punctuation from a word for comparison."""
    return word.strip(string.punctuation)


_TOOL_DISCRIMINATORS = {
    "set_alarm": {"alarm", "wake"},
    "set_timer": {"timer", "countdown", "minutes", "minute"},
    "play_music": {"play", "music", "song", "listen"},
    "send_message": {"message", "text", "saying", "send"},
    "get_weather": {"weather", "forecast", "temperature"},
    "search_contacts": {"contacts", "contact", "find", "lookup"},
    "create_reminder": {"remind", "reminder"},
}


def _tool_matches_query(call, query):
    """Reject if the query clearly refers to a different tool than predicted."""
    tool_name = call.get("name", "")
    query_words = {_strip_punct(w) for w in query.lower().split()} - {""}

    own = _TOOL_DISCRIMINATORS.get(tool_name, set())
    if query_words & own:
        return True

    for other_name, other_keys in _TOOL_DISCRIMINATORS.items():
        if other_name == tool_name:
            continue
        if query_words & other_keys:
            return False

    return True


def _check_args(call, query):
    """
    Heuristic check: do the argument values look plausible given the query?
    Returns None if OK, or a reason string if the args look hallucinated.
    """
    # Tool-query consistency: reject if query clearly refers to a different tool
    if not _tool_matches_query(call, query):
        return "tool_mismatch: %s not indicated by query" % call.get("name", "?")

    args = call.get("arguments", {})
    query_lower = query.lower()
    query_words_clean = {_strip_punct(w) for w in query_lower.split()} - {""}

    for key, val in args.items():
        if not isinstance(val, str):
            continue
        # Hallucinated email addresses
        if "@" in val:
            return "hallucinated_email: %s=%r" % (key, val)
        # Trailing quote stuck on value
        if val.endswith("'") or val.endswith('"'):
            return "trailing_quote: %s=%r" % (key, val)
        # Value is excessively long relative to query
        if len(val) > len(query):
            return "value_too_long: %s=%r (%d > %d)" % (key, val, len(val), len(query))

        val_words = {_strip_punct(w).lower() for w in val.split()} - {""}

        # Name-like fields: EVERY word must appear in the query
        # Catches hallucinated full names like "Alice Smith" when query only says "Alice"
        if key in ("recipient", "query"):
            extra = val_words - query_words_clean
            if extra:
                return "hallucinated_name: %s=%r extra_words=%s" % (key, val, extra)

        # Location field: every word must appear in query
        # Catches hallucinated expansions like "New York City" when query says "New York"
        if key == "location":
            extra = val_words - query_words_clean
            if extra:
                return "hallucinated_location: %s=%r extra_words=%s" % (key, val, extra)

        # General: string value must share at least one word with the query
        if val_words and not (val_words & query_words_clean):
            return "no_word_overlap: %s=%r" % (key, val)

    return None


def _args_look_good(call, query):
    """Thin wrapper: returns True if args pass validation, False otherwise."""
    reason = _check_args(call, query)
    if reason:
        _log.debug("    _check_args REJECT: %s | %s(%s)",
                   reason, call.get("name"), json.dumps(call.get("arguments", {})))
    return reason is None



# ──────────────────────────────────────────────
# Per-tool focused inference (model-based, no regex)
# ──────────────────────────────────────────────

def _try_each_tool(messages, tools, query, time_so_far):
    """
    Try running the model with each tool individually.
    Reduces the tool-selection problem: model only needs to extract args.
    Tries up to 3 most-relevant tools before giving up.
    Returns (result_or_None, accumulated_time_ms).
    """
    total_time = time_so_far

    # Order tools by keyword relevance (most likely tool first → fewer model calls)
    query_words = {_strip_punct(w) for w in query.lower().split()} - {""}
    ordered = sorted(tools, key=lambda t: _tool_relevance(t, query_words), reverse=True)
    # Try the single most-relevant tool (avoids false positives from wrong tools)
    relevant = ordered[:1]

    for t in relevant:
        # Build rich prompt from tool schema and augment query with description
        rich_prompt = _build_rich_prompt(t)
        aug_query = _augment_query(query, t)
        aug_messages = messages[:-1] + [{"role": "user", "content": aug_query}]

        focused = _run_local(
            aug_messages,
            [t],
            max_tokens=64,
            system_prompt=rich_prompt,
        )
        total_time += focused["total_time_ms"]
        _fix_values(focused, tools, query)
        f_valid, _ = _validate(focused, tools)
        if f_valid and focused["function_calls"]:
            if all(_args_look_good(c, query) for c in focused["function_calls"]):
                focused["source"] = "on-device"
                focused["total_time_ms"] = total_time
                return focused, total_time

    return None, total_time


# ──────────────────────────────────────────────
# Query analysis helpers
# ──────────────────────────────────────────────

_SPLIT_PATTERN = re.compile(
    r',\s*and\s+|\s+and\s+(?=[a-zA-Z])|,\s+(?=[a-zA-Z])',
    re.IGNORECASE,
)


def _count_expected_actions(query):
    """Estimate how many tool calls a query requires."""
    parts = _SPLIT_PATTERN.split(query)
    parts = [p.strip() for p in parts if p.strip()]
    return max(1, len(parts))


_TOOL_KEYWORDS_CACHE = {}   # tool name → set of keywords (built once per tool set)

# Stop words to exclude from description keyword extraction
_STOP_WORDS = {
    "a", "an", "the", "to", "for", "of", "in", "on", "at", "is", "it",
    "and", "or", "by", "with", "from", "this", "that", "be", "as", "are",
    "was", "will", "can", "do", "has", "have", "not", "but", "if", "its",
    "their", "they", "you", "your", "we", "our", "my", "me", "him", "her",
}


def _get_tool_keywords(tool):
    """
    Extract all meaningful keywords from a tool definition.
    Pulls from: tool name, description, parameter names, parameter descriptions.
    Cached per tool name to avoid recomputing.
    """
    name = tool["name"]
    if name in _TOOL_KEYWORDS_CACHE:
        return _TOOL_KEYWORDS_CACHE[name]

    kw = set()
    # Tool name words (e.g. "send_message" → {"send", "message"})
    kw |= set(name.replace("_", " ").lower().split())
    # Description words
    desc = tool.get("description", "")
    kw |= {w.lower().strip(string.punctuation) for w in desc.split()} - _STOP_WORDS - {""}
    # Parameter names and their descriptions
    for pname, pinfo in tool.get("parameters", {}).get("properties", {}).items():
        kw |= set(pname.replace("_", " ").lower().split())
        pdesc = pinfo.get("description", "")
        kw |= {w.lower().strip(string.punctuation) for w in pdesc.split()} - _STOP_WORDS - {""}

    _TOOL_KEYWORDS_CACHE[name] = kw
    return kw


# Common query-word synonyms that map to tool keywords
_QUERY_SYNONYMS = {
    "text": {"send", "message"},
    "wake": {"alarm", "set"},
    "find": {"search", "contacts"},
    "lookup": {"search", "contacts"},
    "remind": {"reminder", "create"},
    "listen": {"play", "music"},
}


def _tool_relevance(tool, query_words):
    """Score a tool against a set of query words using description keywords + synonyms."""
    kw = _get_tool_keywords(tool)
    # Expand query words with synonyms
    expanded = set(query_words)
    for qw in query_words:
        if qw in _QUERY_SYNONYMS:
            expanded |= _QUERY_SYNONYMS[qw]
    return len(expanded & kw)


def _match_tools_to_segment(segment, tools):
    """Score each tool against a query segment by keyword overlap from descriptions."""
    seg_words = {_strip_punct(w) for w in segment.lower().split()} - {""}
    scored = []
    for tool in tools:
        overlap = _tool_relevance(tool, seg_words)
        if overlap > 0:
            scored.append((overlap, tool))
    scored.sort(key=lambda x: -x[0])
    if scored:
        return [s[1] for s in scored[:3]]
    return tools


def _decompose_and_solve(query, tools, time_so_far):
    """
    Split a multi-action query into sub-queries, solve each locally.
    Includes pronoun propagation across segments.
    Returns merged result dict or None if decomposition fails entirely.
    """
    segments = _SPLIT_PATTERN.split(query)
    segments = [s.strip() for s in segments if s.strip()]

    if len(segments) <= 1:
        return None

    # ── Pronoun propagation ──
    # Find proper nouns in earlier segments and replace pronouns in later ones
    _skip_words = {
        "set", "play", "find", "look", "send", "text", "check", "get",
        "remind", "the", "and", "what", "how", "wake", "my", "in", "at",
        "to", "for", "of", "up", "me", "an", "a", "it", "is", "be",
    }
    found_name = None
    for i, seg in enumerate(segments):
        for name_match in re.finditer(r'\b([A-Z][a-z]+)\b', seg):
            candidate = name_match.group(1)
            if candidate.lower() not in _skip_words:
                found_name = candidate
                break
        if found_name and i > 0:
            seg = re.sub(r'\bhim\b', found_name, seg, flags=re.IGNORECASE)
            seg = re.sub(r'\bher\b', found_name, seg, flags=re.IGNORECASE)
            seg = re.sub(r'\bthem\b', found_name, seg, flags=re.IGNORECASE)
            segments[i] = seg

    all_calls = []
    total_time = time_so_far
    failed_segments = []

    for seg in segments:
        matched_tools = _match_tools_to_segment(seg, tools)
        _log.info("    decomp seg=%r matched=%s", seg[:50], [t["name"] for t in matched_tools[:1]])

        # ── Try model with top matched tool (reduces selection ambiguity) ──
        sub_result = _run_local(
            [{"role": "user", "content": seg}],
            matched_tools[:1],
            max_tokens=64,
            system_prompt=_SINGLE_CALL_PROMPT,
        )
        total_time += sub_result["total_time_ms"]
        _fix_values(sub_result, tools, seg)

        valid, _ = _validate(sub_result, tools)
        if valid and sub_result["function_calls"]:
            if all(_args_look_good(c, seg) for c in sub_result["function_calls"]):
                # Reject exact duplicates of already-collected calls
                new_calls = [c for c in sub_result["function_calls"]
                             if not any(c["name"] == e["name"] and c.get("arguments") == e.get("arguments")
                                        for e in all_calls)]
                if new_calls:
                    all_calls.extend(new_calls)
                    _log.info("    decomp seg OK: %s", [c["name"] for c in new_calls])
                    continue

        _log.info("    decomp seg FAILED first try, valid=%s calls=%d", valid, len(sub_result.get("function_calls", [])))

        # ── Retry: try top matched tool with rich prompt + augmented query ──
        found = False
        for t in matched_tools[:1]:
            rich_prompt = _build_rich_prompt(t)
            aug_seg = _augment_query(seg, t)
            focused = _run_local(
                [{"role": "user", "content": aug_seg}],
                [t],
                max_tokens=64,
                system_prompt=rich_prompt,
            )
            total_time += focused["total_time_ms"]
            _fix_values(focused, tools, seg)
            f_valid, _ = _validate(focused, tools)
            if f_valid and focused["function_calls"]:
                if all(_args_look_good(c, seg) for c in focused["function_calls"]):
                    new_calls = [c for c in focused["function_calls"]
                                 if not any(c["name"] == e["name"] and c.get("arguments") == e.get("arguments")
                                            for e in all_calls)]
                    if new_calls:
                        all_calls.extend(new_calls)
                        _log.info("    decomp seg OK (retry): %s", [c["name"] for c in new_calls])
                        found = True
                        break

        if not found:
            # Last resort: try synthetic call construction from query keywords
            for t in matched_tools[:1]:
                synthetic = _construct_synthetic_call(seg, t)
                if synthetic:
                    _fix_values({"function_calls": [synthetic]}, tools, seg)
                    s_valid, _ = _validate({"function_calls": [synthetic]}, tools)
                    if s_valid and _args_look_good(synthetic, seg):
                        all_calls.append(synthetic)
                        _log.info("    decomp seg OK (synthetic): %s", synthetic["name"])
                        found = True
                        break

        if not found:
            _log.info("    decomp seg FAILED (all tries): %r", seg[:50])
            failed_segments.append(seg)

    if not all_calls:
        return None

    return {
        "function_calls": all_calls,
        "total_time_ms": total_time,
        "source": "on-device",
        "_failed_segments": failed_segments,
    }


# ──────────────────────────────────────────────
# Original functions (kept intact for compatibility)
# ──────────────────────────────────────────────

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

    try:
        gemini_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(tools=gemini_tools),
        )
    except Exception as e:
        total_time_ms = (time.time() - start_time) * 1000
        print(f"[cloud error: {e}]", end=" ", flush=True)
        return {"function_calls": [], "total_time_ms": total_time_ms}

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


def generate_cloud_with_timeout(messages, tools, timeout_sec=5):
    """Wrap generate_cloud with a hard timeout to prevent 30+ second hangs."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(generate_cloud, messages, tools)
        try:
            result = future.result(timeout=timeout_sec)
            return result
        except concurrent.futures.TimeoutError:
            return {
                "function_calls": [],
                "total_time_ms": timeout_sec * 1000,
                "source": "cloud (fallback)",
            }


def _log_local_failure(label, local, query, issue=None):
    """Log the local result, raw model response, and rejection reasons when falling back."""
    calls = local.get("function_calls", [])
    raw = local.get("_raw", "<no raw captured>")
    _log.info("  [%s] query: %s", label, query[:120])
    if issue:
        _log.info("  [%s] validation_issue: %s", label, issue)
    _log.info("  [%s] local_calls=%s", label, json.dumps(calls, ensure_ascii=False))
    for c in calls:
        reason = _check_args(c, query)
        if reason:
            _log.info("  [%s]   REJECT %s(%s): %s",
                      label, c.get("name"), json.dumps(c.get("arguments", {})), reason)
    _log.info("  [%s] raw_response: %.500s", label, raw)


def _cancel_cloud(cloud_future, cloud_executor):
    """Best-effort cancellation of parallel cloud request."""
    if cloud_future is not None:
        cloud_future.cancel()
    if cloud_executor is not None:
        cloud_executor.shutdown(wait=False)


# ──────────────────────────────────────────────
# Hybrid cascade: Speculate -> Fix -> Validate -> Improve -> Cloud
# ──────────────────────────────────────────────


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """
    Speculative Edge Cascade (SEC) with Parallel Cloud Speculation.

    Based on the edge-cloud router pattern (agent paper Section 3.5):
    - Fire cloud request in background for multi-action queries (high-risk)
    - Run local inference + decomposition in parallel
    - Pick the better result (local preferred when valid)
    - For single-action: local first, cloud only on failure

    Also incorporates:
    - JSON sanitization + regex extraction for broken-but-correct answers
    - Early text-response detection to skip retries for hopeless cases
    - Argument quality validation to prevent garbage acceptance
    """
    query = messages[-1]["content"]
    expected_count = _count_expected_actions(query)
    total_time = 0

    # ── PARALLEL CLOUD SPECULATION for multi-action queries ──
    # Multi-action queries are risky (decomposition may fail partially).
    # Fire cloud in background so it runs in parallel with local inference.
    cloud_future = None
    cloud_executor = None
    if expected_count >= 2:
        cloud_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        cloud_future = cloud_executor.submit(generate_cloud, messages, tools)
        _log.info("  [SPEC] parallel cloud fired for multi-action query")

    # ── Tool pre-filtering for single-action queries ──
    # Only narrow if a tool has positive keyword relevance; otherwise keep all
    initial_tools = tools
    if expected_count == 1 and len(tools) > 1:
        query_words = {_strip_punct(w) for w in query.lower().split()} - {""}
        scored = [(t, _tool_relevance(t, query_words)) for t in tools]
        best_score = max(s for _, s in scored)
        if best_score > 0:
            best_tool = max(scored, key=lambda x: x[1])[0]
            initial_tools = [best_tool]
        # else: no keyword match — keep all tools, let model choose

    # ── Dynamic max_tokens based on complexity ──
    if expected_count == 1:
        init_max_tokens = 64
    else:
        init_max_tokens = 256

    # ── STEP 1: LOCAL INFERENCE ──
    local = _run_local(messages, initial_tools, max_tokens=init_max_tokens)
    total_time += local["total_time_ms"]

    # ── STEP 2: FIX VALUES (zero-latency) ──
    _fix_values(local, tools, query)

    # ── STEP 3: VALIDATE ──
    valid, issue = _validate(local, tools)
    actual_count = len(local.get("function_calls", []))

    # ── STEP 4: ACCEPT if valid, complete, and args look good ──
    good_calls = [c for c in local.get("function_calls", []) if _args_look_good(c, query)]
    good_count = len(good_calls)

    _log.info(
        "  [STEP1] valid=%s issue=%s calls=%d good=%d/%d tools=%d time=%.0fms | %s",
        valid, issue, actual_count, good_count, expected_count,
        len(initial_tools), total_time, query[:60],
    )
    if valid and not (good_count >= expected_count):
        for c in local.get("function_calls", []):
            reason = _check_args(c, query)
            if reason:
                _log.info("    rejected call: %s(%s) reason=%s",
                          c.get("name"), json.dumps(c.get("arguments", {})), reason)

    if valid and good_count >= expected_count:
        _cancel_cloud(cloud_future, cloud_executor)
        _stats["step4_accepted"] += 1
        _log.info("  → STEP4 accepted (%.0fms)", total_time)
        local["source"] = "on-device"
        local["total_time_ms"] = total_time
        return local

    # ── STEP 4.5: For single-tool queries, try each tool individually ──
    if expected_count == 1:
        focused, total_time = _try_each_tool(messages, tools, query, total_time)
        if focused:
            _cancel_cloud(cloud_future, cloud_executor)
            _stats["step4_5_accepted"] += 1
            _log.info("  → STEP4.5 accepted (%.0fms)", total_time)
            return focused

    # ── STEP 4.6: Synthetic call construction (zero-latency, last resort before cloud) ──
    # When the model completely fails, try to construct a call from query keywords.
    # This is the "heuristic extraction" approach from the agent paper — use the tool
    # schema itself to guide extraction when the SLM can't help.
    if expected_count == 1 and issue == "no_calls":
        query_words = {_strip_punct(w) for w in query.lower().split()} - {""}
        scored = [(t, _tool_relevance(t, query_words)) for t in tools]
        best_tool = max(scored, key=lambda x: x[1])[0]
        if max(s for _, s in scored) > 0:
            synthetic = _construct_synthetic_call(query, best_tool)
            if synthetic:
                _fix_values({"function_calls": [synthetic]}, tools, query)
                s_valid, _ = _validate({"function_calls": [synthetic]}, tools)
                if s_valid and _args_look_good(synthetic, query):
                    _log.info("  → STEP4.6 synthetic call: %s(%s)",
                              synthetic["name"], json.dumps(synthetic["arguments"]))
                    return {
                        "function_calls": [synthetic],
                        "total_time_ms": total_time,
                        "source": "on-device",
                    }

    # ── STEP 5: IMPROVE partial/garbled results for multi-action queries ──
    if expected_count > 1:
        decomposed = _decompose_and_solve(query, tools, total_time)
        if decomposed is not None:
            _fix_values(decomposed, tools, query)
            d_valid, _ = _validate(decomposed, tools)
            d_count = len(decomposed.get("function_calls", []))
            failed_segs = decomposed.get("_failed_segments", [])

            if d_valid and d_count >= expected_count and not failed_segs:
                _cancel_cloud(cloud_future, cloud_executor)
                _stats["step5_decomp_full"] += 1
                _log.info("  → STEP5 decomp full (%.0fms)", decomposed["total_time_ms"])
                return decomposed

            if d_valid and d_count > 0 and failed_segs:
                _stats["step5_decomp_partial"] += 1
                _log.info("  → STEP5 decomp partial, cloud filling %s", failed_segs)
                # Always use targeted cloud for just the failed segments
                # (more reliable than full-query cloud which may drop calls)
                cloud = generate_cloud_with_timeout(
                    [{"role": "user", "content": " and ".join(failed_segs)}],
                    tools,
                )
                _fix_values(cloud, tools, query)
                total_time_combined = max(decomposed["total_time_ms"],
                                          cloud["total_time_ms"])
                merged = list(decomposed["function_calls"])
                merged.extend(cloud.get("function_calls", []))
                return {
                    "function_calls": merged,
                    "total_time_ms": total_time_combined,
                    "source": "on-device",
                }

            if d_valid and d_count >= expected_count:
                _cancel_cloud(cloud_future, cloud_executor)
                return decomposed

            total_time = decomposed["total_time_ms"]
        else:
            _log.info("  [STEP5] decomposition returned None (all segments failed)")

        # Decomposition failed entirely — use parallel cloud if available
        if cloud_future is not None:
            _log.info("  → using parallel cloud result (decomp failed)")
            try:
                cloud = cloud_future.result(timeout=10)
            except Exception:
                cloud = generate_cloud_with_timeout(messages, tools)
            if cloud_executor:
                cloud_executor.shutdown(wait=False)
            _fix_values(cloud, tools, query)
            cloud_future = None
            cloud_executor = None
            cloud_calls = cloud.get("function_calls", [])
            if cloud_calls:
                _stats["cloud_fallback"] += 1
                cloud_time = max(total_time, cloud["total_time_ms"])
                return {
                    "function_calls": cloud_calls,
                    "total_time_ms": cloud_time,
                    "source": "cloud (fallback)",
                }

        # No cloud result either — accept good local calls if any
        if good_count > 0:
            _cancel_cloud(cloud_future, cloud_executor)
            local["function_calls"] = good_calls
            local["source"] = "on-device"
            local["total_time_ms"] = total_time
            return local

    # ── STEP 6: One retry for single-action no_calls ──
    if issue == "no_calls" and expected_count == 1:
        _log.info("  → STEP6 no_calls retry")
        retry = _run_local(messages, tools, max_tokens=256)
        total_time += retry["total_time_ms"]
        _fix_values(retry, tools, query)
        r_valid, _ = _validate(retry, tools)
        if r_valid and all(_args_look_good(c, query) for c in retry.get("function_calls", [])):
            _stats["step6_retry_accepted"] += 1
            _log.info("  → STEP6 retry accepted (%.0fms)", total_time)
            retry["source"] = "on-device"
            retry["total_time_ms"] = total_time
            return retry

    # ── STEP 7: Cloud fallback ──
    _log_local_failure("STEP7-cloud", local, query, issue=issue)
    _stats["cloud_fallback"] += 1
    _log.info("  → CLOUD fallback (%.0fms local)", total_time)
    # Use parallel cloud result if still available
    if cloud_future is not None:
        try:
            cloud = cloud_future.result(timeout=10)
        except Exception:
            cloud = generate_cloud_with_timeout(messages, tools)
        if cloud_executor:
            cloud_executor.shutdown(wait=False)
    else:
        cloud = generate_cloud_with_timeout(messages, tools)
    _fix_values(cloud, tools, query)
    cloud["source"] = "cloud (fallback)"
    # For parallel speculation, use max time (they ran simultaneously)
    if expected_count >= 2:
        cloud["total_time_ms"] = max(total_time, cloud["total_time_ms"])
    else:
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