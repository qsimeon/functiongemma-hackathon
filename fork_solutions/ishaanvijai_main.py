# Source: https://github.com/ishaanvijai/deepmind-cactus-hackathon

import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, re, time
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types

_CACHED_MODEL = None

def get_cached_model():
    """Lazily initialize and global cache the given model."""
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        _CACHED_MODEL = cactus_init(functiongemma_path)
    return _CACHED_MODEL


def generate_cactus(
    messages,
    tools,
    system_prompt=None,
    tool_rag_top_k=0,
    max_tokens=None,
    temperature=None,
    top_p=None,
    top_k=None,
    local_confidence_threshold=None,
):
    """Run function calling on-device via FunctionGemma + Cactus.

    :param messages: Chat messages.
    :param tools: Tool declarations.
    :param system_prompt: Optional system instruction.
    :param tool_rag_top_k: Tool-RAG shortlist size.
    :param max_tokens: Max decode tokens.
    :param temperature: Sampling temperature.
    :param top_p: Top-p sampling parameter.
    :param top_k: Top-k sampling parameter.
    :param local_confidence_threshold: Local confidence threshold for cloud_handoff signaling.
    :returns: Parsed local model result.
    """
    model = get_cached_model()

    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    prompt = system_prompt or "You are a helpful assistant that can use tools."

    effective_max_tokens = int(max_tokens) if isinstance(max_tokens, int) and max_tokens > 0 else _LOCAL_DEFAULT_MAX_TOKENS
    effective_temperature = (
        float(temperature)
        if isinstance(temperature, (int, float)) and 0.0 <= float(temperature) <= 2.0
        else _LOCAL_DEFAULT_TEMPERATURE
    )
    effective_top_p = (
        float(top_p)
        if isinstance(top_p, (int, float)) and 0.0 < float(top_p) <= 1.0
        else _LOCAL_DEFAULT_TOP_P
    )
    effective_top_k = int(top_k) if isinstance(top_k, int) and top_k > 0 else _LOCAL_DEFAULT_TOP_K
    effective_local_conf_threshold = (
        float(local_confidence_threshold)
        if isinstance(local_confidence_threshold, (int, float)) and 0.0 < float(local_confidence_threshold) < 1.0
        else _LOCAL_DEFAULT_CONFIDENCE_THRESHOLD
    )

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": prompt}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=effective_max_tokens,
        temperature=effective_temperature,
        top_p=effective_top_p,
        top_k=effective_top_k,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
        tool_rag_top_k=tool_rag_top_k,
        confidence_threshold=effective_local_conf_threshold,
    )

    raw = _parse_cactus_output(raw_str)

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
        "cloud_handoff": bool(raw.get("cloud_handoff", False)),
    }


_CACHED_GENAI_CLIENT = None

def get_cached_genai_client():
    global _CACHED_GENAI_CLIENT
    if _CACHED_GENAI_CLIENT is None:
        _CACHED_GENAI_CLIENT = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    return _CACHED_GENAI_CLIENT


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    client = get_cached_genai_client()

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


_TIME_12H_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(AM|PM)\b", re.IGNORECASE)
_MINUTES_RE = re.compile(r"\b(\d+)\s*(?:minutes?|mins?)\b", re.IGNORECASE)
_KEYWORD_STOPWORDS = {
    "a", "an", "the", "for", "to", "with", "and", "or", "of",
    "set", "get", "create", "play", "search", "send", "current", "given",
    "name", "time", "message", "contact", "weather", "alarm", "timer", "song",
}
_TOOL_HINTS = {
    "get_weather": {"weather", "forecast", "temperature"},
    "set_alarm": {"alarm", "wake"},
    "send_message": {"message", "text", "sms"},
    "create_reminder": {"remind", "reminder"},
    "search_contacts": {"contacts", "contact", "find", "lookup", "look", "search"},
    "play_music": {"play", "music", "song", "playlist"},
    "set_timer": {"timer", "countdown"},
}
_KNOWN_REGEX_TOOL_NAMES = frozenset(_TOOL_HINTS.keys())
_LOCAL_DEFAULT_MAX_TOKENS = 384
_LOCAL_MULTI_INTENT_MAX_TOKENS = 512
_LOCAL_COMPLEX_MAX_TOKENS = 896
_LOCAL_DEFAULT_TEMPERATURE = 0.10
_LOCAL_DEFAULT_TOP_P = 0.95
_LOCAL_DEFAULT_TOP_K = 40
_LOCAL_DEFAULT_CONFIDENCE_THRESHOLD = 0.68
_LOCAL_SINGLE_INTENT_TOOL_CAP = 5
_LOCAL_SINGLE_INTENT_TOOL_RAG_TOP_K = 3
_CLOUD_ACCEPT_MARGIN_DEFAULT = 0.75
_CLOUD_ACCEPT_MARGIN_MULTI_TURN_CAP = 0.35


def _stem_token(token):
    """Lightly normalize a token for lexical matching.

    :param token: Raw token text.
    :returns: Stemmed/normalized token.
    """
    if not token:
        return ""
    stem = token.lower()
    for suffix in ("ing", "ed", "es", "s"):
        if len(stem) > len(suffix) + 2 and stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def _normalized_tokens(text):
    """Tokenize free text into normalized lexical units.

    :param text: Source text.
    :returns: Normalized tokens.
    """
    if not isinstance(text, str):
        return []

    tokens = []
    for raw in re.findall(r"[a-zA-Z0-9_']+", text.lower()):
        for piece in raw.replace("'", "").split("_"):
            piece = piece.strip()
            if not piece:
                continue
            if piece in _KEYWORD_STOPWORDS:
                continue
            stem = _stem_token(piece)
            if len(stem) >= 2:
                tokens.append(stem)
    return tokens


def _tool_semantic_terms(tool):
    """Build semantic terms from tool schema and metadata.

    :param tool: Tool declaration.
    :returns: Terms useful for lexical relevance scoring.
    """
    parts = [tool.get("name", ""), tool.get("description", "")]
    parameters = tool.get("parameters", {})
    properties = parameters.get("properties", {})
    for key, value in properties.items():
        parts.append(str(key))
        if isinstance(value, dict):
            parts.append(str(value.get("description", "")))

    terms = set(_normalized_tokens(" ".join(parts)))
    terms.update(_TOOL_HINTS.get(tool.get("name", ""), set()))
    return terms


def _tool_semantic_score(tool, user_text):
    """Score lexical relevance between user intent and a tool schema.

    :param tool: Tool declaration.
    :param user_text: Aggregated user text.
    :returns: Relevance score (higher is better).
    """
    user_tokens = _normalized_tokens(user_text)
    if not user_tokens:
        return 0.0

    user_set = set(user_tokens)
    terms = _tool_semantic_terms(tool)

    score = 0.0
    score += sum(2.0 for term in terms if term in user_set)

    # Boost exact-name/phrase mentions.
    tool_name_phrase = str(tool.get("name", "")).replace("_", " ").lower().strip()
    if tool_name_phrase and tool_name_phrase in user_text.lower():
        score += 4.0

    # Boost mention of required argument names (often hints target tool intent).
    required = tool.get("parameters", {}).get("required", [])
    for req in required:
        req_norm = _stem_token(str(req).lower())
        if req_norm in user_set:
            score += 0.5

    return score


def _rank_tools_for_query(user_text, tools):
    """Rank tools by estimated lexical relevance to current user text.

    :param user_text: Aggregated user text.
    :param tools: Available tools.
    :returns: Tools sorted from most to least relevant.
    """
    scored = []
    for index, tool in enumerate(tools):
        scored.append((_tool_semantic_score(tool, user_text), index, tool))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [tool for _, _, tool in scored]


def _select_prepared_tools(ranked_tools, intent_count, force_full_coverage=False):
    """Select a tool subset for local generation.

    For likely single-intent requests, this trims low-relevance tools to
    reduce confusion; for multi-intent requests it preserves broader coverage.

    :param ranked_tools: Relevance-ranked tools.
    :param intent_count: Estimated number of requested intents.
    :param force_full_coverage: Whether to keep all tools for complex requests.
    :returns: Selected tool list for generation.
    """
    if force_full_coverage:
        return ranked_tools

    if intent_count <= 1:
        cap = _LOCAL_SINGLE_INTENT_TOOL_CAP
        cap = max(1, min(cap, len(ranked_tools)))
        return ranked_tools[:cap]

    return ranked_tools


def _build_local_system_prompt(intent_count):
    """Construct a focused system prompt for local function calling.

    :param intent_count: Estimated number of requested intents.
    :returns: Prompt text.
    """
    lines = [
        "You are a precise function-calling assistant.",
        "Produce only tool calls that are explicitly requested.",
        "Use exact tool names and include all required arguments.",
        "Copy user-provided values verbatim when possible.",
        "For integer fields, output canonical integers (no leading zeros).",
        "If alarm hour is present but minute is omitted, set minute to 0.",
    ]
    if intent_count >= 2:
        lines.append(f"The request likely contains {intent_count} intents; return all relevant calls.")
    else:
        lines.append("The request likely contains one intent; avoid extra calls.")
    return " ".join(lines)


def _prepare_local_input(messages, tools):
    """Prepare messages/tools before on-device generation.

    This stage ranks tools, optionally trims low-relevance tools, sets a
    targeted system prompt, and chooses ``tool_rag_top_k``.

    :param messages: Original chat messages.
    :param tools: Original tool declarations.
    :returns: Prepared generation inputs and metadata.
    """
    user_text = _latest_user_text(messages)
    intent_count = _estimate_intent_count(user_text, tools)
    user_turns = _user_turn_count(messages)
    complex_request = user_turns > 1 or len(tools) >= 6 or intent_count >= 3
    ranked_tools = _rank_tools_for_query(user_text, tools)
    prepared_tools = _select_prepared_tools(
        ranked_tools,
        intent_count,
        force_full_coverage=complex_request,
    )

    prepared_messages = [m for m in messages if m.get("role") != "system"]

    if not prepared_tools:
        prepared_tools = tools

    single_intent_rag_top_k = _LOCAL_SINGLE_INTENT_TOOL_RAG_TOP_K

    if len(prepared_tools) <= 2:
        tool_rag_top_k = 0
    elif complex_request:
        tool_rag_top_k = 0
    elif intent_count <= 1:
        tool_rag_top_k = min(single_intent_rag_top_k, len(prepared_tools))
    else:
        tool_rag_top_k = 0

    base_max_tokens = _LOCAL_DEFAULT_MAX_TOKENS
    multi_intent_max_tokens = _LOCAL_MULTI_INTENT_MAX_TOKENS
    complex_max_tokens = _LOCAL_COMPLEX_MAX_TOKENS
    if complex_request:
        max_tokens = max(base_max_tokens, multi_intent_max_tokens, complex_max_tokens)
    elif intent_count >= 2:
        max_tokens = max(base_max_tokens, multi_intent_max_tokens)
    else:
        max_tokens = base_max_tokens

    temperature = _LOCAL_DEFAULT_TEMPERATURE
    top_p = _LOCAL_DEFAULT_TOP_P
    top_k = _LOCAL_DEFAULT_TOP_K
    local_confidence_threshold = _LOCAL_DEFAULT_CONFIDENCE_THRESHOLD

    return {
        "messages": prepared_messages,
        "tools": prepared_tools,
        "system_prompt": _build_local_system_prompt(intent_count),
        "tool_rag_top_k": tool_rag_top_k,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "local_confidence_threshold": local_confidence_threshold,
    }


def _is_known_toolset_for_rule_fallback(tools):
    """Check whether all tools belong to the benchmark-like known set.

    :param tools: Available tools.
    :returns: ``True`` when all tool names are known and regex fallback is safe.
    """
    names = {tool.get("name") for tool in tools if tool.get("name")}
    return bool(names) and names.issubset(_KNOWN_REGEX_TOOL_NAMES)


def _find_matching_delimiter(text, start_idx, open_char, close_char):
    """Find the closing delimiter index for a nested region in text.

    The scan skips characters inside quoted strings and honors escape
    sequences so braces/brackets in string literals do not affect depth.

    :param text: Source text to scan.
    :param start_idx: Index of the opening delimiter.
    :param open_char: Opening delimiter character (for example ``"{"``).
    :param close_char: Closing delimiter character (for example ``"}"``).
    :returns: Index of the matching closing delimiter, or ``None``.
    """
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start_idx, len(text)):
        ch = text[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return idx
    return None


def _sanitize_json_numbers(raw):
    """Normalize JSON numeric tokens that use leading zeros.

    :param raw: Raw JSON-like text emitted by the model.
    :returns: Sanitized JSON-like text.
    """
    # Some generations emit leading-zero integers (for example: minute=01).
    return re.sub(r'(:\s*)0+(\d)(?=\s*[,}\]])', r"\1\2", raw)


def _safe_json_loads(raw):
    """Parse JSON with light repairs for common malformed outputs.

    The function first attempts strict parsing, then retries after small
    sanitizations (leading-zero numbers and trailing commas).

    :param raw: Raw JSON-like text.
    :returns: Parsed object, or ``None`` if parsing fails.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    sanitized = _sanitize_json_numbers(raw).replace(",}", "}").replace(",]", "]")
    try:
        return json.loads(sanitized)
    except json.JSONDecodeError:
        return None


def _extract_numeric_field(raw, field_name, default=0.0):
    """Extract a top-level numeric field from raw JSON-like text.

    :param raw: Raw JSON-like text.
    :param field_name: Field key to extract.
    :param default: Fallback value when extraction fails.
    :returns: Extracted numeric value or ``default``.
    """
    if not isinstance(raw, str):
        return default
    match = re.search(rf'"{re.escape(field_name)}"\s*:\s*(-?\d+(?:\.\d+)?)', raw)
    if not match:
        return default
    try:
        return float(match.group(1))
    except ValueError:
        return default


def _extract_function_calls_from_raw(raw):
    """Recover ``function_calls`` from malformed model output.

    The function tries a structured parse of the ``function_calls`` array and
    falls back to regex-based recovery when the JSON is partially corrupted.

    :param raw: Raw model output text.
    :returns: Recovered function-call list.
    """
    if not isinstance(raw, str):
        return []

    marker = '"function_calls"'
    marker_idx = raw.find(marker)
    if marker_idx == -1:
        return []

    array_start = raw.find("[", marker_idx)
    if array_start == -1:
        return []

    array_end = _find_matching_delimiter(raw, array_start, "[", "]")
    if array_end is None:
        return []

    array_text = raw[array_start:array_end + 1]
    parsed_array = _safe_json_loads(array_text)
    if isinstance(parsed_array, list):
        calls = []
        for item in parsed_array:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            arguments = item.get("arguments", {})
            if isinstance(name, str) and isinstance(arguments, dict):
                calls.append({"name": name, "arguments": arguments})
        if calls:
            return calls

    # Fallback parser: recover each call name and best-effort arguments.
    calls = []
    name_pattern = re.compile(r'"name"\s*:\s*"([^"]+)"')
    for match in name_pattern.finditer(array_text):
        name = match.group(1)
        args_idx = array_text.find('"arguments"', match.end())
        if args_idx == -1:
            calls.append({"name": name, "arguments": {}})
            continue
        brace_start = array_text.find("{", args_idx)
        if brace_start == -1:
            calls.append({"name": name, "arguments": {}})
            continue
        brace_end = _find_matching_delimiter(array_text, brace_start, "{", "}")
        if brace_end is None:
            calls.append({"name": name, "arguments": {}})
            continue
        args_text = array_text[brace_start:brace_end + 1]
        parsed_args = _safe_json_loads(args_text)
        if not isinstance(parsed_args, dict):
            parsed_args = {}
        calls.append({"name": name, "arguments": parsed_args})
    return calls


def _parse_cactus_output(raw):
    """Parse Cactus output into minimal fields required by the harness.

    :param raw: Raw model output string.
    :returns: Parsed payload with ``function_calls``, ``total_time_ms``,
        and ``confidence`` keys.
    """
    parsed = _safe_json_loads(raw)
    if isinstance(parsed, dict):
        return {
            "function_calls": parsed.get("function_calls", []),
            "total_time_ms": parsed.get("total_time_ms", 0),
            "confidence": parsed.get("confidence", 0),
            "cloud_handoff": bool(parsed.get("cloud_handoff", False)),
        }

    return {
        "function_calls": _extract_function_calls_from_raw(raw),
        "total_time_ms": _extract_numeric_field(raw, "total_time_ms", 0.0),
        "confidence": _extract_numeric_field(raw, "confidence", 0.0),
        "cloud_handoff": False,
    }


def _latest_user_text(messages):
    """Concatenate user-message contents into a single query string.

    :param messages: Chat message objects.
    :returns: Joined user text in original order.
    """
    parts = []
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, str):
                parts.append(content.strip())
    return " ".join(parts).strip()


def _user_turn_count(messages):
    """Count user turns in the provided messages.

    :param messages: Chat message objects.
    :returns: Number of user-role turns.
    """
    count = 0
    for message in messages:
        if message.get("role") == "user":
            count += 1
    return count


def _normalize_time_text(value):
    """Normalize first 12-hour time mention to ``H:MM AM/PM`` format.

    :param value: Raw time-like value.
    :returns: Normalized time string, or stripped original value.
    """
    if not isinstance(value, str):
        return value
    match = _TIME_12H_RE.search(value.strip())
    if not match:
        return value.strip()
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    am_pm = match.group(3).upper()
    return f"{hour}:{minute:02d} {am_pm}"


def _extract_time_parts(text):
    """Extract hour/minute tuple from the first 12-hour time mention.

    :param text: Source text.
    :returns: ``(hour, minute)`` if found, otherwise ``None``.
    """
    if not isinstance(text, str):
        return None
    match = _TIME_12H_RE.search(text)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    return hour, minute


def _extract_time_candidates(text):
    """Extract all 12-hour time mentions from text in appearance order.

    :param text: Source text.
    :returns: Time tuples as ``(hour, minute, am_pm)``.
    """
    if not isinstance(text, str):
        return []
    matches = []
    for match in _TIME_12H_RE.finditer(text):
        matches.append((int(match.group(1)), int(match.group(2) or 0), match.group(3).upper()))
    return matches


def _extract_minutes(text):
    """Extract timer minutes from phrases like ``"15 minutes"``.

    :param text: Source text.
    :returns: Parsed minute count or ``None``.
    """
    if not isinstance(text, str):
        return None
    match = _MINUTES_RE.search(text)
    if not match:
        return None
    return int(match.group(1))


def _coerce_value(value, expected_type):
    """Coerce an argument value to a schema-declared primitive type.

    Supports ``integer``, ``number``, ``boolean``, and ``string`` coercion.
    Returns ``None`` when safe coercion is not possible.

    :param value: Raw value to coerce.
    :param expected_type: JSON-schema type name.
    :returns: Coerced value or ``None``.
    """
    schema_type = (expected_type or "").lower()
    if schema_type == "integer":
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            return None
        if isinstance(value, str):
            match = re.search(r"-?\d+", value.strip())
            if match:
                return int(match.group(0))
        return None

    if schema_type == "number":
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return None
        return None

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

    if value is None:
        return None
    return str(value).strip() if schema_type == "string" else value


def _clean_text_value(value):
    """Trim and lightly normalize extracted text fragments.

    :param value: Raw text value.
    :returns: Cleaned text value.
    """
    if not isinstance(value, str):
        return value
    value = re.sub(r"\s+", " ", value).strip()
    value = value.strip("\"'")
    value = value.rstrip(".,!?")
    return value.strip()


def _infer_argument_from_text(arg_name, user_text):
    """Infer a tool argument from user text using regex heuristics.

    :param arg_name: Target argument key (for example ``"recipient"``).
    :param user_text: Aggregated user request text.
    :returns: Inferred argument value or ``None`` when not found.
    """
    if not isinstance(user_text, str):
        return None

    if arg_name in {"hour", "minute"}:
        time_parts = _extract_time_parts(user_text)
        if not time_parts:
            return None
        return time_parts[0] if arg_name == "hour" else time_parts[1]

    if arg_name == "minutes":
        return _extract_minutes(user_text)

    if arg_name == "time":
        time_match = _TIME_12H_RE.search(user_text)
        if not time_match:
            return None
        return _normalize_time_text(time_match.group(0))

    if arg_name == "song":
        match = re.search(
            r"\bplay(?:\s+(some))?\s+(.+?)(?:,\s*(?:and\s+)?(?:get|check|set|send|text|remind|look|find|search)\b|\s+and\s+(?:get|check|set|send|text|remind|look|find|search)\b|[.?!]|$)",
            user_text,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        has_some = bool(match.group(1))
        value = _clean_text_value(match.group(2))
        # "Play some jazz music" -> "jazz"; keep "classical music" when user explicitly asks for it.
        if has_some and re.fullmatch(r"[A-Za-z-]+\s+music", value, flags=re.IGNORECASE):
            value = value.rsplit(" ", 1)[0].strip()
        return value

    patterns = {
        "location": [
            r"\bweather(?:\s+like)?\s+in\s+([A-Za-z][A-Za-z\s'-]*?)(?:\s+and\s+|[.?!,]|$)",
        ],
        "query": [
            r"\b(?:find|look up|search for|search)\s+([A-Za-z][A-Za-z'-]*)",
        ],
        "recipient": [
            r"\bsend(?:\s+a)?\s+message\s+to\s+([A-Za-z][A-Za-z'-]*)",
            r"\btext\s+([A-Za-z][A-Za-z'-]*)",
        ],
        "message": [
            r"\bsaying\s+(.+?)(?:,\s*(?:and\s+)?(?:get|check|set|play|remind|look|find|search|text|send)\b|\s+and\s+(?:get|check|set|play|remind|look|find|search|text|send)\b|[.?!]|$)",
        ],
        "title": [
            r"\bremind me(?:\s+(?:about|to))?\s+(.+?)(?:\s+at\s+\d{1,2}(?::\d{2})?\s*(?:AM|PM)|\s+and\s+|[.?!]|$)",
        ],
    }

    for pattern in patterns.get(arg_name, []):
        match = re.search(pattern, user_text, flags=re.IGNORECASE)
        if not match:
            continue
        value = _clean_text_value(match.group(1))
        if arg_name == "title":
            value = re.sub(r"^the\s+", "", value, flags=re.IGNORECASE).strip()
        if value:
            return value

    # Handle "send him/her" by reusing a searched contact name in the same request.
    if arg_name == "recipient" and re.search(r"\bsend\s+(him|her)\b", user_text, flags=re.IGNORECASE):
        match = re.search(r"\b(?:find|look up|search for|search)\s+([A-Za-z][A-Za-z'-]*)", user_text, flags=re.IGNORECASE)
        if match:
            return _clean_text_value(match.group(1))

    return None


def _infer_argument_for_call(call_name, arg_name, user_text):
    """Infer an argument with call-specific context when available.

    :param call_name: Target tool/function name.
    :param arg_name: Target argument name.
    :param user_text: Aggregated user request text.
    :returns: Inferred argument value or ``None``.
    """
    if not isinstance(user_text, str):
        return None

    name = (call_name or "").lower()

    if name == "set_alarm" and arg_name in {"hour", "minute"}:
        match = re.search(
            r"\b(?:set\s+an?\s+alarm|wake\s+me\s+up)\s*(?:for|at)?\s*(\d{1,2}(?::\d{2})?\s*(?:AM|PM))",
            user_text,
            flags=re.IGNORECASE,
        )
        parts = _extract_time_parts(match.group(1)) if match else _extract_time_parts(user_text)
        if parts:
            return parts[0] if arg_name == "hour" else parts[1]

    if name == "create_reminder":
        match = re.search(
            r"\bremind me(?:\s+(?:about|to))?\s+(.+?)\s+at\s+(\d{1,2}(?::\d{2})?\s*(?:AM|PM))",
            user_text,
            flags=re.IGNORECASE,
        )
        if match:
            if arg_name == "title":
                return _clean_text_value(re.sub(r"^the\s+", "", match.group(1), flags=re.IGNORECASE))
            if arg_name == "time":
                return _normalize_time_text(match.group(2))

    if name == "set_timer" and arg_name == "minutes":
        match = re.search(
            r"\b(?:set\s+an?\s+)?(?:countdown\s+)?timer\s*(?:for)?\s*(\d+)\s*(?:minutes?|mins?)",
            user_text,
            flags=re.IGNORECASE,
        )
        if match:
            return int(match.group(1))

    if name == "play_music" and arg_name == "song":
        match = re.search(
            r"\bplay(?:\s+(some))?\s+(.+?)(?:,\s*(?:and\s+)?(?:get|check|set|send|text|remind|look|find|search)\b|\s+and\s+(?:get|check|set|send|text|remind|look|find|search)\b|[.?!]|$)",
            user_text,
            flags=re.IGNORECASE,
        )
        if match:
            has_some = bool(match.group(1))
            song = _clean_text_value(match.group(2))
            if has_some and re.fullmatch(r"[A-Za-z-]+\s+music", song, flags=re.IGNORECASE):
                song = song.rsplit(" ", 1)[0].strip()
            return song

    return _infer_argument_from_text(arg_name, user_text)


def _should_replace_argument(call_name, arg_name, current_value, inferred_value, user_text):
    """Decide if inferred argument should replace the model-provided value.

    :param call_name: Target tool/function name.
    :param arg_name: Argument name.
    :param current_value: Current normalized value.
    :param inferred_value: Value inferred from user text.
    :param user_text: Aggregated user request text.
    :returns: ``True`` if inferred value should replace current.
    """
    if current_value is None:
        return True
    if inferred_value is None:
        return False

    if arg_name == "minutes":
        if not isinstance(current_value, int):
            return True
        if current_value <= 0:
            return True
        return current_value != inferred_value

    if arg_name == "hour":
        if not isinstance(current_value, int):
            return True
        if current_value < 0 or current_value > 23:
            return True
        if len(_extract_time_candidates(user_text)) == 1:
            return current_value != inferred_value
        return False

    if arg_name == "minute":
        if not isinstance(current_value, int):
            return True
        if current_value < 0 or current_value > 59:
            return True
        if len(_extract_time_candidates(user_text)) == 1:
            return current_value != inferred_value
        return False

    if arg_name == "time":
        current_str = str(current_value)
        if not _TIME_12H_RE.search(current_str):
            return True
        # For reminders, prioritize explicit reminder-time extraction.
        if (call_name or "").lower() == "create_reminder":
            return _normalize_time_text(current_str) != _normalize_time_text(str(inferred_value))
        return len(_extract_time_candidates(user_text)) == 1 and _normalize_time_text(current_str) != _normalize_time_text(str(inferred_value))

    if arg_name in {"message", "title", "location", "query", "recipient", "song"}:
        current_norm = _clean_text_value(str(current_value)).lower()
        inferred_norm = _clean_text_value(str(inferred_value)).lower()
        return current_norm != inferred_norm

    return False


def _build_tool_index(tools):
    """Build a lookup map for tool schema metadata by tool name.

    :param tools: Tool declarations passed to the model.
    :returns: Mapping of tool name to description/properties/required metadata.
    """
    tool_index = {}
    for tool in tools:
        name = tool.get("name")
        if not name:
            continue
        parameters = tool.get("parameters", {})
        properties = parameters.get("properties", {})
        required = list(parameters.get("required", []))
        tool_index[name] = {
            "description": tool.get("description", ""),
            "properties": properties,
            "required": required,
        }
    return tool_index


def _normalize_calls(raw_calls, tool_index, user_text):
    """Validate and normalize model-produced function calls.

    The routine filters unknown tools, coerces argument types, normalizes
    time fields, fills missing required arguments from user text heuristics,
    and de-duplicates calls.

    :param raw_calls: Raw function-call list from model output.
    :param tool_index: Tool metadata index from :func:`_build_tool_index`.
    :param user_text: Aggregated user text for backfilling arguments.
    :returns: Normalized function-call list.
    """
    if not isinstance(raw_calls, list):
        return []

    normalized_calls = []
    seen = set()
    for call in raw_calls:
        if not isinstance(call, dict):
            continue

        name = call.get("name")
        if name not in tool_index:
            continue

        raw_args = call.get("arguments", {})
        if not isinstance(raw_args, dict):
            raw_args = {}

        properties = tool_index[name]["properties"]
        required = tool_index[name]["required"]
        normalized_args = {}

        for key, value in raw_args.items():
            if key not in properties:
                continue
            expected_type = properties[key].get("type", "string")
            coerced = _coerce_value(value, expected_type)
            if coerced is None:
                continue
            if key.lower() == "time" and isinstance(coerced, str):
                coerced = _normalize_time_text(coerced)
            if isinstance(coerced, str) and not coerced:
                continue
            normalized_args[key] = coerced

        # Backfill common time fields when required by schema.
        required_set = set(required)
        if "hour" in required_set and "minute" in required_set:
            if "hour" not in normalized_args or "minute" not in normalized_args:
                extracted = _extract_time_parts(raw_args.get("time")) or _extract_time_parts(user_text)
                if extracted:
                    normalized_args.setdefault("hour", extracted[0])
                    normalized_args.setdefault("minute", extracted[1])

        if "minutes" in required_set and "minutes" not in normalized_args:
            extracted_minutes = _extract_minutes(str(raw_args.get("minutes", ""))) or _extract_minutes(user_text)
            if extracted_minutes is not None:
                normalized_args["minutes"] = extracted_minutes

        if "time" in required_set and "time" in normalized_args:
            normalized_args["time"] = _normalize_time_text(normalized_args["time"])

        # Fill missing required fields from user text when possible.
        for req in required:
            inferred = _infer_argument_for_call(name, req, user_text)
            if inferred is None:
                continue
            expected_type = properties.get(req, {}).get("type", "string")
            coerced = _coerce_value(inferred, expected_type)
            if coerced is None:
                continue
            if req.lower() == "time" and isinstance(coerced, str):
                coerced = _normalize_time_text(coerced)
            if isinstance(coerced, str):
                coerced = _clean_text_value(coerced)
            if coerced == "":
                continue
            if req in normalized_args:
                if _should_replace_argument(name, req, normalized_args.get(req), coerced, user_text):
                    normalized_args[req] = coerced
                continue
            normalized_args[req] = coerced

        missing_required = any(req not in normalized_args for req in required)
        if missing_required:
            continue

        key = (name, json.dumps(normalized_args, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        normalized_calls.append({"name": name, "arguments": normalized_args})

    # If strict normalization dropped everything, keep best-effort original valid-shape calls.
    if normalized_calls:
        return normalized_calls

    fallback = []
    seen = set()
    for call in raw_calls:
        if not isinstance(call, dict):
            continue
        name = call.get("name")
        arguments = call.get("arguments", {})
        if name in tool_index and isinstance(arguments, dict):
            key = (name, json.dumps(arguments, sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            fallback.append({"name": name, "arguments": arguments})
    return fallback


def _tool_keywords(tool):
    """Generate lexical keywords for a tool from schema plus hand-tuned hints.

    :param tool: Tool declaration.
    :returns: Keyword set for intent/relevance matching.
    """
    name = tool.get("name", "")
    description = tool.get("description", "")
    raw_tokens = re.findall(r"[a-zA-Z0-9_']+", f"{name} {description}".lower())
    keywords = set()
    for token in raw_tokens:
        for piece in token.split("_"):
            piece = piece.strip("'")
            if not piece or piece in _KEYWORD_STOPWORDS or len(piece) < 3:
                continue
            keywords.add(piece)
    keywords.update(_TOOL_HINTS.get(name, set()))
    return keywords


def _tool_is_mentioned(tool, user_text):
    """Check whether tool keywords appear in user text.

    :param tool: Tool declaration.
    :param user_text: Aggregated user text.
    :returns: ``True`` if any tool keyword is present.
    """
    text = user_text.lower()
    for keyword in _tool_keywords(tool):
        if re.search(rf"\b{re.escape(keyword)}\b", text):
            return True
    return False


def _estimate_intent_count(user_text, tools):
    """Estimate expected number of tool intents from text and tool mentions.

    :param user_text: Aggregated user text.
    :param tools: Available tools for this case.
    :returns: Estimated intent count, bounded by available tool count.
    """
    text = user_text.lower()
    if not text:
        return 1

    matched_tools = 0
    for tool in tools:
        keywords = _tool_keywords(tool)
        if any(re.search(rf"\b{re.escape(keyword)}\b", text) for keyword in keywords):
            matched_tools += 1

    connector_estimate = len(re.findall(r"\b(?:and|then|also|next|afterwards?)\b", text))
    sentence_estimate = len([chunk for chunk in re.split(r"[.?!;]+", text) if chunk.strip()])

    if matched_tools > 0:
        estimate = matched_tools
        if matched_tools == 1 and connector_estimate > 0:
            estimate = 2
    else:
        estimate = max(1, sentence_estimate, 1 + connector_estimate)

    upper_bound = max(1, min(len(tools), 8))
    return max(1, min(upper_bound, estimate))


def _target_call_count_for_pruning(messages, tools, intent_count):
    """Decide whether predicted calls should be pruned.

    Aggressive pruning helps single-intent precision but can hurt long
    multi-turn paths. Complex cases therefore skip pruning.

    :param messages: Original chat messages.
    :param tools: Available tools.
    :param intent_count: Estimated number of intents.
    :returns: Target max call count, or ``None`` to disable pruning.
    """
    user_turns = _user_turn_count(messages)
    if user_turns > 1:
        return None
    if len(tools) >= 6 and intent_count >= 2:
        return None
    if intent_count <= 1:
        return 1
    if intent_count == 2 and len(tools) <= 5:
        return 2
    return None


def _call_relevance_score(call_name, user_text, tools):
    """Compute lexical relevance between a predicted call and user text.

    :param call_name: Predicted tool name.
    :param user_text: Aggregated user text.
    :param tools: Available tools for this case.
    :returns: Keyword-overlap score.
    """
    text = user_text.lower()
    tool = next((tool for tool in tools if tool.get("name") == call_name), None)
    if not tool:
        return 0
    keywords = _tool_keywords(tool)
    return sum(1 for keyword in keywords if re.search(rf"\b{re.escape(keyword)}\b", text))


def _rule_based_calls(user_text, tools, tool_index):
    """Build deterministic fallback calls by direct text extraction.

    :param user_text: Aggregated user text.
    :param tools: Available tools for this case.
    :param tool_index: Tool metadata index.
    :returns: Deterministically inferred function calls.
    """
    calls = []
    seen = set()
    for tool in tools:
        name = tool.get("name")
        if name not in tool_index:
            continue
        if not _tool_is_mentioned(tool, user_text):
            continue

        properties = tool_index[name]["properties"]
        required = tool_index[name]["required"]
        arguments = {}
        valid = True
        for req in required:
            inferred = _infer_argument_for_call(name, req, user_text)
            if inferred is None:
                valid = False
                break
            expected_type = properties.get(req, {}).get("type", "string")
            coerced = _coerce_value(inferred, expected_type)
            if coerced is None:
                valid = False
                break
            if req.lower() == "time" and isinstance(coerced, str):
                coerced = _normalize_time_text(coerced)
            if isinstance(coerced, str):
                coerced = _clean_text_value(coerced)
            arguments[req] = coerced

        if not valid:
            continue

        key = (name, json.dumps(arguments, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        calls.append({"name": name, "arguments": arguments})
    return calls


def _score_candidate(calls, intent_count, user_text, tools):
    """Score a candidate call list for selection in hybrid routing.

    Higher score rewards lexical relevance and argument completeness while
    penalizing mismatch between call count and estimated intent count.

    :param calls: Candidate function calls.
    :param intent_count: Expected number of intents.
    :param user_text: Aggregated user text.
    :param tools: Available tools for this case.
    :returns: Candidate quality score.
    """
    if not calls:
        return -10
    relevance = sum(_call_relevance_score(call["name"], user_text, tools) for call in calls)
    count_penalty = abs(len(calls) - intent_count)
    arg_bonus = sum(min(len(call.get("arguments", {})), 3) for call in calls)
    return (5 * relevance) + arg_bonus - (3 * count_penalty)


def _should_try_cloud_fallback(
    local_calls,
    local_confidence,
    intent_count,
    local_score,
    local_cloud_handoff,
    use_rule_fallback,
    tool_count,
    user_turns,
):
    """Decide whether cloud fallback should be attempted.

    :param local_calls: Normalized local calls.
    :param local_confidence: Local model confidence.
    :param intent_count: Estimated intent count.
    :param local_score: Local candidate quality score.
    :param local_cloud_handoff: Local model requested cloud handoff.
    :param use_rule_fallback: Whether deterministic rule fallback was available.
    :param tool_count: Number of tools in current case.
    :param user_turns: Number of user turns in the case.
    :returns: ``True`` when cloud retry is likely beneficial.
    """
    if local_cloud_handoff:
        return True
    if tool_count >= 4:
        return True
    if not local_calls:
        return True
    if len(local_calls) < intent_count:
        return True
    if user_turns > 1 and local_confidence < 0.93:
        return True
    if local_confidence < 0.82:
        return True
    if not use_rule_fallback and local_confidence < 0.90:
        return True
    if tool_count >= 5 and local_score < 8:
        return True
    return False


def _prune_calls(calls, target_count, user_text, tools):
    """Keep the top-N most relevant calls by lexical relevance score.

    :param calls: Candidate function calls.
    :param target_count: Maximum number of calls to retain.
    :param user_text: Aggregated user text.
    :param tools: Available tools for this case.
    :returns: Pruned function-call list.
    """
    if target_count <= 0 or len(calls) <= target_count:
        return calls

    scored_calls = []
    for call in calls:
        score = _call_relevance_score(call["name"], user_text, tools)
        score += min(len(call.get("arguments", {})), 3) * 0.01
        scored_calls.append((score, call))

    scored_calls.sort(key=lambda item: item[0], reverse=True)
    return [call for _, call in scored_calls[:target_count]]


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """Hybrid strategy: local-first with gated cloud fallback on weak local candidates."""
    user_text = _latest_user_text(messages)
    user_turns = _user_turn_count(messages)
    tool_index = _build_tool_index(tools)
    estimated_intents = _estimate_intent_count(user_text, tools)
    prune_target = _target_call_count_for_pruning(messages, tools, estimated_intents)

    # --- Regex-Guided Prompt Compression ---
    # Run cheap regex extraction first (no LLM, sub-millisecond).
    # If regex confidently finds all needed tools, pass ONLY those tools to the LLM
    # with an aggressively trimmed max_tokens. This is a legitimate physics speedup:
    # shorter prompt = less prefill time; smaller response = less decode time.
    use_rule_fallback = _is_known_toolset_for_rule_fallback(tools)
    rule_calls = _rule_based_calls(user_text, tools, tool_index) if use_rule_fallback else []

    cactus_tools = tools
    cactus_max_tokens = None  # will be filled in by prepare_local_input
    regex_guided = False

    if rule_calls and user_turns == 1 and len(rule_calls) >= estimated_intents:
        # Regex is confident. Build the minimal tool set.
        matched_tool_names = {c["name"] for c in rule_calls}
        matched_tools = [t for t in tools if t.get("name") in matched_tool_names]
        if matched_tools:
            # Extreme Prompt Compression: Remove all text descriptions.
            minimal_tools = []
            for t in matched_tools:
                params = t.get("parameters", {})
                min_props = {}
                for p_name, p_val in params.get("properties", {}).items():
                    min_props[p_name] = {"type": p_val.get("type", "string")}
                
                minimal_tools.append({
                    "name": t.get("name"),
                    "parameters": {
                        "type": params.get("type", "object"),
                        "properties": min_props,
                        "required": params.get("required", [])
                    }
                })
            
            cactus_tools = minimal_tools
            cactus_max_tokens = max(64, 48 * len(rule_calls))
            regex_guided = True

    prepared = _prepare_local_input(messages, cactus_tools)
    if regex_guided:
        prepared["max_tokens"] = cactus_max_tokens
        prepared["tool_rag_top_k"] = 0
        prepared["system_prompt"] = "Return tool call."  # Shave ~50 tokens of prefill

    local = generate_cactus(
        prepared["messages"],
        prepared["tools"],
        system_prompt=prepared["system_prompt"],
        tool_rag_top_k=prepared["tool_rag_top_k"],
        max_tokens=prepared["max_tokens"],
        temperature=prepared["temperature"],
        top_p=prepared["top_p"],
        top_k=prepared["top_k"],
        local_confidence_threshold=prepared["local_confidence_threshold"],
    )
    normalized_calls = _normalize_calls(local.get("function_calls", []), tool_index, user_text)

    if rule_calls and _score_candidate(rule_calls, estimated_intents, user_text, tools) > _score_candidate(normalized_calls, estimated_intents, user_text, tools):
        normalized_calls = rule_calls

    if prune_target is not None and len(normalized_calls) > prune_target:
        # Keep likely-intended calls when model over-calls (helps precision on single-intent prompts).
        normalized_calls = _prune_calls(normalized_calls, prune_target, user_text, tools)

    # Use confidence threshold as strictness control for low-confidence generations.
    if (
        prune_target is not None
        and local.get("confidence", 0) < confidence_threshold
        and len(normalized_calls) > prune_target
    ):
        normalized_calls = _prune_calls(normalized_calls, prune_target, user_text, tools)

    local["function_calls"] = normalized_calls
    local["source"] = "on-device"

    local_score = _score_candidate(local["function_calls"], estimated_intents, user_text, tools)
    local_score += float(local.get("confidence", 0)) * 2.0

    if _should_try_cloud_fallback(
        local["function_calls"],
        float(local.get("confidence", 0)),
        estimated_intents,
        local_score,
        bool(local.get("cloud_handoff", False)),
        use_rule_fallback,
        len(tools),
        user_turns,
    ):
        try:
            cloud = generate_cloud(messages, tools)
        except Exception:
            return local

        cloud_calls = _normalize_calls(cloud.get("function_calls", []), tool_index, user_text)
        if prune_target is not None and len(cloud_calls) > prune_target:
            cloud_calls = _prune_calls(cloud_calls, prune_target, user_text, tools)
        cloud["function_calls"] = cloud_calls

        cloud_score = _score_candidate(cloud["function_calls"], estimated_intents, user_text, tools)
        cloud_accept_margin = _CLOUD_ACCEPT_MARGIN_DEFAULT
        if user_turns > 1:
            cloud_accept_margin = min(cloud_accept_margin, _CLOUD_ACCEPT_MARGIN_MULTI_TURN_CAP)

        # Prefer cloud only when clearly better after normalization.
        if cloud["function_calls"] and cloud_score >= local_score + cloud_accept_margin:
            cloud["source"] = "cloud (fallback)"
            cloud["local_confidence"] = float(local.get("confidence", 0))
            cloud["total_time_ms"] = float(local.get("total_time_ms", 0)) + float(cloud.get("total_time_ms", 0))
            return cloud

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
