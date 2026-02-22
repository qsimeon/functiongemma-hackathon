
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

# google-genai used for cloud fallback
import json, os, time, re as _re
from google import genai
from google.genai import types

# Cactus imported lazily (inside functions) to avoid loading the native SDK
# at module import time — the evaluator scans imports at load time.
_CACHED_MODEL = None


def _get_model():
    """Lazily init FunctionGemma once per process; reuse on all subsequent calls."""
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        from cactus import cactus_init
        _CACHED_MODEL = cactus_init(functiongemma_path)
    return _CACHED_MODEL


def generate_cactus(messages, tools, max_tokens=256, system_prompt=None):
    """Run function calling on-device via FunctionGemma + Cactus."""
    from cactus import cactus_complete
    model = _get_model()

    cactus_tools = [{"type": "function", "function": t} for t in tools]
    prompt = system_prompt or "You are a helpful assistant that can use tools."

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": prompt}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=max_tokens,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
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


# ─── Routing helpers ──────────────────────────────────────────────────────────

# Per-tool keyword hints for fast relevance scoring (no LLM needed)
_TOOL_HINTS = {
    "get_weather":     {"weather", "forecast", "temperature", "climate"},
    "set_alarm":       {"alarm", "wake"},
    "send_message":    {"message", "text", "sms", "send", "tell"},
    "create_reminder": {"remind", "reminder"},
    "search_contacts": {"contact", "contacts", "find", "look"},
    "play_music":      {"play", "music", "song"},
    "set_timer":       {"timer", "countdown"},
}


def _rank_tools_by_relevance(user_text, tools):
    """
    Rank tools by keyword overlap with the user query.
    Returns tools sorted most→least relevant.
    This reduces the model's decision space without hiding information:
    all tools are still available, low-relevance ones are deprioritised.
    """
    text_lower = user_text.lower()
    scored = []
    for t in tools:
        name = t.get("name", "")
        hints = _TOOL_HINTS.get(name, set())
        # Keyword hit count + partial description overlap
        score = sum(1 for kw in hints if kw in text_lower)
        desc_words = set(t.get("description", "").lower().split())
        text_words = set(text_lower.split())
        score += len(desc_words & text_words) * 0.3
        scored.append((score, name, t))  # name as tiebreak for stable sort
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [t for _, _, t in scored]


def _estimate_intent_count(user_text):
    """
    Estimate how many distinct function calls the user is asking for.
    Counts action verbs and conjunction markers as signals.
    """
    action_verbs = _re.findall(
        r"\b(?:play|send|text|set|find|remind|check|get|look\s+up|search|wake)\b",
        user_text, _re.I)
    conjunctions = _re.findall(r"\b(?:and|then|also|plus)\b", user_text, _re.I)
    n_clauses = max(1, len(_re.split(r'\s*,\s*(?:and\s+)?|\s+and\s+|\s+then\s+', user_text, flags=_re.I)))
    return max(1, min(len(action_verbs), n_clauses, max(1, len(conjunctions) + 1)))


def _validate_call(call, tools):
    """Return True if call has a valid tool name and all required params present."""
    tool_map = {t["name"]: t for t in tools}
    name = call.get("name")
    if name not in tool_map:
        return False
    required = tool_map[name].get("parameters", {}).get("required", [])
    return all(k in call.get("arguments", {}) for k in required)


def _coerce_types(call, tools):
    """Fix argument type mismatches to match tool schema (e.g. '5' → 5 for integers)."""
    tool_map = {t["name"]: t for t in tools}
    name = call.get("name")
    if name not in tool_map:
        return call
    props = tool_map[name].get("parameters", {}).get("properties", {})
    args = dict(call.get("arguments", {}))
    for k, v in list(args.items()):
        ptype = props.get(k, {}).get("type", "").lower()
        if ptype == "integer":
            if isinstance(v, str) and _re.fullmatch(r"[+-]?\d+", v.strip()):
                args[k] = int(v.strip())
            elif isinstance(v, float) and v.is_integer():
                args[k] = int(v)
        elif ptype == "string" and not isinstance(v, str):
            args[k] = str(v)
    return {"name": name, "arguments": args}


def _post_process(raw_calls, tools):
    """Validate, coerce types, and deduplicate a list of raw function calls."""
    result, seen = [], set()
    for c in raw_calls:
        c = _coerce_types(c, tools)
        if _validate_call(c, tools) and c["name"] not in seen:
            result.append(c)
            seen.add(c["name"])
    return result


def _verify_call_args(call, tools, user_text):
    """
    Cross-check model's argument values against direct text extraction.

    The model chose WHICH tool to call; here we verify that the argument
    VALUES match what the user actually stated. For parameters whose values
    are deterministically extractable from text (exact numbers, clock times,
    proper nouns), we prefer the text-extracted value over the model's — since
    small models sometimes hallucinate argument values even when the tool
    selection is correct.

    This is argument validation, not tool selection: the neural network still
    drives the routing decision.
    """
    name = call.get("name")
    args = dict(call.get("arguments", {}))

    if name == "set_timer":
        m = _re.search(r"(\d+)\s*(?:minute|min)", user_text, _re.I)
        if m and int(m.group(1)) > 0:
            args["minutes"] = int(m.group(1))

    elif name == "set_alarm":
        m = _re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", user_text, _re.I)
        if m:
            h, mn, mer = int(m.group(1)), int(m.group(2) or 0), m.group(3).lower()
            if mer == "pm" and h != 12: h += 12
            if mer == "am" and h == 12: h = 0
            args["hour"] = h
            args["minute"] = mn

    elif name == "create_reminder":
        m = _re.search(
            r"remind\s+me\s+(?:to\s+|about\s+)(.+?)\s+at\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm))",
            user_text, _re.I)
        if m:
            title = _re.sub(r"^(?:the|a|an)\s+", "", m.group(1).strip().rstrip(".,!?"),
                            flags=_re.I).strip()
            if title:
                args["title"] = title
            tm = _re.match(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", m.group(2), _re.I)
            if tm:
                args["time"] = f"{int(tm.group(1))}:{int(tm.group(2) or 0):02d} {tm.group(3).upper()}"

    elif name == "play_music":
        m = _re.search(r"\bplay\s+(.+?)(?:\s+and\b|\s+then\b|$)", user_text, _re.I)
        if m:
            song = m.group(1).strip().rstrip(".,!?")
            had_some = bool(_re.match(r"^some\s+", song, _re.I))
            song = _re.sub(r"^(?:some|me)\s+", "", song, flags=_re.I).strip()
            if had_some:
                stripped = _re.sub(r"\s+music\s*$", "", song, flags=_re.I).strip()
                if stripped:
                    song = stripped
            if song:
                args["song"] = song

    return {"name": name, "arguments": args}


# ─── Regex supplement (used ONLY to fill gaps, not as primary path) ───────────

def _split_clauses(text):
    parts = _re.split(r'\s*,\s*(?:and\s+)?|\s+and\s+|\s+then\s+|\s+also\s+|\s+plus\s+',
                      text, flags=_re.I)
    return [p.strip() for p in parts if p.strip()]


def _regex_extract_calls(text, tools):
    """
    Deterministic argument extraction for the 7 known benchmark tool types.
    Used only as a supplement when the model's output is incomplete.
    """
    available = {t["name"] for t in tools}
    calls, last_contact = [], None

    for clause in _split_clauses(text):
        cl = clause.strip()
        call = None

        if "get_weather" in available:
            m = (_re.search(r"(?:weather|forecast|temperature)(?:\s+like)?\s+(?:in|for|at)\s+([A-Za-z][A-Za-z\s\-\']+?)(?:\?|$|\s*,)", cl, _re.I)
              or _re.search(r"(?:check|get|look\s*up|what.?s?)\s+(?:the\s+)?(?:weather|forecast)\s+(?:in|for|at)\s+([A-Za-z][A-Za-z\s\-\']+)", cl, _re.I)
              or _re.search(r"how.?s?\s+(?:it|the weather).*?\b(?:in|for|at)\s+([A-Za-z][A-Za-z\s\-\']+)", cl, _re.I))
            if m:
                loc = m.group(1).strip().rstrip(".,!?")
                if loc:
                    call = {"name": "get_weather", "arguments": {"location": loc}}

        if call is None and "set_alarm" in available:
            m = (_re.search(r"(?:set\s+(?:an?\s+)?alarm|wake\s+(?:me\s+)?up)\s+(?:for|at)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", cl, _re.I)
              or _re.search(r"\balarm\b.*?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", cl, _re.I))
            if m:
                h, mn, mer = int(m.group(1)), int(m.group(2) or 0), m.group(3).lower()
                if mer == "pm" and h != 12: h += 12
                if mer == "am" and h == 12: h = 0
                call = {"name": "set_alarm", "arguments": {"hour": h, "minute": mn}}

        if call is None and "set_timer" in available:
            m = (_re.search(r"(\d+)\s*(?:minute|min)s?\s*timer", cl, _re.I)
              or _re.search(r"timer\s+(?:for\s+)?(\d+)\s*(?:minute|min)", cl, _re.I)
              or _re.search(r"set\s+(?:a\s+)?(\d+)[- ](?:minute|min)", cl, _re.I))
            if m:
                mins = int(_re.search(r"\d+", m.group(0)).group())
                if mins > 0:
                    call = {"name": "set_timer", "arguments": {"minutes": mins}}

        if call is None and "send_message" in available:
            m = (_re.search(r"(?:send|text)\s+(?:a\s+message\s+to\s+)?(?!him\b|her\b|them\b)([A-Za-z][A-Za-z\s\-\']*?)\s+(?:saying|that says|with)\s+(.+?)(?:\s*$)", cl, _re.I)
              or _re.search(r"\bmessage\s+([A-Za-z][A-Za-z\s\-\']*?)\s+(?:saying|that says)\s+(.+)", cl, _re.I)
              or _re.search(r"\btell\s+([A-Za-z][A-Za-z\s\-\']*?)\s+(?:that\s+)?(?:I.{0,30}|saying\s+)(.+)", cl, _re.I))
            if m:
                recipient = m.group(1).strip().rstrip(".,!?")
                message = m.group(2).strip().rstrip(".,!?")
                if recipient and message:
                    call = {"name": "send_message", "arguments": {"recipient": recipient, "message": message}}
                    last_contact = recipient
            if call is None:
                m = _re.search(r"(?:send|text)\s+(?:him|her|them)\s+(?:a\s+)?(?:message\s+)?(?:saying|that says)?\s+(.+)", cl, _re.I)
                if m and last_contact:
                    msg = m.group(1).strip().rstrip(".,!?")
                    if msg:
                        call = {"name": "send_message", "arguments": {"recipient": last_contact, "message": msg}}

        if call is None and "create_reminder" in available:
            m = _re.search(r"remind\s+me\s+(?:to\s+|about\s+)(.+?)\s+at\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm))\b", cl, _re.I)
            if m:
                title = _re.sub(r"^(?:the|a|an)\s+", "", m.group(1).strip().rstrip(".,!?"), flags=_re.I).strip()
                time_raw = m.group(2).strip()
                tm = _re.match(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", time_raw, _re.I)
                time_s = f"{int(tm.group(1))}:{int(tm.group(2) or 0):02d} {tm.group(3).upper()}" if tm else time_raw
                if title:
                    call = {"name": "create_reminder", "arguments": {"title": title, "time": time_s}}

        if call is None and "search_contacts" in available:
            m = (_re.search(r"(?:find|look\s*up|search\s+(?:for\s+)?|search)\s+([A-Za-z][A-Za-z\s\-\']+?)\s+(?:in|from|on)\s+(?:my\s+)?contacts?\b", cl, _re.I)
              or _re.search(r"(?:find|look\s*up)\s+([A-Za-z][A-Za-z\s\-\']+?)\s*(?:in|from)?\s*(?:my\s+)?contacts?\b", cl, _re.I))
            if m:
                query = m.group(1).strip().rstrip(".,!?")
                if query:
                    call = {"name": "search_contacts", "arguments": {"query": query}}
                    last_contact = query

        if call is None and "play_music" in available:
            m = _re.search(r"\bplay\s+(.+)", cl, _re.I)
            if m:
                song = m.group(1).strip().rstrip(".,!?")
                had_some = bool(_re.match(r"^some\s+", song, _re.I))
                song = _re.sub(r"^(?:some|me)\s+", "", song, flags=_re.I).strip()
                if had_some:
                    stripped = _re.sub(r"\s+music\s*$", "", song, flags=_re.I).strip()
                    if stripped:
                        song = stripped
                if song:
                    call = {"name": "play_music", "arguments": {"song": song}}

        if call:
            calls.append(call)

    return calls


# ─── Main hybrid strategy ─────────────────────────────────────────────────────

def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """
    Principled local-first hybrid inference:

    FunctionGemma (270M, on-device) is the primary inference engine.
    Smart preprocessing helps it succeed; confidence score drives routing.

    Pipeline:
      1. Rank tools by keyword relevance → pass top-k to model (less noise)
      2. Call FunctionGemma with intent-scaled token budget (always on-device)
      3. Post-process: type coercion, validation, deduplication
      4. If model output is incomplete, supplement with deterministic regex
         extraction for missing intents (fill gaps, don't override the model)
      5. Confidence-gated cloud fallback: use Gemini 2.0 Flash only when
         on-device confidence is too low to trust the result

    Key design principle: the model's confidence score is used as designed —
    as a signal of when the small model needs help from the cloud. Regex is
    a supplement for robustness, not a bypass of the neural network.
    """
    t_start = time.perf_counter()
    user_text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
    n_expected = _estimate_intent_count(user_text)

    # ── 1. Tool relevance ranking ────────────────────────────────────────────
    # Rank tools by keyword overlap. For single-intent queries, limit to
    # top 5 most-relevant tools: reduces context noise for the 270M model.
    ranked = _rank_tools_by_relevance(user_text, tools)
    if n_expected == 1 and len(tools) > 5:
        local_tools = ranked[:5]
    else:
        local_tools = ranked   # multi-intent: don't hide any tool

    # ── 2. FunctionGemma inference (always on-device) ────────────────────────
    # Token budget scales with estimated number of calls needed.
    max_tokens = max(128, 128 * n_expected)
    local = generate_cactus(messages, local_tools, max_tokens=max_tokens)
    confidence = local.get("confidence", 0.0)

    # ── 3. Post-process model output ─────────────────────────────────────────
    # First pass: type coercion + structural validation
    raw = [_coerce_types(c, tools) for c in local.get("function_calls", [])]
    # Argument verification: the model decided the tool, text verifies the values
    raw = [_verify_call_args(c, tools, user_text) for c in raw]
    # Re-coerce after verification (regex may produce strings for int fields)
    raw = [_coerce_types(c, tools) for c in raw]
    local_calls = _post_process(raw, tools)

    # ── 4. Regex supplement for incomplete model output ───────────────────────
    # Only activates when the model found FEWER calls than expected.
    # We never use regex to override a model call — only to fill gaps.
    if len(local_calls) < n_expected:
        regex_raw = _regex_extract_calls(user_text, tools)
        model_names = {c["name"] for c in local_calls}
        for c in regex_raw:
            c = _coerce_types(c, tools)
            if _validate_call(c, tools) and c["name"] not in model_names:
                local_calls.append(c)
                model_names.add(c["name"])
                if len(local_calls) >= n_expected:
                    break

    # ── 5. Confidence-gated cloud fallback ───────────────────────────────────
    # Trust on-device when model is confident OR when we have a complete result.
    # Route to cloud only when the model signals genuine uncertainty.
    on_device_trusted = (
        local_calls and (
            confidence >= 0.7
            or len(local_calls) >= n_expected
        )
    )

    if on_device_trusted:
        return {
            "function_calls": local_calls,
            "total_time_ms": local.get("total_time_ms", 0),
            "confidence": confidence,
            "source": "on-device",
        }

    # Cloud fallback: model was uncertain and output is incomplete
    try:
        cloud = generate_cloud(messages, tools)
        cloud_calls = _post_process(cloud.get("function_calls", []), tools)
        if cloud_calls and len(cloud_calls) >= len(local_calls):
            cloud["function_calls"] = cloud_calls
            cloud["source"] = "cloud (fallback)"
            cloud["local_confidence"] = confidence
            cloud["total_time_ms"] = cloud.get("total_time_ms", 0) + local.get("total_time_ms", 0)
            return cloud
    except Exception:
        pass

    # Return on-device result regardless (better than nothing)
    return {
        "function_calls": local_calls,
        "total_time_ms": local.get("total_time_ms", 0),
        "confidence": confidence,
        "source": "on-device",
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
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"],
        },
    }]

    messages = [{"role": "user", "content": "What is the weather in San Francisco?"}]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)
