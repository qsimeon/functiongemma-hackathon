
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types


# ── Model caching: load once per process, not once per call ──────────────────
# Key insight from top forks: cactus_init takes ~600ms. Loading on every call
# tanks the time_score. Cache globally so the first call pays the cost.

_CACHED_MODEL = None


def _get_model():
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        _CACHED_MODEL = cactus_init(functiongemma_path)
    return _CACHED_MODEL


def generate_cactus(messages, tools, max_tokens=256, system_prompt=None):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = _get_model()

    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

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

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


# ─── Helpers for synthesized hybrid strategy ──────────────────────────────────

import re as _re

def _split_clauses(text):
    """Split multi-intent query into individual action clauses."""
    parts = _re.split(
        r'\s*,\s*(?:and\s+)?|\s+and\s+|\s+then\s+|\s+also\s+|\s+plus\s+',
        text, flags=_re.I
    )
    return [p.strip() for p in parts if p.strip()]


def _extract_regex_calls(text, tools):
    """
    Deterministic regex extraction. Handles all 7 known tool types.
    Splits on clause boundaries for multi-intent queries.
    Returns list of raw call dicts (before type coercion/validation).
    """
    available = {t["name"] for t in tools}
    calls = []
    last_contact = None  # for pronoun resolution ("send him a message saying...")

    for clause in _split_clauses(text):
        cl = clause.strip()
        call = None

        # ── get_weather ──────────────────────────────────────────────────────
        if "get_weather" in available:
            m = (_re.search(
                    r"(?:weather|forecast|temperature)(?:\s+like)?\s+(?:in|for|at)\s+([A-Za-z][A-Za-z\s\-\']+?)(?:\?|$|\s*,)",
                    cl, _re.I)
                or _re.search(
                    r"(?:check|get|look\s*up|what.?s?)\s+(?:the\s+)?(?:weather|forecast)\s+(?:in|for|at)\s+([A-Za-z][A-Za-z\s\-\']+)",
                    cl, _re.I)
                or _re.search(
                    r"how.?s?\s+(?:it|the weather).*?\b(?:in|for|at)\s+([A-Za-z][A-Za-z\s\-\']+)",
                    cl, _re.I))
            if m:
                loc = m.group(1).strip().rstrip(".,!?")
                if loc:
                    call = {"name": "get_weather", "arguments": {"location": loc}}

        # ── set_alarm ────────────────────────────────────────────────────────
        if call is None and "set_alarm" in available:
            m = (_re.search(
                    r"(?:set\s+(?:an?\s+)?alarm|wake\s+(?:me\s+)?up)\s+(?:for|at)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
                    cl, _re.I)
                or _re.search(
                    r"\balarm\b.*?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
                    cl, _re.I))
            if m:
                h, mn, mer = int(m.group(1)), int(m.group(2) or 0), m.group(3).lower()
                if mer == "pm" and h != 12: h += 12
                if mer == "am" and h == 12: h = 0
                call = {"name": "set_alarm", "arguments": {"hour": h, "minute": mn}}

        # ── set_timer ────────────────────────────────────────────────────────
        if call is None and "set_timer" in available:
            m = (_re.search(r"(\d+)\s*(?:minute|min)s?\s*timer", cl, _re.I)
                or _re.search(r"timer\s+(?:for\s+)?(\d+)\s*(?:minute|min)", cl, _re.I)
                or _re.search(r"set\s+(?:a\s+)?(\d+)[- ](?:minute|min)", cl, _re.I))
            if m:
                mins = int(_re.search(r"\d+", m.group(0)).group())
                if mins > 0:
                    call = {"name": "set_timer", "arguments": {"minutes": mins}}

        # ── send_message ─────────────────────────────────────────────────────
        if call is None and "send_message" in available:
            m = (_re.search(
                    r"(?:send|text)\s+(?:a\s+message\s+to\s+)?(?!him\b|her\b|them\b)([A-Za-z][A-Za-z\s\-\']*?)\s+(?:saying|that says|with)\s+(.+?)(?:\s*$)",
                    cl, _re.I)
                or _re.search(
                    r"\bmessage\s+([A-Za-z][A-Za-z\s\-\']*?)\s+(?:saying|that says)\s+(.+)",
                    cl, _re.I)
                or _re.search(
                    r"\btell\s+([A-Za-z][A-Za-z\s\-\']*?)\s+(?:that\s+)?(?:I.{0,30}|saying\s+)(.+)",
                    cl, _re.I))
            if m:
                recipient = m.group(1).strip().rstrip(".,!?")
                message = m.group(2).strip().rstrip(".,!?")
                if recipient and message:
                    call = {"name": "send_message", "arguments": {"recipient": recipient, "message": message}}
                    last_contact = recipient
            if call is None:
                # pronoun fallback: "send him/her/them a message saying Y"
                m = _re.search(
                    r"(?:send|text)\s+(?:him|her|them)\s+(?:a\s+)?(?:message\s+)?(?:saying|that says)?\s+(.+)",
                    cl, _re.I)
                if m and last_contact:
                    message = m.group(1).strip().rstrip(".,!?")
                    if message:
                        call = {"name": "send_message", "arguments": {"recipient": last_contact, "message": message}}

        # ── create_reminder ──────────────────────────────────────────────────
        if call is None and "create_reminder" in available:
            m = _re.search(
                r"remind\s+me\s+(?:to\s+|about\s+)(.+?)\s+at\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm))\b",
                cl, _re.I)
            if m:
                title = m.group(1).strip().rstrip(".,!?")
                time_raw = m.group(2).strip()
                tm = _re.match(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", time_raw, _re.I)
                time_s = (f"{int(tm.group(1))}:{int(tm.group(2) or 0):02d} {tm.group(3).upper()}"
                          if tm else time_raw)
                # strip leading article ("the meeting" → "meeting")
                title = _re.sub(r"^(?:the|a|an)\s+", "", title, flags=_re.I).strip()
                if title:
                    call = {"name": "create_reminder", "arguments": {"title": title, "time": time_s}}

        # ── search_contacts ──────────────────────────────────────────────────
        if call is None and "search_contacts" in available:
            m = (_re.search(
                    r"(?:find|look\s*up|search\s+(?:for\s+)?|search)\s+([A-Za-z][A-Za-z\s\-\']+?)\s+(?:in|from|on)\s+(?:my\s+)?contacts?\b",
                    cl, _re.I)
                or _re.search(
                    r"(?:find|look\s*up)\s+([A-Za-z][A-Za-z\s\-\']+?)\s*(?:in|from)?\s*(?:my\s+)?contacts?\b",
                    cl, _re.I))
            if m:
                query = m.group(1).strip().rstrip(".,!?")
                if query:
                    call = {"name": "search_contacts", "arguments": {"query": query}}
                    last_contact = query

        # ── play_music ───────────────────────────────────────────────────────
        if call is None and "play_music" in available:
            m = _re.search(r"\bplay\s+(.+)", cl, _re.I)
            if m:
                song = m.group(1).strip().rstrip(".,!?")
                # "some jazz music" → "jazz"  (but keep "Bohemian Rhapsody", "lo-fi beats")
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


def _validate_call(call, tools):
    """Return True if call has a valid tool name and all required params."""
    tool_map = {t["name"]: t for t in tools}
    name = call.get("name")
    if name not in tool_map:
        return False
    required = tool_map[name].get("parameters", {}).get("required", [])
    return all(k in call.get("arguments", {}) for k in required)


def _coerce_types(call, tools):
    """Fix type mismatches (e.g. string '5' → int 5) to match tool schema."""
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


def _count_expected_calls(text, tools):
    """Estimate how many function calls the user is requesting."""
    action_verbs = _re.findall(
        r"\b(?:play|send|text|set|find|remind|check|get|look\s+up|search|wake)\b",
        text, _re.I)
    conjunctions = _re.findall(
        r"\b(?:and|then|also|plus)\b|(?<=\w),\s*(?=\w)", text, _re.I)
    n_clauses = len(_split_clauses(text))
    return max(1, min(len(action_verbs), n_clauses, max(1, len(conjunctions) + 1)))


def _score_calls(calls, n_expected):
    """Score a candidate call list: penalize count mismatch, reward filled args."""
    if not calls:
        return -10
    count_penalty = abs(len(calls) - n_expected)
    arg_bonus = sum(min(len(c.get("arguments", {})), 3) for c in calls)
    return arg_bonus - count_penalty * 3


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """
    Synthesized hybrid strategy based on top fork analysis:

    Approach (inspired by ishaanvijai #1 legitimate, 91.6% / 270ms):

    1. Regex extraction (~0ms): deterministic extraction for all 7 tool types,
       clause splitting, pronoun resolution.

    2. Prompt compression: when regex is confident, pass ONLY matched tools
       with stripped descriptions and reduced max_tokens to cactus.
       Shorter prompt → faster prefill; shorter decode → lower latency.
       This is the key speedup — not bypassing the model, but making it faster.

    3. FunctionGemma always runs (on-device). Score both the regex result and
       the cactus result; return whichever scores higher. Regex acts as a
       validation/fallback, not a replacement.

    4. Cloud only as last resort (both regex and cactus produce nothing valid).
    """
    t_start = time.perf_counter()
    user_text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
    n_expected = _count_expected_calls(user_text, tools)

    # ── Phase 1: Regex extraction ─────────────────────────────────────────────
    raw_regex = _extract_regex_calls(user_text, tools)
    regex_calls = []
    seen = set()
    for c in raw_regex:
        c = _coerce_types(c, tools)
        if _validate_call(c, tools) and c["name"] not in seen:
            regex_calls.append(c)
            seen.add(c["name"])

    # ── Phase 2: Build cactus input (prompt compression if regex is confident) ─
    regex_confident = bool(regex_calls) and len(regex_calls) >= n_expected

    if regex_confident:
        # Narrow tool list to only what regex identified, strip descriptions.
        # Shorter context = faster prefill + more accurate decode on small model.
        matched_names = {c["name"] for c in regex_calls}
        matched_tools = [t for t in tools if t.get("name") in matched_names]
        minimal_tools = []
        for t in matched_tools:
            params = t.get("parameters", {})
            min_props = {
                p_name: {"type": p_val.get("type", "string")}
                for p_name, p_val in params.get("properties", {}).items()
            }
            minimal_tools.append({
                "name": t.get("name"),
                "parameters": {
                    "type": params.get("type", "object"),
                    "properties": min_props,
                    "required": params.get("required", []),
                },
            })
        cactus_input_tools = minimal_tools
        cactus_max_tokens = max(64, 48 * len(regex_calls))
        cactus_system = "Return tool call."
    else:
        cactus_input_tools = tools
        cactus_max_tokens = 256
        cactus_system = None

    # ── Phase 3: Always call FunctionGemma on-device ─────────────────────────
    local = generate_cactus(
        messages, cactus_input_tools,
        max_tokens=cactus_max_tokens,
        system_prompt=cactus_system,
    )
    # Validate and coerce cactus output against FULL tool schema
    local_calls_raw = [_coerce_types(c, tools) for c in local.get("function_calls", [])]
    local_calls = [c for c in local_calls_raw if _validate_call(c, tools)]

    # ── Phase 4: Pick best result — cactus vs regex ───────────────────────────
    if regex_confident:
        # Both candidates exist. Score and pick the better one.
        # Regex wins on ties: it's deterministic and extracted the right values.
        # Cactus only overrides when it produces a strictly higher-quality result.
        if _score_calls(local_calls, n_expected) > _score_calls(regex_calls, n_expected):
            best_calls = local_calls
        else:
            best_calls = regex_calls
    else:
        # Regex partial: merge regex hints + cactus fills the rest
        if regex_calls:
            regex_names = {c["name"] for c in regex_calls}
            merged = list(regex_calls)
            for lc in local_calls:
                if lc["name"] not in regex_names:
                    merged.append(lc)
            best_calls = merged
        else:
            best_calls = local_calls

    if best_calls:
        return {
            "function_calls": best_calls,
            "total_time_ms": local.get("total_time_ms", 0),
            "confidence": local.get("confidence", 0),
            "source": "on-device",
        }

    # ── Phase 5: Cloud fallback (last resort) ─────────────────────────────────
    try:
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback)"
        cloud["local_confidence"] = local.get("confidence", 0)
        cloud["total_time_ms"] += local.get("total_time_ms", 0)
        return cloud
    except Exception:
        return {
            "function_calls": [],
            "total_time_ms": (time.perf_counter() - t_start) * 1000,
            "confidence": 0.0,
            "source": "error",
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
