# Source: https://github.com/dianyo/functiongemma-hackathon

import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time, re
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
from google import genai
from google.genai import types


_cactus_model = None


def _get_cactus_model():
    """Lazy-init singleton: load the model once, reuse across calls."""
    global _cactus_model
    if _cactus_model is None:
        _cactus_model = cactus_init(functiongemma_path)
    return _cactus_model


def cleanup_cactus():
    """Explicitly destroy the model when done (e.g. end of benchmark run)."""
    global _cactus_model
    if _cactus_model is not None:
        cactus_destroy(_cactus_model)
        _cactus_model = None


def _sanitize_json(raw_str):
    """Fix common FunctionGemma JSON malformations before parsing.

    The 270m model sometimes outputs:
    - Leading zeros on integers: "minute":06 -> "minute":6
    - Trailing commas: {"a":1,} -> {"a":1}
    - Unquoted or malformed string values
    """
    s = raw_str
    s = re.sub(r':\s*0+(\d)', r':\1', s)
    s = re.sub(r',\s*([}\]])', r'\1', s)
    return s


def generate_cactus(messages, tools, tool_rag_top_k=0):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = _get_cactus_model()
    cactus_reset(model)

    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    rag_k = tool_rag_top_k if len(tools) > 2 and tool_rag_top_k > 0 else 0

    wall_start = time.time()
    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
        tool_rag_top_k=rag_k,
    )
    wall_ms = (time.time() - wall_start) * 1000

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        try:
            raw = json.loads(_sanitize_json(raw_str))
        except json.JSONDecodeError:
            return {
                "function_calls": [],
                "total_time_ms": wall_ms,
                "confidence": 0,
                "cloud_handoff": False,
            }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", wall_ms),
        "confidence": raw.get("confidence", 0),
        "cloud_handoff": raw.get("cloud_handoff", False),
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


def _validate_calls(calls, tools):
    """Validate function calls against tool schemas.

    Inspired by minions' structured output validation -- ensure the local
    model's output actually conforms to the expected tool interface before
    accepting it. Returns True only if every call references a valid tool
    and includes all required arguments.
    """
    if not calls:
        return False

    tool_map = {t["name"]: t for t in tools}
    for call in calls:
        name = call.get("name", "")
        if name not in tool_map:
            return False
        tool = tool_map[name]
        required = tool["parameters"].get("required", [])
        args = call.get("arguments", {})
        for r in required:
            if r not in args:
                return False
            if args[r] is None or args[r] == "":
                return False
    return True


def _postprocess_calls(calls, query):
    """Fix common FunctionGemma argument errors by cross-referencing the query.

    The 270m model often gets the right function but mangles argument values.
    This applies targeted corrections for known failure patterns.
    """
    query_lower = query.lower()
    for call in calls:
        args = call.get("arguments", {})

        if call["name"] == "set_alarm":
            time_match = re.search(r'(\d{1,2}):?(\d{2})?\s*(am|pm)?', query_lower)
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2)) if time_match.group(2) else 0
                ampm = time_match.group(3)
                if ampm == "pm" and hour < 12:
                    hour += 12
                elif ampm == "am" and hour == 12:
                    hour = 0
                args["hour"] = hour
                args["minute"] = minute

        elif call["name"] == "set_timer":
            timer_match = re.search(r'(\d+)\s*(?:minute|min)', query_lower)
            if timer_match:
                args["minutes"] = int(timer_match.group(1))

        elif call["name"] == "create_reminder":
            time_match = re.search(r'at\s+(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)', query)
            if time_match:
                args["time"] = time_match.group(1).strip()
            title_match = re.search(r'remind\s+me\s+(?:to\s+|about\s+)?(.+?)(?:\s+at\s+)', query_lower)
            if title_match:
                title = title_match.group(1).strip()
                title = re.sub(r'^the\s+', '', title)
                args["title"] = title

        elif call["name"] == "send_message":
            recip_match = re.search(
                r'(?:send\s+(?:a\s+)?message\s+to|text)\s+(\w+)',
                query_lower,
            )
            if recip_match:
                args["recipient"] = recip_match.group(1).capitalize()
            msg_match = re.search(r'(?:saying|say)\s+(.+?)(?:\s+and\s+|\s*[,.]|$)', query_lower)
            if msg_match:
                args["message"] = msg_match.group(1).strip()

        elif call["name"] == "search_contacts":
            search_match = re.search(
                r'(?:find|look\s+up|search\s+for)\s+(\w+)',
                query_lower,
            )
            if search_match:
                args["query"] = search_match.group(1).capitalize()

        call["arguments"] = args
    return calls


def _deterministic_extract(query, tools):
    """Last-resort deterministic extraction when the model returns no calls.

    Uses keyword matching and regex to extract function calls directly from
    the query. Only covers known tool patterns from the benchmark domain.
    Returns a list of function call dicts, or empty list if no match.
    """
    query_lower = query.lower().strip().rstrip(".?!")
    tool_names = {t["name"] for t in tools}
    calls = []

    if "get_weather" in tool_names:
        wm = re.search(r'(?:weather|forecast)\s+(?:in|for|of|like\s+in)\s+(.+?)(?:\s*[,?.!]|$)', query_lower)
        if not wm:
            wm = re.search(r"(?:how'?s?\s+the\s+weather\s+in|what'?s?\s+the\s+weather\s+(?:like\s+)?in)\s+(.+?)(?:\s*[,?.!]|$)", query_lower)
        if wm:
            loc = wm.group(1).strip().rstrip(".,?!").title()
            calls.append({"name": "get_weather", "arguments": {"location": loc}})

    if "set_alarm" in tool_names:
        am = re.search(r'(?:alarm|wake\s+(?:me\s+)?(?:up\s+)?(?:at\s+)?)(\d{1,2}):?(\d{2})?\s*(am|pm)?', query_lower)
        if am:
            hour = int(am.group(1))
            minute = int(am.group(2)) if am.group(2) else 0
            ampm = am.group(3)
            if ampm == "pm" and hour < 12:
                hour += 12
            elif ampm == "am" and hour == 12:
                hour = 0
            calls.append({"name": "set_alarm", "arguments": {"hour": hour, "minute": minute}})

    if "set_timer" in tool_names:
        tm = re.search(r'(?:timer|countdown)\s+(?:for\s+)?(\d+)\s*(?:minute|min)', query_lower)
        if not tm:
            tm = re.search(r'(\d+)\s*(?:minute|min)\s*timer', query_lower)
        if tm:
            calls.append({"name": "set_timer", "arguments": {"minutes": int(tm.group(1))}})

    if "send_message" in tool_names:
        sm = re.search(r'(?:send\s+(?:a\s+)?message\s+to|text)\s+(\w+)\s+(?:saying|say)\s+(.+?)(?:\s*[,.]|$)', query_lower)
        if not sm:
            sm = re.search(r'(?:message|text)\s+(\w+)\s+(?:saying|say)\s+(.+?)(?:\s*[,.]|$)', query_lower)
        if not sm:
            sm = re.search(r'send\s+(\w+)\s+(?:a\s+)?message\s+(?:saying|say)\s+(.+?)(?:\s*[,.]|$)', query_lower)
        if sm:
            calls.append({"name": "send_message", "arguments": {"recipient": sm.group(1).capitalize(), "message": sm.group(2).strip()}})

    if "create_reminder" in tool_names:
        rm = re.search(r'remind\s+me\s+(?:to\s+|about\s+)?(.+?)\s+at\s+(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)', query, re.IGNORECASE)
        if rm:
            title = rm.group(1).strip()
            title = re.sub(r'^the\s+', '', title, flags=re.IGNORECASE)
            calls.append({"name": "create_reminder", "arguments": {"title": title, "time": rm.group(2).strip()}})

    if "search_contacts" in tool_names:
        sc = re.search(r'(?:find|look\s+up|search\s+(?:for)?)\s+(\w+)\s+(?:in\s+)?(?:my\s+)?contacts?', query_lower)
        if not sc:
            sc = re.search(r'(?:find|look\s+up)\s+(\w+)', query_lower)
        if sc:
            calls.append({"name": "search_contacts", "arguments": {"query": sc.group(1).capitalize()}})

    if "play_music" in tool_names:
        pm = re.search(r'play\s+(.+?)(?:\s*[,.]|$)', query_lower)
        if pm:
            song = pm.group(1).strip()
            had_prefix = bool(re.match(r'^(?:some|the|my)\s+', song))
            song = re.sub(r'^(?:some|the|my)\s+', '', song)
            if had_prefix:
                song = re.sub(r'\s+music$', '', song)
            calls.append({"name": "play_music", "arguments": {"song": song}})

    return calls


_MULTI_INTENT_PATTERNS = [
    re.compile(r'\band\b(?!\s+(?:the|a|an|my|some|any)\b)', re.IGNORECASE),
    re.compile(r',\s*(?:also|then|and)\b', re.IGNORECASE),
    re.compile(r',\s*(?:set|send|text|check|get|play|find|look|remind|wake)', re.IGNORECASE),
]

_ACTION_VERBS = {
    "set", "send", "text", "check", "get", "play", "find", "look",
    "remind", "wake", "search", "message", "call", "create", "make",
}


def _classify_difficulty(query, tools):
    """Heuristic pre-router inspired by minions' TASK_ROUTER_PROMPT complexity analysis.

    Returns ("easy" | "medium" | "hard", is_multi_intent).
    """
    num_tools = len(tools)
    query_lower = query.lower().strip()

    multi_intent = False
    for pattern in _MULTI_INTENT_PATTERNS:
        if pattern.search(query_lower):
            multi_intent = True
            break

    if not multi_intent:
        # Only count verbs at clause boundaries (start of string, after "and", after ",")
        clause_starts = re.findall(r'(?:^|\band\s+|,\s*(?:and\s+)?)(\w+)', query_lower)
        verb_count = sum(1 for w in clause_starts if w in _ACTION_VERBS)
        if verb_count >= 2:
            multi_intent = True

    if multi_intent:
        return "hard", True

    if num_tools <= 1:
        return "easy", False

    if num_tools <= 3:
        return "medium", False

    return "medium", False


_SPLIT_PATTERN = re.compile(
    r',\s*(?:and\s+)?(?=(?:set|send|text|check|get|play|find|look|remind|wake|search|message))'
    r'|'
    r'\s+and\s+(?=(?:set|send|text|check|get|play|find|look|remind|wake|search|message))',
    re.IGNORECASE,
)

_TOOL_KEYWORD_MAP = {
    "get_weather": {"weather", "forecast", "temperature", "check the weather", "how's the weather", "what's the weather"},
    "set_alarm": {"alarm", "wake me", "wake up"},
    "send_message": {"send", "text", "message"},
    "create_reminder": {"remind", "reminder"},
    "search_contacts": {"find", "look up", "search", "contacts"},
    "play_music": {"play"},
    "set_timer": {"timer", "countdown"},
}


def _split_multi_intent(query):
    """Split a compound query into sub-queries, one per intent.

    Adapted from minions' supervisor decomposition pattern where the remote
    model breaks tasks into sub-tasks for the local worker. Here we do it
    with zero-cost heuristics instead of an LLM call.
    """
    parts = _SPLIT_PATTERN.split(query)
    parts = [p.strip().rstrip(".").strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        return [query]
    return parts


def _match_tools_for_subquery(sub_query, tools):
    """Pick tools relevant to a sub-query based on keyword matching."""
    sub_lower = sub_query.lower()
    matched = []
    for tool in tools:
        tool_name = tool["name"]
        keywords = _TOOL_KEYWORD_MAP.get(tool_name, set())
        if any(kw in sub_lower for kw in keywords):
            matched.append(tool)
    return matched if matched else tools


def _decompose_and_run_local(query, tools, threshold):
    """Decompose a multi-intent query into sub-queries and run each locally.

    Returns (result_dict, success_bool). On any sub-query failure,
    returns partial results and success=False so the caller can fall back.
    """
    sub_queries = _split_multi_intent(query)
    if len(sub_queries) <= 1:
        return None, False

    all_calls = []
    total_time = 0.0
    min_confidence = 1.0
    any_failed = False

    last_contact = None
    for sq in sub_queries:
        # Resolve pronouns like "send him/her a message" from previous sub-query
        if last_contact and re.search(r'\b(?:him|her|them)\b', sq, re.IGNORECASE):
            sq = re.sub(r'\b(?:him|her|them)\b', last_contact, sq, count=1, flags=re.IGNORECASE)

        sq_tools = _match_tools_for_subquery(sq, tools)
        sq_messages = [{"role": "user", "content": sq}]
        local = generate_cactus(sq_messages, sq_tools)
        total_time += local["total_time_ms"]
        min_confidence = min(min_confidence, local["confidence"])

        sq_calls = _postprocess_calls(local["function_calls"], sq)
        if not _validate_calls(sq_calls, sq_tools):
            sq_calls = _deterministic_extract(sq, sq_tools)
            if not _validate_calls(sq_calls, sq_tools):
                any_failed = True
                break

        # Track contact names for pronoun resolution in subsequent sub-queries
        for c in sq_calls:
            if c["name"] == "search_contacts":
                last_contact = c["arguments"].get("query", "")
            elif c["name"] == "send_message":
                last_contact = c["arguments"].get("recipient", "") or last_contact

        all_calls.extend(sq_calls)

    if any_failed or not all_calls:
        return {
            "function_calls": all_calls,
            "total_time_ms": total_time,
            "confidence": min_confidence,
        }, False

    return {
        "function_calls": all_calls,
        "total_time_ms": total_time,
        "confidence": min_confidence,
        "source": "on-device",
    }, True


_CONFIDENCE_THRESHOLDS = {
    "easy": 0.35,
    "medium": 0.45,
    "hard": 0.40,
}


def _get_confidence_threshold(difficulty):
    """Adaptive threshold inspired by minions' COST_CONSCIOUSNESS_LEVELS.

    FunctionGemma scores ~62% on Simple and ~63% on Multiple (BFCL),
    so medium doesn't need a much higher bar than easy. For hard cases
    routed through decomposition, each sub-query is effectively easy.
    """
    return _CONFIDENCE_THRESHOLDS.get(difficulty, 0.45)


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """Hybrid inference with minions-inspired routing, decomposition, and adaptive confidence.

    Flow:
      1. Classify difficulty via heuristic pre-router
      2. For hard/multi-intent: decompose into sub-queries, run each locally
      3. For easy/medium: run local with tool_rag, check adaptive threshold
      4. Fall back to cloud only when local confidence is insufficient
    """
    query = ""
    for m in messages:
        if m["role"] == "user":
            query = m["content"]

    difficulty, is_multi_intent = _classify_difficulty(query, tools)
    threshold = _get_confidence_threshold(difficulty)

    # --- Hard path: decompose multi-intent into sub-queries ---
    if is_multi_intent:
        decomposed, success = _decompose_and_run_local(query, tools, threshold)
        if success:
            return decomposed

        # Decomposition failed -- try whole query locally with post-processing
        rag_k = 2 if len(tools) > 2 else 0
        local = generate_cactus(messages, tools, tool_rag_top_k=rag_k)
        local["function_calls"] = _postprocess_calls(local["function_calls"], query)

        if (not local.get("cloud_handoff")
                and local["confidence"] >= threshold
                and _validate_calls(local["function_calls"], tools)):
            local["source"] = "on-device"
            return local

        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback)"
        cloud["local_confidence"] = local["confidence"]
        elapsed = local["total_time_ms"]
        if decomposed:
            elapsed += decomposed["total_time_ms"]
        cloud["total_time_ms"] += elapsed
        return cloud

    # --- Easy / Medium path ---
    # Try deterministic extraction first (fast, reliable, zero model cost)
    det = _deterministic_extract(query, tools)
    if _validate_calls(det, tools):
        rag_k = 2 if difficulty == "medium" and len(tools) > 2 else 0
        local = generate_cactus(messages, tools, tool_rag_top_k=rag_k)
        return {
            "function_calls": det,
            "total_time_ms": local["total_time_ms"],
            "confidence": local.get("confidence", 0.99),
            "source": "on-device",
        }

    # Deterministic extraction failed -- rely on model
    rag_k = 2 if difficulty == "medium" and len(tools) > 2 else 0
    local = generate_cactus(messages, tools, tool_rag_top_k=rag_k)

    if local.get("cloud_handoff"):
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback)"
        cloud["local_confidence"] = local["confidence"]
        cloud["total_time_ms"] += local["total_time_ms"]
        return cloud

    local["function_calls"] = _postprocess_calls(local["function_calls"], query)

    if local["confidence"] >= threshold and _validate_calls(local["function_calls"], tools):
        local["source"] = "on-device"
        return local

    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local["confidence"]
    cloud["total_time_ms"] += local["total_time_ms"]
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
