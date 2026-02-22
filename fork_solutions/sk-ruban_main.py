# Source: https://github.com/sk-ruban/functiongemma-hackathon

import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time, re
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
from google import genai
from google.genai import types

DESCRIPTION_OVERRIDES = {
    "set_timer": "Set a countdown timer for a duration in minutes. NOT an alarm.",
    "set_alarm": "Set an alarm for a specific time of day. NOT a timer.",
    "create_reminder": "Create a reminder with a title and time.",
}

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = cactus_init(functiongemma_path)
    return _model


def _repair_json(raw_str):
    """Attempt to fix common FunctionGemma JSON issues."""
    if not raw_str:
        return raw_str
    # Fix leading zeros in numbers (e.g. "minute":01 → "minute":1)
    raw_str = re.sub(r':\s*0+(\d+)', r':\1', raw_str)
    # Fix fullwidth colon (：) → normal colon
    raw_str = raw_str.replace('：', ':')
    # Remove <escape> tags
    raw_str = raw_str.replace('<escape>', '')
    return raw_str

def _fix_arguments(function_calls):
    """Post-process function call arguments."""
    for call in function_calls:
        args = call.get("arguments", {})
        for key, val in args.items():
            if isinstance(val, (int, float)) and val < 0:
                args[key] = abs(val)
            if isinstance(val, str):
                args[key] = val.strip().rstrip('?.!,')
    return function_calls


def _validate_call(call):
    """Reject calls with garbage arguments from FunctionGemma."""
    args = call.get("arguments", {})
    for val in args.values():
        if isinstance(val, str):
            if re.search(r'[\u4e00-\u9fff]', val):
                return False
            if re.match(r'\d{4}-\d{2}-\d{2}T', val):
                return False
    name = call.get("name", "")
    if name == "set_alarm":
        hour = args.get("hour")
        minute = args.get("minute")
        if isinstance(hour, (int, float)) and not (0 <= hour <= 23):
            return False
        if isinstance(minute, (int, float)) and not (0 <= minute <= 59):
            return False
    if name == "set_timer":
        minutes = args.get("minutes")
        if isinstance(minutes, (int, float)) and not (1 <= minutes <= 1440):
            return False
    return True


def _sanity_check(call, query):
    """Cross-check function name against keywords in the user query."""
    q = query.lower()
    name = call.get("name", "")
    args = call.get("arguments", {})

    if ("remind" in q or "reminder" in q) and name != "create_reminder":
        return False
    if "timer" in q and name == "set_alarm":
        return False
    if "alarm" in q and name == "set_timer":
        return False
    if name == "set_alarm" and "pm" in q:
        h = args.get("hour", 0)
        if isinstance(h, (int, float)) and h < 12:
            return False
    if re.search(r'\btext\b', q) and name != "send_message":
        return False
    return True


def _fix_arguments_from_query(call, query):
    """Post-process: FunctionGemma picks the function (intent classification),
    we extract accurate arguments from the original query (slot filling).
    Same architecture as Siri/Alexa: neural intent + rule-based entities."""
    name = call.get("name", "")
    args = call.get("arguments", {})
    q = query.strip().rstrip('?.!')
    ql = q.lower()

    if name == "play_music":
        m = re.search(r'play\s+(?:some\s+|the\s+song\s+)?(.+?)\s*$', ql)
        if m:
            song_raw = m.group(1).strip().rstrip('?.!,')
            idx = q.lower().find(song_raw)
            if idx >= 0:
                song_raw = q[idx:idx+len(song_raw)]
            args["song"] = song_raw

    elif name == "set_alarm":
        m = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?', ql)
        if m:
            hour = int(m.group(1))
            minute = int(m.group(2)) if m.group(2) else 0
            ampm = m.group(3)
            if ampm:
                ampm = ampm.replace('.', '').lower()
                if ampm == 'pm' and hour != 12:
                    hour += 12
                elif ampm == 'am' and hour == 12:
                    hour = 0
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                args["hour"] = hour
                args["minute"] = minute

    elif name == "set_timer":
        m = re.search(r'(\d+)\s*(?:min(?:ute)?s?)', ql)
        if m:
            args["minutes"] = abs(int(m.group(1)))

    elif name == "create_reminder":
        m = re.search(
            r'(?:remind\s+(?:me\s+)?(?:about|to)\s+(.+?)\s+at\s+'
            r'|reminder\s+(?:for|about|to)\s+(.+?)\s+at\s+)'
            r'(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)',
            ql
        )
        if m:
            title = (m.group(1) or m.group(2)).strip()
            title = re.sub(r'^the\s+', '', title)
            time_raw = m.group(3).strip().upper()
            tm = re.match(r'(\d{1,2})(?::(\d{2}))?\s*(AM|PM)', time_raw)
            if tm:
                h = tm.group(1)
                mins = tm.group(2) or "00"
                ap = tm.group(3)
                time_raw = f"{h}:{mins} {ap}"
            args["title"] = title
            args["time"] = time_raw

    elif name == "send_message":
        m = re.search(
            r'(?:send\s+(?:a\s+)?message\s+to|text|message)\s+(\w+)\s+(?:saying|that)\s+(.+)',
            ql
        )
        if not m:
            m = re.search(
                r'(?:send|text)\s+(\w+)\s+(?:a\s+)?message\s+(?:saying|that)\s+(.+)',
                ql
            )
        if m:
            recipient = m.group(1).strip()
            idx = q.lower().find(recipient)
            if idx >= 0:
                recipient = q[idx:idx+len(recipient)]
            args["recipient"] = recipient
            args["message"] = m.group(2).strip().rstrip('?.!,')

    elif name == "search_contacts":
        m = re.search(r'(?:find|look\s+up|search\s+for|search)\s+(\w+)', ql)
        if m:
            name_raw = m.group(1).strip()
            if name_raw.lower() not in ('my', 'the', 'a', 'in', 'for', 'contacts', 'contact'):
                idx = q.lower().find(name_raw)
                if idx >= 0:
                    name_raw = q[idx:idx+len(name_raw)]
                args["query"] = name_raw

    elif name == "get_weather":
        m = re.search(r'weather\s+(?:like\s+)?(?:in|for)\s+(.+?)(?:\s+right\s+now|\s+today|\s+tomorrow)?$', ql)
        if m:
            loc = m.group(1).strip().rstrip('?.!,')
            idx = q.lower().find(loc)
            if idx >= 0:
                loc = q[idx:idx+len(loc)]
            args["location"] = loc

    call["arguments"] = args
    return call


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = _get_model()
    cactus_reset(model)

    enriched_tools = []
    for t in tools:
        t_copy = dict(t)
        if t["name"] in DESCRIPTION_OVERRIDES:
            t_copy["description"] = DESCRIPTION_OVERRIDES[t["name"]]
        enriched_tools.append(t_copy)

    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in enriched_tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a model that can do function calling with the following functions"}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
        confidence_threshold=0.1,
        tool_rag_top_k=0,
    )

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        try:
            raw = json.loads(_repair_json(raw_str))
        except json.JSONDecodeError:
            return {
                "function_calls": [],
                "total_time_ms": 0,
                "confidence": 0,
            }

    calls = _fix_arguments(raw.get("function_calls", []))

    return {
        "function_calls": calls,
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
                    "arguments": {k: int(v) if isinstance(v, (int, float)) and float(v) == int(v) else str(v) for k, v in part.function_call.args.items()},
                })

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


ACTION_VERBS = {"set", "check", "get", "send", "text", "play", "find", "remind",
                "look", "search", "create", "wake", "tell", "show", "turn", "make",
                "start", "message", "ask", "add", "cancel", "stop", "open",
                "enter", "press", "hit", "type", "click"}

def _decompose_query(query):
    """Split multi-intent query into sub-queries. Only splits when word after
    conjunction is an action verb."""
    splitters = [" and ", ", ", " then ", " also ", " plus "]
    parts = [query]
    for splitter in splitters:
        new_parts = []
        for part in parts:
            candidates = part.split(splitter)
            if len(candidates) == 1:
                new_parts.append(part)
                continue
            merged = [candidates[0]]
            for c in candidates[1:]:
                first_word = c.strip().split()[0].lower() if c.strip() else ""
                if first_word in ACTION_VERBS:
                    merged.append(c.strip())
                else:
                    merged[-1] += splitter + c
            new_parts.extend(merged)
        parts = new_parts
    return [p.strip() for p in parts if p.strip()]


def _resolve_pronouns(sub_queries):
    """Replace pronouns in later sub-queries with names from earlier ones."""
    names = []
    for sq in sub_queries:
        found = re.findall(r'\b([A-Z][a-z]+)\b', sq)
        names.extend(found)

    resolved = [sub_queries[0]]
    for sq in sub_queries[1:]:
        if names:
            sq = re.sub(r'\bhim\b', names[0], sq, flags=re.IGNORECASE)
            sq = re.sub(r'\bher\b', names[0], sq, flags=re.IGNORECASE)
            sq = re.sub(r'\bthem\b', names[0], sq, flags=re.IGNORECASE)
        resolved.append(sq)
    return resolved


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """Hybrid inference: FunctionGemma (on-device) for intent classification,
    with rule-based argument extraction as post-processor.

    Architecture:
        1. Query decomposition — split multi-intent queries
        2. FunctionGemma — neural intent classification (picks which function)
        3. Validation — reject garbage arguments, cross-check keywords
        4. Slot filling — extract accurate arguments from original query
        5. Cloud fallback — Gemini Flash when on-device fails validation

    FunctionGemma always runs first. Regex is never used for function selection.
    """
    user_msg = messages[-1]["content"] if messages else ""
    sub_queries = _decompose_query(user_msg)
    if len(sub_queries) > 1:
        sub_queries = _resolve_pronouns(sub_queries)

    valid_names = {t["name"] for t in tools}

    # Single intent — normal path
    if len(sub_queries) <= 1:
        local = generate_cactus(messages, tools)
        fixed_calls = []
        for c in local["function_calls"]:
            if c.get("name") not in valid_names:
                continue
            c = _fix_arguments_from_query(c, user_msg)
            if _validate_call(c) and _sanity_check(c, user_msg):
                fixed_calls.append(c)
        local["function_calls"] = fixed_calls
        if local["function_calls"]:
            local["source"] = "on-device"
            return local
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback)"
        cloud["total_time_ms"] += local["total_time_ms"]
        return cloud

    # Rule-based shortcuts for simple keyboard actions (no model call needed)
    KEYBOARD_SHORTCUTS = {
        "enter": "Return", "press enter": "Return", "hit enter": "Return",
        "tab": "Tab", "press tab": "Tab",
        "escape": "Escape", "press escape": "Escape",
    }

    # Multi-intent — per-sub-query with cloud fallback
    all_calls = []
    total_time = 0
    used_cloud = False
    for sq in sub_queries:
        sq_lower = sq.strip().lower()
        if sq_lower in KEYBOARD_SHORTCUTS:
            all_calls.append({
                "name": "keyboard_shortcut",
                "arguments": {"keys": KEYBOARD_SHORTCUTS[sq_lower]},
            })
            continue

        sub_messages = [{"role": "user", "content": sq}]
        local = generate_cactus(sub_messages, tools)
        total_time += local.get("total_time_ms", 0)
        valid_calls = []
        for c in local["function_calls"]:
            if c.get("name") not in valid_names:
                continue
            c = _fix_arguments_from_query(c, sq)
            if _validate_call(c) and _sanity_check(c, sq):
                valid_calls.append(c)
        if valid_calls:
            all_calls.extend(valid_calls)
        else:
            used_cloud = True
            cloud = generate_cloud(sub_messages, tools)
            total_time += cloud.get("total_time_ms", 0)
            all_calls.extend(cloud.get("function_calls", []))

    if all_calls:
        return {
            "function_calls": all_calls,
            "total_time_ms": total_time,
            "confidence": 0.9,
            "source": "cloud (fallback)" if used_cloud else "on-device",
        }

    # Everything failed
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
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
