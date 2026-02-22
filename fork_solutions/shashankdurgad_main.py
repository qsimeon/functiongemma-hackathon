# Source: https://github.com/shashankdurgad/functiongemma-hackathon
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time, re
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types


# ================================================================
#  Routing utilities (inlined for single-file submission)
# ================================================================

def decompose_query(content):
    parts = re.split(r',?\s+and\s+|,?\s+then\s+|,\s+(?=[a-z])', content, flags=re.IGNORECASE)
    return [p.strip().rstrip('.').strip() for p in parts if p.strip() and len(p.strip()) > 5]

def validate_calls(calls, tools):
    tool_names = {t["name"] for t in tools}
    if not calls:
        return False, ["no function calls produced"]
    for call in calls:
        if call.get("name") not in tool_names:
            return False, [f"invalid tool name: {call.get('name')}"]
    for call in calls:
        tool = next((t for t in tools if t["name"] == call.get("name")), None)
        if tool:
            missing = [r for r in tool["parameters"].get("required", []) if r not in call.get("arguments", {})]
            if missing:
                return False, [f"missing required args for {call['name']}: {missing}"]
    return True, []

def calls_signature(calls):
    if not calls:
        return "EMPTY"
    normalized = []
    for c in sorted(calls, key=lambda x: x.get("name", "")):
        normalized.append((c["name"], json.dumps(c.get("arguments", {}), sort_keys=True)))
    return str(normalized)

def ensemble_vote(results):
    groups = {}
    for r in results:
        sig = calls_signature(r.get("function_calls", []))
        groups.setdefault(sig, []).append(r)
    best_group = max(groups.values(), key=len)
    agreement = len(best_group) / len(results)
    best = max(best_group, key=lambda r: r.get("confidence", 0))
    best["total_time_ms"] = max(r.get("total_time_ms", 0) for r in results)
    return best, agreement

def boost_confidence(confidence, agreement):
    if agreement >= 1.0:
        return min(1.0, confidence * 1.3)
    elif agreement >= 2 / 3:
        return min(1.0, confidence * 1.1)
    return confidence * 0.7

def get_ensemble_size(num_tools, has_multiple_actions):
    if num_tools == 1 and not has_multiple_actions:
        return 1
    if num_tools <= 2 and not has_multiple_actions:
        return 1  # 2 tools is still simple enough for single inference
    return 5 if has_multiple_actions else 3

def get_dynamic_threshold(num_tools, has_multiple_actions):
    if num_tools == 1 and not has_multiple_actions:
        return 0.15  # Trivial: 1 tool, just pick it
    if num_tools <= 2 and not has_multiple_actions:
        return 0.25
    if num_tools <= 3 and not has_multiple_actions:
        return 0.45
    if has_multiple_actions:
        return 0.55
    return 0.50

def tool_complexity_bonus(num_tools):
    """Fewer tools = simpler decision space = higher confidence bonus."""
    if num_tools == 1:
        return 1.4   # trivial: only one choice
    if num_tools == 2:
        return 1.25  # very simple
    if num_tools == 3:
        return 1.1   # moderate
    if num_tools == 4:
        return 1.0   # no bonus
    return 0.95      # 5+ tools: slight penalty

def dedup_calls(calls):
    seen, unique = set(), []
    for c in calls:
        key = (c["name"], json.dumps(c.get("arguments", {}), sort_keys=True))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique

def heuristic_tool_match(content, tools):
    content_lower = content.lower()
    scored = []
    for tool in tools:
        score = sum(3 for w in tool["name"].replace("_", " ").split() if w in content_lower)
        score += sum(1 for w in tool.get("description", "").lower().split() if len(w) > 3 and w in content_lower)
        if score >= 3:
            scored.append((score, tool))
    scored.sort(key=lambda x: -x[0])
    matched = []
    for _, tool in scored:
        args = {}
        props = tool["parameters"].get("properties", {})
        required = tool["parameters"].get("required", [])
        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "string")
            pdesc = pinfo.get("description", "").lower()
            tags = {pname} | set(pdesc.split())
            if ptype == "integer":
                if "hour" in tags:
                    m = re.search(r'(\d{1,2})(?::\d{2})?\s*(?:am|pm)?', content_lower)
                    if m: args[pname] = int(m.group(1))
                elif "minute" in tags:
                    m = re.search(r'\d+:(\d+)', content_lower)
                    args[pname] = int(m.group(1)) if m else 0
                else:
                    nums = re.findall(r'\b(\d+)\b', content_lower)
                    if nums: args[pname] = int(nums[0])
            elif ptype == "string":
                if tags & {"location", "city"}:
                    m = re.search(r'\bin\s+([a-z][a-z\s]*?)(?:\s*[.?,!]|\s+and\s+|$)', content_lower)
                    if m: args[pname] = m.group(1).strip().title()
                elif tags & {"recipient", "person"}:
                    m = re.search(r'\bto\s+([a-z]+)', content_lower)
                    if m: args[pname] = m.group(1).title()
                elif "message" in tags:
                    m = re.search(r'\bsaying\s+(.+?)(?:\s+and\s+|[.,!]|$)', content_lower)
                    if m: args[pname] = m.group(1).strip()
                elif tags & {"song", "playlist"}:
                    m = re.search(r'\bplay\s+(?:some\s+)?(.+?)(?:\s+and\s+|[.,!]|$)', content_lower)
                    if m: args[pname] = m.group(1).strip()
                elif "title" in tags:
                    m = re.search(r'\babout\s+(.+?)(?:\s+at\s+|\s+and\s+|[.,!]|$)', content_lower)
                    if not m:
                        m = re.search(r'\bremind\s+(?:me\s+)?(?:to\s+)?(.+?)(?:\s+at\s+|\s+and\s+|[.,!]|$)', content_lower)
                    if m: args[pname] = m.group(1).strip()
                elif "time" in pname and ("time" in pdesc or "when" in pdesc):
                    m = re.search(r'(\d{1,2}:\d{2}\s*(?:am|pm))', content_lower, re.IGNORECASE)
                    if m: args[pname] = m.group(1).strip().upper()
                elif tags & {"query", "search"}:
                    m = re.search(r'\b(?:find|look\s+up|search\s+for?)\s+([a-z]+)', content_lower)
                    if m: args[pname] = m.group(1).title()
        if all(r in args for r in required):
            matched.append({"name": tool["name"], "arguments": args})
    return matched


# ================================================================
#  On-Device Backend
# ================================================================

def generate_cactus(messages, tools):
    model = cactus_init(functiongemma_path)
    cactus_tools = [{"type": "function", "function": t} for t in tools]
    tool_names_str = ", ".join(t["name"] for t in tools)
    system_prompt = (
        f"You are a function calling assistant. You MUST respond with a function call "
        f"using one of these tools: [{tool_names_str}]. "
        f"Output ONLY valid JSON function calls. Do not output conversational text."
    )
    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": system_prompt}] + messages,
        tools=cactus_tools, force_tools=True, max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )
    cactus_destroy(model)
    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {"function_calls": [], "total_time_ms": 0, "confidence": 0}
    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


# ================================================================
#  Cloud Backend
# ================================================================

def generate_cloud(messages, tools):
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"], description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            ) for t in tools
        ])
    ]
    contents = [m["content"] for m in messages if m["role"] == "user"]
    start = time.time()
    resp = client.models.generate_content(
        model="gemini-2.0-flash", contents=contents,
        config=types.GenerateContentConfig(tools=gemini_tools),
    )
    ms = (time.time() - start) * 1000
    calls = []
    for cand in resp.candidates:
        for part in cand.content.parts:
            if part.function_call:
                calls.append({"name": part.function_call.name, "arguments": dict(part.function_call.args)})
    return {"function_calls": calls, "total_time_ms": ms}


# ================================================================
#  Cloud Decomposition
# ================================================================

def cloud_decompose_query(content, tools):
    try:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        prompt = (
            f"Split this user request into separate, independent sub-tasks. "
            f"Each sub-task should map to exactly one tool.\n"
            f"Available tools: {[t['name'] for t in tools]}\n"
            f"User request: \"{content}\"\n"
            f"Return ONLY a JSON array of strings, no markdown."
        )
        text = client.models.generate_content(model="gemini-2.0-flash", contents=[prompt]).text.strip()
        text = re.sub(r'^```\w*\n?', '', text)
        text = re.sub(r'\n?```$', '', text).strip()
        subs = json.loads(text)
        if isinstance(subs, list) and len(subs) >= 2:
            print(f"  Cloud decomposition: {subs}")
            return subs
    except Exception as e:
        print(f"  Cloud decomposition failed ({e}), using regex fallback")
    return decompose_query(content)


# ================================================================
#  Hybrid Generator
# ================================================================

def _run_ensemble(local_fn, messages, tools, size):
    start = time.time()
    results = [local_fn(messages, tools) for _ in range(size)]
    wall_ms = (time.time() - start) * 1000
    best, agreement = ensemble_vote(results)
    return best, agreement, wall_ms


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    content = messages[-1]["content"].lower() if messages else ""
    num_tools = len(tools)

    conjunction_count = content.count(" and ") + content.count(", and ") + content.count(" then ")
    # Also count comma-separated clauses (e.g. "set alarm, play music")
    comma_clauses = len(re.findall(r',\s+(?=[a-z])', content))
    conjunction_count = max(conjunction_count, comma_clauses)

    # Count action verbs that map to available tools for better expected count
    _ACTION_VERBS = {
        "get_weather": ["weather", "forecast", "temperature"],
        "set_alarm": ["alarm", "wake me"],
        "send_message": ["message", "text ", "tell ", "saying"],
        "create_reminder": ["remind", "reminder"],
        "search_contacts": ["find ", "look up", "search", "contacts"],
        "play_music": ["play ", "music", "song"],
        "set_timer": ["timer", "countdown"],
    }
    tool_name_set = {t["name"] for t in tools}
    action_hits = 0
    for tool_name, verbs in _ACTION_VERBS.items():
        if tool_name in tool_name_set:
            if any(v in content for v in verbs):
                action_hits += 1
    # Use action verb count as a secondary signal
    if action_hits >= 2:
        conjunction_count = max(conjunction_count, action_hits - 1)

    has_multi = conjunction_count > 0
    expected = max(1 + conjunction_count, action_hits) if action_hits >= 2 else 1 + conjunction_count

    threshold = get_dynamic_threshold(num_tools, has_multi)
    ens_size = get_ensemble_size(num_tools, has_multi)

    print(f"\n--- Hybrid Routing ---")
    print(f"  Query: {content[:80]}")
    print(f"  Tools: {num_tools} | Multi-action: {has_multi} | Threshold: {threshold} | Ensemble: {ens_size}")

    best, agreement, wall_ms = _run_ensemble(generate_cactus, messages, tools, ens_size)
    calls = best.get("function_calls", [])
    raw_conf = best.get('confidence', 0)
    conf = boost_confidence(raw_conf, agreement)
    # Apply tool complexity bonus: fewer tools = trust local more
    conf = min(1.0, conf * tool_complexity_bonus(num_tools))

    print(f"  Ensemble: agreement={agreement:.0%} | wall={wall_ms:.0f}ms | conf={raw_conf:.4f}->{conf:.4f} (tools_bonus={tool_complexity_bonus(num_tools):.2f})")
    print(f"  Calls ({len(calls)}): {[c.get('name','?') for c in calls]}")

    valid, failures = validate_calls(calls, tools)
    if valid and has_multi and len(calls) < expected:
        failures.append(f"expected {expected} calls, got {len(calls)}")
        conf *= 0.7

    print(f"  Validation: {'PASS' if valid else 'FAIL'}", end="")
    if failures:
        print(f" [{', '.join(failures)}]")
    else:
        print()

    use_local = valid and conf >= threshold

    # Semantic sanity check: do the returned calls match the action verbs in the query?
    if use_local and calls and not has_multi:
        _VERB_TO_TOOL = {
            "weather": "get_weather", "forecast": "get_weather",
            "alarm": "set_alarm", "wake": "set_alarm",
            "message": "send_message", "text ": "send_message", "saying": "send_message",
            "remind": "create_reminder", "reminder": "create_reminder",
            "find ": "search_contacts", "look up": "search_contacts", "contacts": "search_contacts",
            "play ": "play_music", "music": "play_music", "song": "play_music",
            "timer": "set_timer", "countdown": "set_timer",
        }
        tool_name_set = {t["name"] for t in tools}
        expected_tools = set()
        for verb, tname in _VERB_TO_TOOL.items():
            if verb in content and tname in tool_name_set:
                expected_tools.add(tname)
        returned_tools = {c["name"] for c in calls}
        if expected_tools and not (returned_tools & expected_tools):
            print(f"  Sanity check FAILED: returned {returned_tools} but expected {expected_tools}")
            use_local = False
            calls = []

    if not use_local and not calls:
        print(f"  Trying HEURISTIC...")
        h_calls = heuristic_tool_match(content, tools)
        if h_calls:
            h_valid, h_fail = validate_calls(h_calls, tools)
            print(f"    Heuristic: {[c['name'] for c in h_calls]} valid={h_valid}")
            if h_valid:
                calls, conf = h_calls, 0.65
                best["function_calls"], best["confidence"] = calls, conf
                valid, use_local = True, conf >= threshold
                print(f"    Accepted! conf=0.65 >= {threshold}")

    if not use_local and has_multi:
        print(f"  Attempting DECOMPOSITION...")
        subs = cloud_decompose_query(content, tools)
        print(f"  Sub-queries ({len(subs)}): {subs}")
        dec_calls, dec_time, dec_min_conf, ok = [], 0, 1.0, True
        for sq in subs:
            sb, sa, sw = _run_ensemble(generate_cactus, [{"role": "user", "content": sq}], tools, 3)
            sc = boost_confidence(sb.get("confidence", 0), sa)
            sb["confidence"] = sc
            s_calls = sb.get("function_calls", [])
            dec_time += sb.get("total_time_ms", 0)
            dec_min_conf = min(dec_min_conf, sc)
            s_valid, s_fail = validate_calls(s_calls, tools)
            print(f"    Sub '{sq[:50]}': conf={sc:.4f} agr={sa:.0%} calls={[c['name'] for c in s_calls] if s_calls else []} valid={s_valid}")
            if s_valid and sc >= 0.2:
                dec_calls.extend(s_calls)
            else:
                ok = False
                break
        dec_calls = dedup_calls(dec_calls)
        if ok and len(dec_calls) >= len(subs):
            print(f"  >>> ON-DEVICE (decomposed, {len(dec_calls)} calls)")
            print(f"--- End Routing ---")
            return {"function_calls": dec_calls, "total_time_ms": wall_ms + dec_time,
                    "confidence": dec_min_conf, "source": "on-device"}
        print(f"  Decomposition failed")

    label = "ON-DEVICE" if use_local else "CLOUD FALLBACK"
    print(f"  >>> {label}")
    print(f"--- End Routing ---")

    if use_local:
        best["source"] = "on-device"
        best["total_time_ms"] = wall_ms
        return best

    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = conf
    cloud["total_time_ms"] += wall_ms
    return cloud


if __name__ == "__main__":
    tools = [{
        "name": "get_weather", "description": "Get current weather for a location",
        "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City name"}}, "required": ["location"]},
    }]
    messages = [{"role": "user", "content": "What is the weather in San Francisco?"}]
    result = generate_hybrid(messages, tools)
    print(f"\nSource: {result.get('source')} | Time: {result['total_time_ms']:.0f}ms")
    for c in result["function_calls"]:
        print(f"  {c['name']}({json.dumps(c['arguments'])})")
