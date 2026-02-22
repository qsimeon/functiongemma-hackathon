# Source: https://github.com/amoping1/functiongemma-hackathon

import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, re, time
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types

_CLOUD_FAILURE_STREAK = 0
_CLOUD_COOLDOWN_CASES = 0


def _parse_time_12h(text):
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap]m)\b", text, re.IGNORECASE)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2) or "0")
    ampm = m.group(3).lower()
    if ampm == "pm" and hour != 12:
        hour += 12
    if ampm == "am" and hour == 12:
        hour = 0
    return hour, minute


def _format_time_12h(hour, minute):
    ampm = "AM" if hour < 12 or hour == 24 else "PM"
    h12 = hour % 12
    if h12 == 0:
        h12 = 12
    return f"{h12}:{minute:02d} {ampm}"


def _clean_trailing_punct(text):
    return re.sub(r"[.?!]+$", "", text).strip()


def _normalize_function_calls(calls):
    normalized = []
    seen = set()

    for call in calls:
        name = call.get("name")
        args = dict(call.get("arguments", {}))

        for key, val in list(args.items()):
            if isinstance(val, str):
                args[key] = _clean_trailing_punct(val)

        key = (name, json.dumps(args, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        normalized.append({"name": name, "arguments": args})

    return normalized


def _required_fields_by_tool(tools):
    req = {}
    for t in tools:
        params = t.get("parameters", {})
        req[t.get("name")] = params.get("required", [])
    return req


def _properties_by_tool(tools):
    props = {}
    for t in tools:
        params = t.get("parameters", {})
        props[t.get("name")] = params.get("properties", {})
    return props


def _to_gemini_schema_type(json_type):
    mapping = {
        "object": "OBJECT",
        "array": "ARRAY",
        "string": "STRING",
        "integer": "INTEGER",
        "number": "NUMBER",
        "boolean": "BOOLEAN",
    }
    return mapping.get((json_type or "").lower(), "STRING")


def _coerce_value_to_type(value, spec):
    expected = spec.get("type")
    if expected == "integer":
        if isinstance(value, bool):
            return value, False
        if isinstance(value, int):
            return value, True
        if isinstance(value, float) and value.is_integer():
            return int(value), True
        if isinstance(value, str):
            m = re.search(r"-?\d+", value)
            if m:
                return int(m.group(0)), True
        return value, False
    if expected == "number":
        if isinstance(value, bool):
            return value, False
        if isinstance(value, (int, float)):
            return float(value), True
        if isinstance(value, str):
            try:
                return float(value.strip()), True
            except ValueError:
                return value, False
        return value, False
    if expected == "boolean":
        if isinstance(value, bool):
            return value, True
        if isinstance(value, str):
            low = value.strip().lower()
            if low in {"true", "yes", "1"}:
                return True, True
            if low in {"false", "no", "0"}:
                return False, True
        return value, False
    if expected == "string":
        if isinstance(value, str):
            return value, True
        return str(value), True
    if expected == "array":
        if isinstance(value, list):
            return value, True
        return [value], True
    return value, True


def _infer_default_from_description(description, expected_type):
    if not description:
        return None
    text = description.strip()
    patterns = [
        r"default(?:\s+value)?\s*(?:is|=|to|of)\s*['\"]?([^'\".;,\n]+)",
        r"if\s+not\s+specified[^.]*default(?:s)?\s*(?:to|is)\s*['\"]?([^'\".;,\n]+)",
    ]
    token = None
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            token = m.group(1).strip()
            break
    if token is None:
        return None

    if expected_type == "integer":
        if re.fullmatch(r"-?\d+", token):
            return int(token)
        return None
    if expected_type == "number":
        try:
            return float(token)
        except ValueError:
            return None
    if expected_type == "boolean":
        low = token.lower()
        if low in {"true", "yes", "1"}:
            return True
        if low in {"false", "no", "0"}:
            return False
        return None
    if expected_type == "array":
        return [token]
    return token


def _build_planner_system_prompt(tools, previous_calls=None, expected_count=None, mode="full", feedback=None):
    compact_tools = []
    for t in tools:
        params = t.get("parameters", {})
        props = params.get("properties", {})
        compact_props = {}
        for k, v in props.items():
            compact_props[k] = {"type": v.get("type")}
        compact_tools.append(
            {
                "name": t.get("name"),
                "description": (t.get("description", "") or "")[:120],
                "required": params.get("required", []),
                "properties": compact_props,
            }
        )
    schema = {
        "tools": compact_tools
    }
    prompt = (
        "You are a strict tool planner.\n"
        "Return only tool calls that satisfy the user request.\n"
        "Output only JSON-compatible tool calls.\n"
        "Use only the provided tools and argument names.\n"
        "Infer optional defaults when parameter descriptions define explicit defaults.\n"
        "If multiple operations are requested, return all of them.\n"
        f"Tool schema:\n{json.dumps(schema, ensure_ascii=True)}"
    )
    if mode == "plan_tools":
        prompt += (
            "\nStep objective: select which tools should be called."
            "\nPrefer correct tool names and call count over argument completeness."
            "\nArguments may be minimal placeholders."
        )
    elif mode == "fix":
        prompt += (
            "\nStep objective: fix schema and coverage issues in the previous draft."
            "\nKeep valid calls and repair only incorrect/incomplete ones."
        )
    if expected_count is not None:
        prompt += (
            f"\nTarget number of calls: {expected_count}."
            "\nTry to return exactly this many calls when supported by user intent."
        )
    if previous_calls is not None:
        prompt += (
            "\nCritique the previous draft and fix wrong tool names, missing calls, "
            f"or invalid arguments.\nPrevious draft:\n{json.dumps(previous_calls, ensure_ascii=True)}"
        )
    if feedback:
        prompt += f"\nValidation feedback:\n{json.dumps(feedback, ensure_ascii=True)}"
    return prompt


def _repair_calls_with_schema(calls, tools):
    """Coerce argument types and drop invalid calls using tool schemas."""
    repaired = []
    properties = _properties_by_tool(tools)
    required = _required_fields_by_tool(tools)
    allowed_names = {t.get("name") for t in tools}

    for call in calls:
        name = call.get("name")
        if name not in allowed_names:
            continue

        args = dict(call.get("arguments", {}))
        tool_props = properties.get(name, {})

        # Keep only known parameters and coerce by schema.
        args = {k: v for k, v in args.items() if k in tool_props}
        for key, spec in tool_props.items():
            if key not in args:
                continue
            coerced, ok = _coerce_value_to_type(args[key], spec)
            if ok:
                args[key] = coerced

        for key, spec in tool_props.items():
            if key in args:
                continue
            default_val = _infer_default_from_description(spec.get("description", ""), spec.get("type"))
            if default_val is not None:
                args[key] = default_val

        if all(field in args for field in required.get(name, [])):
            repaired.append({"name": name, "arguments": args})

    return _normalize_function_calls(repaired)


def _score_candidate(calls, messages, tools):
    """Compute structural quality score for repaired calls."""
    repaired = _repair_calls_with_schema(calls, tools)
    required = _required_fields_by_tool(tools)
    props = _properties_by_tool(tools)
    expected_count = _infer_expected_intent_count(messages, tools)

    if not repaired:
        return {
            "calls": [],
            "validity": 0.0,
            "coverage": 0.0,
            "completeness": 0.0,
            "score": 0.0,
            "expected_count": expected_count,
        }

    valid = 0
    required_hits = 0
    required_total = 0
    for call in repaired:
        name = call.get("name")
        args = call.get("arguments", {})
        req = required.get(name, [])
        required_total += len(req)
        required_hits += sum(1 for k in req if k in args)
        if all(k in args for k in req) and name in props:
            valid += 1

    validity = valid / max(1, len(repaired))
    coverage = min(1.0, len(repaired) / max(1, expected_count))
    completeness = required_hits / max(1, required_total)
    score = (0.45 * validity) + (0.35 * coverage) + (0.20 * completeness)
    return {
        "calls": repaired,
        "validity": validity,
        "coverage": coverage,
        "completeness": completeness,
        "score": score,
        "expected_count": expected_count,
    }


def _schema_issue_report(calls, tools, expected_count):
    issues = []
    if len(calls) < expected_count:
        issues.append({
            "type": "missing_calls",
            "message": f"Expected about {expected_count} calls, got {len(calls)}",
        })

    required = _required_fields_by_tool(tools)
    allowed = {t.get("name") for t in tools}
    for idx, call in enumerate(calls):
        name = call.get("name")
        args = call.get("arguments", {})
        if name not in allowed:
            issues.append({
                "type": "invalid_tool",
                "message": f"Call {idx} uses unknown tool '{name}'",
            })
            continue
        missing = [field for field in required.get(name, []) if field not in args]
        if missing:
            issues.append({
                "type": "missing_required_fields",
                "message": f"Call {idx} for '{name}' missing required fields: {missing}",
            })
    return issues


def _infer_expected_intent_count(messages, tools):
    """Estimate likely call count from punctuation/conjunction structure."""
    user_text = " ".join(
        m.get("content", "") for m in messages if m.get("role") == "user"
    ).strip()
    if not user_text:
        return 0
    connectors = len(re.findall(r"\band\b|;|,", user_text, flags=re.IGNORECASE))
    return max(1, min(len(tools), connectors + 1))


def _tokenize(text):
    return re.findall(r"[a-z0-9_]+", (text or "").lower())


def _select_tool_subset(messages, tools, expected_count):
    """Generic tool preselection via lexical overlap with tool metadata."""
    user_text = " ".join(
        m.get("content", "") for m in messages if m.get("role") == "user"
    )
    user_tokens = set(_tokenize(user_text))
    scored = []
    for t in tools:
        params = t.get("parameters", {})
        props = params.get("properties", {})
        blob = " ".join(
            [t.get("name", ""), t.get("description", "")]
            + list(props.keys())
            + [v.get("description", "") for v in props.values() if isinstance(v, dict)]
        )
        tool_tokens = set(_tokenize(blob))
        overlap = len(user_tokens.intersection(tool_tokens))
        # Prefer tools with stronger schema signal when ties occur.
        schema_weight = len(params.get("required", [])) * 0.1
        scored.append((overlap + schema_weight, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_k = max(1, min(len(tools), expected_count + 1))
    subset = [t for _, t in scored[:top_k]]
    return subset if subset else tools


def _tool_ambiguity_score(tools):
    """Estimate ambiguity from shared arg names/types across tools."""
    props = _properties_by_tool(tools)
    sig_counts = {}
    for name, schema_props in props.items():
        keys = tuple(sorted((k, v.get("type", "any")) for k, v in schema_props.items()))
        sig_counts[keys] = sig_counts.get(keys, 0) + 1
    if not sig_counts:
        return 0.0
    duplicated = sum(c for c in sig_counts.values() if c > 1)
    return duplicated / max(1, len(tools))


def _should_route_cloud_first(messages, tools):
    """Generic routing signal based on structure and schema ambiguity."""
    user_text = " ".join(
        m.get("content", "") for m in messages if m.get("role") == "user"
    ).strip()
    if not user_text:
        return False
    complexity = len(re.findall(r"\band\b|;|,", user_text, flags=re.IGNORECASE))
    ambiguity = _tool_ambiguity_score(tools)
    return complexity >= 2 and (len(tools) >= 4 or ambiguity >= 0.25)


def _tick_cloud_cooldown():
    global _CLOUD_COOLDOWN_CASES
    if _CLOUD_COOLDOWN_CASES > 0:
        _CLOUD_COOLDOWN_CASES -= 1


def _cloud_is_available():
    return _CLOUD_COOLDOWN_CASES <= 0 and bool(os.environ.get("GEMINI_API_KEY"))


def _record_cloud_result(success):
    global _CLOUD_FAILURE_STREAK, _CLOUD_COOLDOWN_CASES
    if success:
        _CLOUD_FAILURE_STREAK = 0
        _CLOUD_COOLDOWN_CASES = 0
        return
    _CLOUD_FAILURE_STREAK += 1
    _CLOUD_COOLDOWN_CASES = min(10, 2 ** min(_CLOUD_FAILURE_STREAK, 3))


def _is_local_output_valid(local, messages, tools):
    calls = local.get("function_calls", [])
    if not calls:
        return False

    user_text = " ".join(
        m.get("content", "") for m in messages if m.get("role") == "user"
    ).lower()
    allowed_names = {t.get("name") for t in tools}
    required = _required_fields_by_tool(tools)
    properties = _properties_by_tool(tools)

    # Heuristic expected call count from prompt complexity.
    if user_text.count(" and ") + user_text.count(";") >= 1 and len(calls) < 2:
        return False

    for c in calls:
        name = c.get("name")
        if name not in allowed_names:
            return False
        args = c.get("arguments", {})
        for field in required.get(name, []):
            if field not in args:
                return False

        # Light type checks against declared schema.
        tool_props = properties.get(name, {})
        for key, val in args.items():
            expected = tool_props.get(key, {}).get("type")
            if expected == "integer" and not isinstance(val, int):
                return False
            if expected == "number" and not isinstance(val, (int, float)):
                return False
            if expected == "string" and not isinstance(val, str):
                return False
            if expected == "boolean" and not isinstance(val, bool):
                return False
            if expected == "array" and not isinstance(val, list):
                return False

    return True


def generate_cactus(messages, tools, previous_calls=None, expected_count=None, mode="full", feedback=None):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = cactus_init(functiongemma_path)

    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": _build_planner_system_prompt(
            tools,
            previous_calls=previous_calls,
            expected_count=expected_count,
            mode=mode,
            feedback=feedback,
        )}] + messages,
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


def _run_two_step_local(messages, tools, expected_count):
    """Two-step local decode: choose tools first, then fill arguments."""
    by_name = {t.get("name"): t for t in tools}
    step1 = generate_cactus(
        messages,
        tools,
        expected_count=expected_count,
        mode="plan_tools",
    )
    planned = []
    for c in step1.get("function_calls", []):
        name = c.get("name")
        if name in by_name and name not in planned:
            planned.append(name)

    if planned:
        selected = [by_name[n] for n in planned[: max(1, expected_count + 1)]]
    else:
        selected = _select_tool_subset(messages, tools, expected_count)

    step2 = generate_cactus(
        messages,
        selected,
        expected_count=expected_count,
        mode="full",
    )
    step2["total_time_ms"] += step1.get("total_time_ms", 0)
    step2["confidence"] = max(step1.get("confidence", 0), step2.get("confidence", 0))
    return step2


def generate_cloud(messages, tools, retries=2):
    """Run function calling via Gemini Cloud API."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {
            "function_calls": [],
            "total_time_ms": 0.0,
            "error": "missing_gemini_api_key",
        }

    client = genai.Client(api_key=api_key)

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(
                            type=_to_gemini_schema_type(v.get("type")),
                            description=v.get("description", ""),
                        )
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

    gemini_response = None
    err = None
    for attempt in range(retries + 1):
        try:
            gemini_response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(tools=gemini_tools),
            )
            err = None
            break
        except Exception as e:
            err = e
            if attempt < retries:
                time.sleep(0.35 * (2 ** attempt))

    if gemini_response is None:
        total_time_ms = (time.time() - start_time) * 1000
        return {
            "function_calls": [],
            "total_time_ms": total_time_ms,
            "error": str(err),
        }

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


def generate_hybrid(messages, tools, confidence_threshold=0.80):
    """Phase-2 hybrid strategy: multi-candidate local planning + resilient cloud fallback."""
    _tick_cloud_cooldown()
    expected_count = _infer_expected_intent_count(messages, tools)
    allow_cloud = _cloud_is_available()
    # Candidate A: full-tool local planning.
    cand_a = generate_cactus(messages, tools, expected_count=expected_count, mode="full")
    eval_a = _score_candidate(cand_a.get("function_calls", []), messages, tools)
    cand_a["function_calls"] = eval_a["calls"]

    # Candidate B: two-step planning (tool names first, args second).
    cand_b = _run_two_step_local(messages, tools, expected_count)
    eval_b = _score_candidate(cand_b.get("function_calls", []), messages, tools)
    cand_b["function_calls"] = eval_b["calls"]

    # Candidate C: subset-tool planning for high-cardinality toolsets.
    subset_tools = _select_tool_subset(messages, tools, expected_count)
    cand_c = generate_cactus(messages, subset_tools, expected_count=expected_count, mode="full")
    eval_c = _score_candidate(cand_c.get("function_calls", []), messages, tools)
    cand_c["function_calls"] = eval_c["calls"]

    local, local_eval = max(
        [(cand_a, eval_a), (cand_b, eval_b), (cand_c, eval_c)],
        key=lambda item: item[1]["score"],
    )
    local_conf = local.get("confidence", 0)
    local_valid = _is_local_output_valid(local, messages, tools)

    # One strict local retry with explicit schema feedback.
    issues = _schema_issue_report(local["function_calls"], tools, expected_count)
    if issues:
        fix = generate_cactus(
            messages,
            tools,
            previous_calls=local["function_calls"],
            expected_count=expected_count,
            mode="fix",
            feedback=issues,
        )
        fix_eval = _score_candidate(fix.get("function_calls", []), messages, tools)
        fix["function_calls"] = fix_eval["calls"]
        if fix_eval["score"] > local_eval["score"]:
            local["function_calls"] = fix["function_calls"]
            local["confidence"] = max(local_conf, fix.get("confidence", 0))
            local["total_time_ms"] += fix.get("total_time_ms", 0)
            local_conf = local.get("confidence", 0)
            local_eval = fix_eval
            local_valid = _is_local_output_valid(local, messages, tools)

    local_uncertain = (
        local_eval["score"] < 0.85
        or local_eval["coverage"] < 1.0
        or not local_valid
        or local_conf < confidence_threshold
        or _should_route_cloud_first(messages, tools)
    )

    # Accept local immediately only when strong and complete.
    if not local_uncertain:
        local["source"] = "on-device"
        local["quality_score"] = local_eval["score"]
        return local

    if not allow_cloud:
        local["source"] = "on-device"
        local["quality_score"] = local_eval["score"]
        return local

    cloud = generate_cloud(messages, tools)
    cloud_eval = _score_candidate(cloud.get("function_calls", []), messages, tools)
    cloud["function_calls"] = cloud_eval["calls"]
    if cloud.get("error"):
        _record_cloud_result(False)
        local["source"] = "on-device (cloud-error-fallback)"
        local["cloud_error"] = cloud["error"]
        local["total_time_ms"] += cloud.get("total_time_ms", 0.0)
        local["quality_score"] = local_eval["score"]
        return local

    _record_cloud_result(True)
    if cloud_eval["score"] >= local_eval["score"]:
        cloud["source"] = "cloud (best-of)"
        cloud["local_confidence"] = local_conf
        cloud["local_quality_score"] = local_eval["score"]
        cloud["quality_score"] = cloud_eval["score"]
        cloud["total_time_ms"] += local.get("total_time_ms", 0.0)
        return cloud

    local["source"] = "on-device (best-of)"
    local["quality_score"] = local_eval["score"]
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
