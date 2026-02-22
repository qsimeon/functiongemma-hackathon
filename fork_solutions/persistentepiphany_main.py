# Source: https://github.com/persistentepiphany/functiongemma-hackathon

import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, re, time, traceback
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
from google import genai
from google.genai import types
from classifier import classify_difficulty

# ── Persistent model singleton: init once, reuse across calls ──
_model = cactus_init(functiongemma_path)


# ─── Validation ─────────────────────────────────────────────────

def _validate_local_result(function_calls, tools):
    """Accept local result if every call references a valid tool and has arguments."""
    if not function_calls:
        return False
    valid_names = {t["name"] for t in tools}
    for call in function_calls:
        if not isinstance(call, dict):
            return False
        if call.get("name") not in valid_names:
            return False
        if not isinstance(call.get("arguments"), dict):
            return False
    return True


def _estimate_expected_calls(user_message):
    """Heuristic: count conjunctions/commas to guess how many calls are needed."""
    text = user_message.lower()
    count = 1
    count += text.count(" and ")
    commas = text.count(", ")
    commas -= text.count(", and")
    count += max(0, commas)
    return count


# ─── FunctionGemma prompt format ────────────────────────────────
# The C++ format_gemma_style() uses the WRONG prompt format (raw JSON
# with a generic preamble).  FunctionGemma was trained with a specific
# Jinja2 template using <start_function_declaration> tokens.  We format
# tool declarations in Python and pass them as a developer message so
# the C++ doesn't inject its wrong format.

def _fg_format_value(value):
    """Format a value in FunctionGemma's declaration syntax."""
    if isinstance(value, str):
        return f"<escape>{value}<escape>"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        parts = (f"{k}:{_fg_format_value(v)}" for k, v in sorted(value.items()))
        return "{" + ",".join(parts) + "}"
    if isinstance(value, list):
        return "[" + ",".join(_fg_format_value(i) for i in value) + "]"
    return str(value)


def _fg_format_params(properties, required=None):
    """Format parameter properties matching FunctionGemma's Jinja2 template."""
    skip = {'description', 'type', 'properties', 'required', 'nullable'}
    parts = []
    for key in sorted(properties.keys()):
        if key in skip:
            continue
        val = properties[key]
        pp = [f"description:<escape>{val.get('description', '')}<escape>"]
        ptype = val.get('type', 'string').upper()

        if ptype == 'STRING' and 'enum' in val:
            pp.append(f"enum:{_fg_format_value(val['enum'])}")
        elif ptype == 'OBJECT':
            if 'properties' in val and isinstance(val['properties'], dict):
                inner = _fg_format_params(val['properties'], val.get('required', []))
                pp.append(f"properties:{{{inner}}}")
            if val.get('required'):
                rr = ",".join(f"<escape>{r}<escape>" for r in val['required'])
                pp.append(f"required:[{rr}]")
        elif ptype == 'ARRAY' and 'items' in val and isinstance(val['items'], dict):
            ip = []
            items = val['items']
            for ik in sorted(items.keys()):
                iv = items[ik]
                if iv is None:
                    continue
                if ik == 'properties' and isinstance(iv, dict):
                    inner = _fg_format_params(iv, items.get('required', []))
                    ip.append(f"properties:{{{inner}}}")
                elif ik == 'required':
                    rr = ",".join(f"<escape>{r}<escape>" for r in iv)
                    ip.append(f"required:[{rr}]")
                elif ik == 'type':
                    if isinstance(iv, str):
                        ip.append(f"type:{_fg_format_value(iv.upper())}")
                    else:
                        ip.append(f"type:{_fg_format_value([v.upper() for v in iv])}")
                else:
                    ip.append(f"{ik}:{_fg_format_value(iv)}")
            pp.append(f"items:{{{','.join(ip)}}}")

        pp.append(f"type:<escape>{ptype}<escape>")
        parts.append(f"{key}:{{{','.join(pp)}}}")

    return ",".join(parts)


def _fg_format_declarations(tools):
    """Format tools as FunctionGemma <start_function_declaration> strings."""
    decls = []
    for t in tools:
        params = t.get('parameters', {})
        s = f"declaration:{t['name']}{{description:<escape>{t['description']}<escape>"
        if params:
            s += ",parameters:{"
            if params.get('properties'):
                inner = _fg_format_params(params['properties'], params.get('required', []))
                s += f"properties:{{{inner}}},"
            if params.get('required'):
                rr = ",".join(f"<escape>{r}<escape>" for r in params['required'])
                s += f"required:[{rr}],"
            if params.get('type'):
                s += f"type:<escape>{params['type'].upper()}<escape>"
            s += "}"
        s += "}"
        decls.append(f"<start_function_declaration>{s}<end_function_declaration>")
    return "".join(decls)


def _fg_parse_args(s):
    """Parse key:value argument string from FunctionGemma function call output."""
    args = {}
    i, n = 0, len(s)
    while i < n:
        while i < n and s[i] in ' ,\n\t':
            i += 1
        if i >= n:
            break
        # Key
        j = i
        while j < n and s[j] != ':':
            j += 1
        if j >= n:
            break
        key = s[i:j].strip()
        i = j + 1
        if i >= n:
            break
        # Value
        if s[i:].startswith('<escape>'):
            i += 8
            end = s.find('<escape>', i)
            if end == -1:
                args[key] = s[i:]
                break
            args[key] = s[i:end]
            i = end + 8
        elif s[i] == '{':
            depth, j = 0, i
            while j < n:
                if s[j] == '{':
                    depth += 1
                elif s[j] == '}':
                    depth -= 1
                    if depth == 0:
                        j += 1
                        break
                j += 1
            args[key] = _fg_parse_args(s[i + 1:j - 1])
            i = j
        elif s[i] == '[':
            depth, j = 0, i
            while j < n:
                if s[j] == '[':
                    depth += 1
                elif s[j] == ']':
                    depth -= 1
                    if depth == 0:
                        j += 1
                        break
                j += 1
            args[key] = s[i + 1:j - 1]
            i = j
        else:
            j = i
            while j < n and s[j] not in ',}':
                j += 1
            v = s[i:j].strip()
            if v == 'true':
                args[key] = True
            elif v == 'false':
                args[key] = False
            else:
                try:
                    args[key] = int(v)
                except ValueError:
                    try:
                        args[key] = float(v)
                    except ValueError:
                        args[key] = v
            i = j
    return args


def _fg_parse_response(response_text):
    """Parse function calls from FunctionGemma's raw output text."""
    calls = []
    for raw in re.findall(
        r'<start_function_call>(.*?)<end_function_call>', response_text, re.DOTALL
    ):
        m = re.match(r'call:(\w+)\{(.*)\}$', raw, re.DOTALL)
        if m:
            calls.append({"name": m.group(1), "arguments": _fg_parse_args(m.group(2))})
    return calls


def _fg_coerce_types(function_calls, tools):
    """Convert argument types based on tool parameter schema."""
    schema = {}
    for t in tools:
        props = t.get('parameters', {}).get('properties', {})
        schema[t['name']] = props
    for call in function_calls:
        props = schema.get(call['name'], {})
        for key, value in list(call['arguments'].items()):
            expected = props.get(key, {}).get('type', '').lower()
            if expected == 'integer' and isinstance(value, str):
                try:
                    call['arguments'][key] = int(value)
                except ValueError:
                    pass
            elif expected == 'number' and isinstance(value, str):
                try:
                    call['arguments'][key] = float(value)
                except ValueError:
                    pass
    return function_calls


# ─── Inference backends ─────────────────────────────────────────

def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus.

    We bypass Cactus's C++ chat template (which uses the wrong format)
    by formatting tool declarations ourselves using FunctionGemma's
    Jinja2 template format and passing them as a developer message.
    """
    global _model
    cactus_reset(_model)

    # Format tools as FunctionGemma declarations in a developer message
    decl_content = _fg_format_declarations(tools)
    fg_messages = [{"role": "developer", "content": decl_content}] + messages

    raw_str = cactus_complete(
        _model,
        fg_messages,
        tools=None,           # don't pass tools → C++ won't inject wrong format
        force_tools=False,
        temperature=0.0,
        top_k=1,
        max_tokens=512,
        confidence_threshold=0.01,
        stop_sequences=["<end_of_turn>"],
    )

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        print(f"    [DEBUG] JSON decode failed, raw_str={raw_str[:500]}")
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
        }

    # Parse function calls from raw response text (C++ won't parse them
    # since we didn't pass tools)
    response_text = raw.get("response", "") or ""
    function_calls = _fg_parse_response(response_text)
    function_calls = _fg_coerce_types(function_calls, tools)

    print(
        f"    [DEBUG] confidence={raw.get('confidence', '?'):.3f}"
        f" response={response_text[:200]}"
        f" parsed_calls={function_calls}"
    )

    return {
        "function_calls": function_calls,
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


def generate_cactus_multipass(messages, tools):
    """Run on-device inference with optional second pass for multi-call cases."""
    result = generate_cactus(messages, tools)

    user_msg = messages[-1]["content"] if messages else ""
    expected = _estimate_expected_calls(user_msg)

    if len(result["function_calls"]) < expected:
        result2 = generate_cactus(messages, tools)
        seen = {c["name"] for c in result["function_calls"]}
        for call in result2["function_calls"]:
            if call["name"] not in seen:
                result["function_calls"].append(call)
                seen.add(call["name"])
        result["total_time_ms"] += result2["total_time_ms"]

    return result


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


# ─── Hybrid routing (submission interface) ──────────────────────

def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """Difficulty-aware hybrid routing.

    Classifies query difficulty BEFORE running inference, then picks
    the cheapest path likely to succeed:

        easy   → local only, no cloud overhead
        medium → local first, validate, cloud fallback
        hard   → direct to cloud, skip wasted local pass
    """
    difficulty = classify_difficulty(messages, tools)
    print(f"    [ROUTER] difficulty={difficulty}")

    # ── easy: always local ──────────────────────────────────────
    if difficulty == "easy":
        result = generate_cactus(messages, tools)
        result["source"] = "on-device"
        result["difficulty"] = difficulty
        return result

    # ── medium: local-first with cloud fallback ─────────────────
    if difficulty == "medium":
        local = generate_cactus_multipass(messages, tools)
        if _validate_local_result(local["function_calls"], tools):
            local["source"] = "on-device"
            local["difficulty"] = difficulty
            return local
        try:
            cloud = generate_cloud(messages, tools)
            cloud["source"] = "cloud (fallback)"
            cloud["difficulty"] = difficulty
            cloud["local_confidence"] = local.get("confidence", 0)
            cloud["total_time_ms"] += local["total_time_ms"]
            return cloud
        except Exception as e:
            print(f"    [CLOUD ERROR] fallback failed: {e}")
            traceback.print_exc()
            local["source"] = "on-device"
            local["difficulty"] = difficulty
            return local

    # ── hard: direct to cloud ───────────────────────────────────
    try:
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (direct)"
        cloud["difficulty"] = difficulty
        return cloud
    except Exception as e:
        print(f"    [CLOUD ERROR] direct failed: {e}")
        traceback.print_exc()
        # Cloud unreachable — last-resort local attempt
        local = generate_cactus_multipass(messages, tools)
        local["source"] = "on-device (cloud-failed)"
        local["difficulty"] = difficulty
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
