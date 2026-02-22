# Source: https://github.com/rob-9/FunctionGemma

import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
from google import genai
from google.genai import types


# Change 1: Model singleton — init once, reuse across calls
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = cactus_init(functiongemma_path)
    return _model


def generate_cactus(messages, tools, verbose=False):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = _get_model()
    cactus_reset(model)

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

    if verbose:
        print(f"  [cactus] raw output: {raw_str!r}")

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        if verbose:
            print(f"  [cactus] JSON parse FAILED — raw was: {raw_str!r}")
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
            "_parse_failed": True,
        }

    if verbose:
        print(f"  [cactus] success={raw.get('success')}  cloud_handoff={raw.get('cloud_handoff')}")
        print(f"  [cactus] response text: {raw.get('response')!r}")
        print(f"  [cactus] confidence={raw.get('confidence')}  ttft={raw.get('time_to_first_token_ms')}ms  total={raw.get('total_time_ms')}ms")
        print(f"  [cactus] prefill_tokens={raw.get('prefill_tokens')}  decode_tokens={raw.get('decode_tokens')}  decode_tps={raw.get('decode_tps')}")
        if raw.get("function_calls"):
            for fc in raw["function_calls"]:
                print(f"  [cactus] extracted call: {fc['name']}({json.dumps(fc.get('arguments', {}))})")
        else:
            print(f"  [cactus] extracted calls: (none)")

    result = {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }
    if raw.get("cloud_handoff"):
        result["_cloud_handoff"] = True
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


def generate_hybrid(messages, tools, confidence_threshold=0.99, verbose=False):
    """Baseline hybrid inference strategy; fall back to cloud if Cactus Confidence is below threshold."""
    local = generate_cactus(messages, tools, verbose=verbose)

    if verbose:
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        tool_names = [t["name"] for t in tools]
        print(f"\n  [hybrid] query: {user_msg!r}")
        print(f"  [hybrid] available tools: {tool_names}")
        print(f"  [hybrid] local confidence: {local['confidence']:.4f} (threshold: {confidence_threshold})")
        print(f"  [hybrid] local time: {local['total_time_ms']:.2f}ms")
        if local.get("_parse_failed"):
            print(f"  [hybrid] cactus reason: JSON PARSE FAILURE (raw output was not valid JSON)")
        elif local.get("_cloud_handoff"):
            print(f"  [hybrid] cactus reason: C++ ENGINE CLOUD HANDOFF (first-token entropy too high)")
        if local["function_calls"]:
            for fc in local["function_calls"]:
                print(f"  [hybrid] local predicted: {fc['name']}({json.dumps(fc.get('arguments', {}))})")
        else:
            print(f"  [hybrid] local predicted: (no function calls)")

    if local["confidence"] >= confidence_threshold:
        if verbose:
            print(f"  [hybrid] decision: ON-DEVICE (confidence {local['confidence']:.4f} >= {confidence_threshold})")
        local["source"] = "on-device"
        local["_local_calls"] = local["function_calls"]
        return local

    if verbose:
        if local.get("_parse_failed"):
            reason = "JSON parse failure"
        elif local.get("_cloud_handoff"):
            reason = "C++ engine bailed at first token"
        else:
            reason = f"confidence {local['confidence']:.4f} < {confidence_threshold}"
        print(f"  [hybrid] decision: CLOUD FALLBACK ({reason})")

    cloud = generate_cloud(messages, tools)

    if verbose:
        print(f"  [hybrid] cloud time: {cloud['total_time_ms']:.2f}ms")
        if cloud["function_calls"]:
            for fc in cloud["function_calls"]:
                print(f"  [hybrid] cloud predicted: {fc['name']}({json.dumps(fc.get('arguments', {}))})")
        else:
            print(f"  [hybrid] cloud predicted: (no function calls)")

    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local["confidence"]
    cloud["_local_calls"] = local["function_calls"]
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
