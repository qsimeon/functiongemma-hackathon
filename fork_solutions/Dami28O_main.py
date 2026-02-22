# Source: https://github.com/Dami28O/functiongemma-hackathon

import sys
sys.path.insert(0, "../cactus/python/src")
functiongemma_path = "../cactus/weights/functiongemma-270m-it"

import json, os, time
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
from google import genai
from google.genai import types

# Singleton Model: Initialize once at the top level to avoid reload latency
CACTUS_MODEL = cactus_init(functiongemma_path)

def generate_cactus(messages, tools, use_reasoning=True):
    """V11: Optimized for accuracy with hardened parsing."""
    
    # 1. Reasoning Injection: Add thought_process to tool schema
    actual_tools = []
    if use_reasoning:
        for t in tools:
            t_copy = json.loads(json.dumps(t))
            t_copy["parameters"]["properties"]["thought_process"] = {
                "type": "string",
                "description": "Short logic: 1. Intent. 2. Tool choice. 3. Param logic."
            }
            if "required" not in t_copy["parameters"]:
                t_copy["parameters"]["required"] = []
            if "thought_process" not in t_copy["parameters"]["required"]:
                t_copy["parameters"]["required"].insert(0, "thought_process")
            actual_tools.append(t_copy)
    else:
        actual_tools = tools

    cactus_tools = [{"type": "function", "function": t} for t in actual_tools]

    # V14: Minified rules for 270m performance
    sys_content = (
        "You are a tool caller. Output exactly one JSON object with 'function_calls'.\n"
        "RULES: Integers=digits. Times=H:MM AM/PM. No talk."
    )
    if use_reasoning:
        sys_content = "Think then JSON. Rules: Integers=digits. Times=H:MM AM/PM. No talk."

    raw_str = cactus_complete(
        CACTUS_MODEL,
        [{"role": "system", "content": sys_content}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=400 if use_reasoning else 128,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
        temperature=0.0,
    )

    try:
        # Robust JSON extraction without re: Find the first { and last }
        start_idx = raw_str.find("{")
        end_idx = raw_str.rfind("}")
        if start_idx != -1 and end_idx != -1:
            clean_json = raw_str[start_idx:end_idx+1]
            raw = json.loads(clean_json)
        else:
            raw = json.loads(raw_str)
            
        function_calls = raw.get("function_calls", [])

        # 1. Type Casting & Cleanup (No Regex)
        for call in function_calls:
            tool_def = next((t for t in tools if t["name"] == call["name"]), None)
            if tool_def:
                props = tool_def.get("parameters", {}).get("properties", {})
                args = call.get("arguments", {})
                for p_name, p_schema in props.items():
                    if p_name in args:
                        p_type = p_schema.get("type", "").lower()
                        val = str(args[p_name])
                        try:
                            if p_type == "integer":
                                # V14: Stop at first non-digit (Fixes "10:00" -> "1000" bug)
                                digits = []
                                for i, c in enumerate(val):
                                    if c.isdigit(): digits.append(c)
                                    elif digits: break # Stop once we have digits and hit a separator
                                    elif c == "-" and i == 0: digits.append(c)
                                res = "".join(digits)
                                args[p_name] = int(res) if res else 0
                            elif p_type == "number":
                                seen_dot = False
                                digits = []
                                for i, c in enumerate(val):
                                    if c.isdigit(): digits.append(c)
                                    elif c == "." and not seen_dot:
                                        digits.append(c)
                                        seen_dot = True
                                    elif digits: break
                                    elif c == "-" and i == 0: digits.append(c)
                                res = "".join(digits)
                                args[p_name] = float(res) if res else 0.0
                        except: pass
            
            # Remove reasoning metadata from final output
            if "thought_process" in call.get("arguments", {}):
                del call["arguments"]["thought_process"]

    except Exception:
        return {"function_calls": [], "total_time_ms": 0, "confidence": 0}

    return {
        "function_calls": function_calls,
        "total_time_ms": raw.get("total_time_ms", 0) if "raw" in locals() and isinstance(raw, dict) else 0,
        "confidence": raw.get("confidence", 0) if "raw" in locals() and isinstance(raw, dict) else 0,
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


def calculate_complexity_score(messages, tools):
    """Predict complexity using structural cues (V10)."""
    q = "".join(m["content"].lower() for m in messages if m["role"] == "user")
    
    # 1. Structural Cues - Very sensitive to multi-intent markers
    has_multi = any(c in q for c in [" and ", " then ", ";", " also ", " plus "])
    len_score = len(q.split()) / 12.0 # More sensitive to long queries
    
    # 2. Toolset Complexity
    tool_score = len(tools) / 5.0
    
    return min((0.5 if has_multi else 0) + (len_score * 0.2) + (tool_score * 0.3), 1.0)


def generate_hybrid(messages, tools, confidence_threshold=0.90):
    """V14 Hybrid Strategy: Balanced Locality & F1."""
    complexity = calculate_complexity_score(messages, tools)
    
    # 1. Cloud Handoff (Threshold set to 0.45 for Locality-F1 balance)
    if complexity > 0.45:
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fast-handoff)"
        return cloud

    # 2. Local Model (Tiered Reasoning)
    use_reasoning = complexity > 0.2 
    local = generate_cactus(messages, tools, use_reasoning=use_reasoning)

    if local["confidence"] >= 0.8:
        local["source"] = "on-device"
        return local

    # Fallback
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local["confidence"]
    cloud["total_time_ms"] += local.get("total_time_ms", 0)
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
