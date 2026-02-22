# Source: https://github.com/agentcluck77/functiongemma-hackathon
import sys

sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json
import os
import re
import time

from google import genai
from google.genai import types

from cactus import cactus_complete, cactus_init, cactus_reset

_cactus_model = None


def _get_cactus_model():
    global _cactus_model
    if _cactus_model is None:
        _cactus_model = cactus_init(functiongemma_path)
    return _cactus_model


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = _get_cactus_model()
    cactus_reset(model)

    cactus_tools = [
        {
            "type": "function",
            "function": t,
        }
        for t in tools
    ]

    raw_str = cactus_complete(
        model,
        [
            {
                "role": "system",
                "content": """You are a helpful assistant that can use tools by calling functions. For each user request, output a list of function calls in JSON format. If the user asks for multiple actions, output all required function calls in the same list.

                Examples:

                User: What is the weather in Paris?
                Function call:
                [
                {"name": "get_weather", "arguments": {"location": "Paris"}}
                ]

                User: Set an alarm for 7:30 AM.
                Function call:
                [
                {"name": "set_alarm", "arguments": {"hour": 7, "minute": 30}}
                ]

                User: Send a message to Alice saying good morning.
                Function call:
                [
                {"name": "send_message", "arguments": {"recipient": "Alice", "message": "good morning"}}
                ]

                User: Send a message to Bob saying hi and get the weather in London.
                Function call:
                [
                {"name": "send_message", "arguments": {"recipient": "Bob", "message": "hi"}},
                {"name": "get_weather", "arguments": {"location": "London"}}
                ]

                User: Set an alarm for 7:30 AM and play jazz music.
                Function call:
                [
                {"name": "set_alarm", "arguments": {"hour": 7, "minute": 30}},
                {"name": "play_music", "arguments": {"song": "jazz"}}
                ]

                User: Set a 15 minute timer, remind me to stretch at 4:00 PM, and check the weather in Miami.
                Function call:
                [
                {"name": "set_timer", "arguments": {"minutes": 15}},
                {"name": "create_reminder", "arguments": {"title": "stretch", "time": "4:00 PM"}},
                {"name": "get_weather", "arguments": {"location": "Miami"}}
                ]""",
            }
        ]
        + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=512,
        temperature=0.1,
        confidence_threshold=0.3,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
            "cloud_handoff": False,
        }

    if raw.get("cloud_handoff", False):
        return {
            "function_calls": [],
            "total_time_ms": raw.get("total_time_ms", 0),
            "confidence": 0,
            "cloud_handoff": True,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
        "cloud_handoff": False,
    }


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    gemini_tools = [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            k: types.Schema(
                                type=v["type"].upper(),
                                description=v.get("description", ""),
                            )
                            for k, v in t["parameters"]["properties"].items()
                        },
                        required=t["parameters"].get("required", []),
                    ),
                )
                for t in tools
            ]
        )
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]

    start_time = time.time()

    gemini_response = client.models.generate_content(
        # model="gemini-2.5-flash",
        # model="gemini-3-flash-preview",
        # model="gemini-2.5-flash-lite-preview-09-2025",
        model="gemini-2.5-flash-lite",
        contents=contents,
        config=types.GenerateContentConfig(tools=gemini_tools),
    )

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    for candidate in gemini_response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append(
                    {
                        "name": part.function_call.name,
                        "arguments": dict(part.function_call.args),
                    }
                )

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


def _validate_function_calls(function_calls, tools):
    """
    Validate that each predicted function call references a known tool
    and has all required arguments populated with non-None values.
    Returns False if any call is missing required args, calls an unknown tool,
    or if the result is empty.
    """
    tool_schema = {t["name"]: t["parameters"].get("required", []) for t in tools}

    if not function_calls:
        return False

    for call in function_calls:
        name = call.get("name")
        args = call.get("arguments", {})
        if name not in tool_schema:
            return False
        for required_arg in tool_schema[name]:
            if required_arg not in args or args[required_arg] is None:
                return False

    return True


def _detect_multi_action_request(messages):
    """
    Detect obviously multi-action user prompts and route directly to cloud.
    This reduces latency/overhead on requests that likely need multiple tool calls.
    """
    latest_user_text = ""
    for message in reversed(messages):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            latest_user_text = message["content"].lower()
            break

    if not latest_user_text:
        return False

    connectors = [" and ", " also ", " then ", " plus ", " while ", " as well as "]
    if not any(connector in latest_user_text for connector in connectors):
        return False

    action_keywords = [
        "alarm",
        "timer",
        "weather",
        "message",
        "remind",
        "call",
        "email",
        "play",
        "book",
        "navigate",
        "set",
        "check",
        "send",
        "create",
    ]
    action_count = 0
    for keyword in action_keywords:
        if re.search(rf"\b{re.escape(keyword)}\b", latest_user_text):
            action_count += 1
            if action_count >= 2:
                return True

    return False


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """Hybrid inference: route obvious multi-action prompts to cloud, otherwise fall back as needed."""
    if _detect_multi_action_request(messages):
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (multi-action heuristic)"
        cloud["local_confidence"] = 0.0
        return cloud

    local = generate_cactus(messages, tools)
    valid_local = _validate_function_calls(local["function_calls"], tools)

    if (
        not local.get("cloud_handoff", False)
        and local["confidence"] >= confidence_threshold
        and valid_local
    ):
        local["source"] = "on-device"
        return local

    cloud = generate_cloud(messages, tools)
    if local.get("cloud_handoff", False):
        cloud["source"] = "cloud (fallback: local cloud_handoff)"
    elif not valid_local:
        cloud["source"] = "cloud (fallback: local validation)"
    else:
        cloud["source"] = "cloud (fallback: low confidence)"
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
    tools = [
        {
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
        }
    ]

    messages = [{"role": "user", "content": "What is the weather in San Francisco?"}]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)
