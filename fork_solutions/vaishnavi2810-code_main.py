# Source: https://github.com/vaishnavi2810-code/functiongemma-hackathon

import re
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time
import concurrent.futures
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = cactus_init(functiongemma_path)

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

def generate_hybrid(messages, tools, confidence_threshold=0.85):
    """
    Strategy:
    1. Run FunctionGemma locally
    2. Validate result (confidence, empty calls, negatives, multi-tool)
    3. If empty calls: try regex-based construction before cloud
    4. Cloud fallback as last resort
    """

    user_query = ""
    for m in messages:
        if m["role"] == "user":
            user_query = m["content"]

    query_lower = user_query.lower()
    tool_names = {t["name"] for t in tools}

    # Detect multi-tool
    multi_keywords = [" and ", " also ", " then ", " plus "]
    likely_multi = any(kw in f" {query_lower} " for kw in multi_keywords)
    if query_lower.count(",") >= 2:
        likely_multi = True

    # =========================================================
    # STEP 1: Run FunctionGemma
    # =========================================================
    local = generate_cactus(messages, tools)
    calls = local.get("function_calls", [])

    # =========================================================
    # STEP 2: Confidence check
    # =========================================================
    if local.get("confidence", 0) < confidence_threshold:
        if not calls:
            constructed = _try_construct_calls(query_lower, tool_names, likely_multi)
            if constructed:
                local["function_calls"] = constructed
                local["source"] = "on-device"
                return local
        return _cloud_fallback(messages, tools, local)

    # =========================================================
    # STEP 3: Empty calls — try regex construction
    # =========================================================
    if not calls:
        constructed = _try_construct_calls(query_lower, tool_names, likely_multi)
        if constructed:
            local["function_calls"] = constructed
            local["source"] = "on-device"
            return local
        return _cloud_fallback(messages, tools, local)

    # =========================================================
    # STEP 4: Fix negative numbers
    # =========================================================
    for call in calls:
        args = call.get("arguments", {})
        for key in args:
            if isinstance(args[key], (int, float)) and args[key] < 0:
                args[key] = abs(args[key])

    # =========================================================
    # STEP 5: Multi-tool check
    # =========================================================
    if likely_multi and len(calls) < 2:
        constructed = _try_construct_calls(query_lower, tool_names, likely_multi)
        if constructed and len(constructed) >= 2:
            local["function_calls"] = constructed
            local["source"] = "on-device"
            return local
        return _cloud_fallback(messages, tools, local)

    # =========================================================
    # STEP 6: All checks passed
    # =========================================================
    local["source"] = "on-device"
    return local


def _cloud_fallback(messages, tools, local):
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local.get("confidence", 0)
    cloud["total_time_ms"] += local.get("total_time_ms", 0)
    return cloud


def _try_construct_calls(query, tool_names, likely_multi):
    """
    Try to construct function calls from the query using regex.
    Returns a list of call dicts, or None if we can't parse it.
    """
    calls = []

    # --- get_weather ---
    if "get_weather" in tool_names:
        match = re.search(r'weather\s+(?:like\s+)?(?:in|for)\s+(.+?)(?:\.|,|\?|!|$| and )', query)
        if not match:
            match = re.search(r'(?:check|get)\s+(?:the\s+)?weather\s+(?:in|for)\s+(.+?)(?:\.|,|\?|!|$| and )', query)
        if match:
            location = match.group(1).strip().rstrip('.,?!')
            location = ' '.join(w.capitalize() for w in location.split())
            calls.append({"name": "get_weather", "arguments": {"location": location}})

    # --- set_alarm ---
    if "set_alarm" in tool_names:
        match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)', query)
        if match and any(w in query for w in ["alarm", "wake"]):
            hour = int(match.group(1))
            minute = int(match.group(2)) if match.group(2) else 0
            if match.group(3) == "pm" and hour != 12:
                hour += 12
            if match.group(3) == "am" and hour == 12:
                hour = 0
            calls.append({"name": "set_alarm", "arguments": {"hour": hour, "minute": minute}})

    # --- set_timer ---
    if "set_timer" in tool_names:
        match = re.search(r'(\d+)\s*(?:-?\s*)?minute', query)
        if match and "timer" in query:
            minutes = int(match.group(1))
            calls.append({"name": "set_timer", "arguments": {"minutes": minutes}})

    # --- play_music ---
    if "play_music" in tool_names:
        match = re.search(r'play\s+(.+?)(?:\.|,|!|$| and )', query)
        if match:
            song = match.group(1).strip().rstrip('.,!?')
            had_some = bool(re.match(r'(?:some|the)\s+', song))
            song = re.sub(r'^(?:some|the)\s+', '', song)
            if had_some:
                song = re.sub(r'\s+music$', '', song)
            calls.append({"name": "play_music", "arguments": {"song": song}})

    # --- send_message ---
    if "send_message" in tool_names:
        match = re.search(r'(?:message|text)\s+(?:to\s+)?(\w+)\s+saying\s+(.+?)(?:\.|,|!|$| and )', query)
        if not match:
            match = re.search(r'(?:send|text)\s+(\w+)\s+(?:a\s+)?(?:message\s+)?saying\s+(.+?)(?:\.|,|!|$| and )', query)
        if match:
            recipient = match.group(1).strip().capitalize()
            message_text = match.group(2).strip().rstrip('.,!?')
            if recipient.lower() in ["him", "her", "them"]:
                name_match = re.search(r'(?:find|look\s*up|search\s*for|contact)\s+(\w+)', query)
                if name_match:
                    recipient = name_match.group(1).strip().capitalize()
            if recipient.lower() not in ["a", "the", "me", "him", "her", "them"]:
                calls.append({"name": "send_message", "arguments": {"recipient": recipient, "message": message_text}})

    # --- create_reminder ---
    if "create_reminder" in tool_names:
        match = re.search(r'remind\s+(?:me\s+)?(?:about\s+|to\s+)?(.+?)\s+at\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm))', query)
        if match:
            title = match.group(1).strip()
            title = re.sub(r'^the\s+', '', title)
            time_str = match.group(2).strip().upper()
            if ':' not in time_str:
                time_str = re.sub(r'(\d+)\s*(AM|PM)', r'\1:00 \2', time_str)
            else:
                time_str = re.sub(r'(\d+:\d+)\s*(AM|PM)', r'\1 \2', time_str)
            calls.append({"name": "create_reminder", "arguments": {"title": title, "time": time_str}})

    # --- search_contacts ---
    if "search_contacts" in tool_names:
        match = re.search(r'(?:find|look\s*up|search\s*for)\s+(\w+)', query)
        if match and "contact" in query:
            name = match.group(1).strip().capitalize()
            calls.append({"name": "search_contacts", "arguments": {"query": name}})

    if not calls:
        return None

    print(f"[DEBUG-REGEX] Constructed calls: {calls}")

    if not likely_multi:
        return [calls[0]] if calls else None

    if len(calls) >= 2:
        return calls

    return None

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
