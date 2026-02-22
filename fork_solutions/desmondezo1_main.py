# Source: https://github.com/desmondezo1/functiongemma-hackathon
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time, re
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types

from prompts import (
    BASE_SYSTEM_PROMPT,
    build_cot_messages,
    build_preselect_messages,
    build_hinted_system_prompt,
    build_retry_messages,
    parse_preselect_response,
    majority_vote,
    confidence_band,
)


# ─────────────────────────────────────────────
#  QUERY CLASSIFIER  (keyword-based, free)
# ─────────────────────────────────────────────

CONJUNCTION_PATTERNS = [
    r'\band\b', r'\balso\b', r'\bplus\b', r'\bthen\b',
    r'\bas well\b', r'\bon top of that\b',
]

ACTION_VERBS = [
    "set", "send", "play", "check", "find", "remind", "text",
    "wake", "look up", "get", "search", "message", "call",
    "create", "start", "put on", "stream",
]


def classify_query(messages: list, tools: list) -> dict:
    user_content = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_content = m["content"].lower()
            break

    tool_count = len(tools)
    conjunction_hits = sum(1 for p in CONJUNCTION_PATTERNS if re.search(p, user_content))
    has_conjunction = conjunction_hits > 0
    verb_hits = sum(1 for v in ACTION_VERBS if v in user_content)

    if has_conjunction:
        expected_call_count = max(conjunction_hits + 1, verb_hits)
    else:
        expected_call_count = 1

    expected_call_count = min(expected_call_count, tool_count)

    if expected_call_count >= 2:
        difficulty = "hard"
    elif tool_count > 2:
        difficulty = "medium"
    else:
        difficulty = "easy"

    return {
        "difficulty": difficulty,
        "expected_call_count": expected_call_count,
        "tool_count": tool_count,
        "has_conjunction": has_conjunction,
    }


# ─────────────────────────────────────────────
#  LOW-LEVEL CACTUS CALL
# ─────────────────────────────────────────────

def _run_cactus(messages: list, tools: list, system: str, max_tokens: int = 512, tool_rag_top_k: int = 0) -> dict:
    """Single raw call to FunctionGemma via Cactus."""
    model = cactus_init(functiongemma_path)

    cactus_tools = [{"type": "function", "function": t} for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": system}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=max_tokens,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
        tool_rag_top_k=tool_rag_top_k,
    )

    cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {"function_calls": [], "total_time_ms": 0, "confidence": 0.0}

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0.0),
    }


# ─────────────────────────────────────────────
#  ON-DEVICE STRATEGIES
# ─────────────────────────────────────────────

def generate_cactus(messages: list, tools: list, classification: dict) -> dict:
    """
    Difficulty-matched on-device inference:
      Easy   → direct call
      Medium → CoT + Tool RAG (top 3)
      Hard   → query doubling + CoT + self-consistency (3 runs)
    """
    difficulty = classification["difficulty"]

    # EASY
    if difficulty == "easy":
        return _run_cactus(messages, tools, BASE_SYSTEM_PROMPT)

    # MEDIUM
    if difficulty == "medium":
        cot_messages = build_cot_messages(messages)
        return _run_cactus(cot_messages, tools, BASE_SYSTEM_PROMPT, tool_rag_top_k=3)

    # HARD — query doubling + self-consistency
    # Step 1: pre-selection pass
    preselect_msgs = build_preselect_messages(messages, tools)
    preselect_raw = _run_cactus(
        preselect_msgs, tools,
        system="You are a tool selection assistant. Reply with only a JSON array of tool names.",
        max_tokens=64,
    )
    preselect_text = json.dumps(preselect_raw.get("function_calls", []))
    selected_tools = parse_preselect_response(preselect_text)

    # Step 2: hinted system prompt
    hinted_system = build_hinted_system_prompt(selected_tools)

    # Step 3: self-consistency — 3 runs, majority vote
    cot_messages = build_cot_messages(messages)
    all_calls = []
    total_time = 0
    last_result = {}

    for _ in range(3):
        result = _run_cactus(cot_messages, tools, hinted_system)
        all_calls.append(result["function_calls"])
        total_time += result["total_time_ms"]
        last_result = result

    return {
        "function_calls": majority_vote(all_calls),
        "total_time_ms": total_time,
        "confidence": last_result.get("confidence", 0.0),
    }


# ─────────────────────────────────────────────
#  CLOUD INFERENCE
# ─────────────────────────────────────────────

def generate_cloud(messages: list, tools: list) -> dict:
    """Run function calling via Gemini Flash cloud API."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(
                            type=v["type"].upper(),
                            description=v.get("description", "")
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

    return {"function_calls": function_calls, "total_time_ms": total_time_ms}


# ─────────────────────────────────────────────
#  VALIDATOR
# ─────────────────────────────────────────────

CONFIDENCE_THRESHOLDS = {"easy": 0.50, "medium": 0.65, "hard": 0.55}


def local_result_is_valid(local: dict, classification: dict) -> bool:
    difficulty = classification["difficulty"]
    threshold = CONFIDENCE_THRESHOLDS[difficulty]
    expected = classification["expected_call_count"]
    returned = len(local["function_calls"])

    if local["confidence"] < threshold:
        return False

    if difficulty == "hard" and returned < expected:
        return False

    return True


# ─────────────────────────────────────────────
#  HYBRID ROUTING  (do not change the interface)
# ─────────────────────────────────────────────

def generate_hybrid(messages: list, tools: list, confidence_threshold: float = 0.99) -> dict:
    """
    Hybrid routing pipeline:
    1. Classify query (free, keyword-based)
    2. Run on-device with strategy matched to difficulty
    3. Validate — if confident + correct call count → return local
    4. If in retry band → one retry with hint prompt
    5. If still failing → cloud fallback
    """
    classification = classify_query(messages, tools)
    difficulty = classification["difficulty"]

    # On-device
    local = generate_cactus(messages, tools, classification)

    # Validate
    if local_result_is_valid(local, classification):
        local["source"] = "on-device"
        local["classification"] = difficulty
        return local

    # Retry band — one more attempt before cloud
    band = confidence_band(local["confidence"])
    if band == "retry":
        retry_messages = build_retry_messages(messages, local["function_calls"])
        retry_system = build_hinted_system_prompt(
            [c["name"] for c in local["function_calls"]]
        )
        retry = _run_cactus(retry_messages, tools, retry_system)

        if local_result_is_valid(retry, classification):
            retry["source"] = "on-device (retry)"
            retry["classification"] = difficulty
            retry["total_time_ms"] += local["total_time_ms"]
            return retry

        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback after retry)"
        cloud["local_confidence"] = local["confidence"]
        cloud["total_time_ms"] += local["total_time_ms"] + retry["total_time_ms"]
        cloud["classification"] = difficulty
        return cloud

    # Direct cloud fallback
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local["confidence"]
    cloud["total_time_ms"] += local["total_time_ms"]
    cloud["classification"] = difficulty
    return cloud


# ─────────────────────────────────────────────
#  PRETTY PRINT
# ─────────────────────────────────────────────

def print_result(label: str, result: dict):
    print(f"\n=== {label} ===")
    print(f"  Source:         {result.get('source', 'unknown')}")
    print(f"  Classified as:  {result.get('classification', 'unknown')}")
    if "confidence" in result:
        print(f"  Confidence:     {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"  Local conf:     {result['local_confidence']:.4f} → cloud")
    print(f"  Total time:     {result['total_time_ms']:.2f}ms")
    calls = result.get("function_calls", [])
    if calls:
        for call in calls:
            print(f"  ↳ {call['name']}({json.dumps(call['arguments'])})")
    else:
        print("  ↳ (no function calls returned)")


# ─────────────────────────────────────────────
#  EXAMPLE USAGE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        {
            "label": "Easy — weather",
            "messages": [{"role": "user", "content": "What is the weather in San Francisco?"}],
            "tools": [{"name": "get_weather", "description": "Get current weather for a location",
                       "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City name"}}, "required": ["location"]}}],
        },
        {
            "label": "Medium — pick right tool",
            "messages": [{"role": "user", "content": "Send a message to John saying hello."}],
            "tools": [
                {"name": "get_weather", "description": "Get current weather for a location",
                 "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}},
                {"name": "send_message", "description": "Send a message to a contact",
                 "parameters": {"type": "object", "properties": {"recipient": {"type": "string"}, "message": {"type": "string"}}, "required": ["recipient", "message"]}},
                {"name": "set_alarm", "description": "Set an alarm",
                 "parameters": {"type": "object", "properties": {"hour": {"type": "integer"}, "minute": {"type": "integer"}}, "required": ["hour", "minute"]}},
            ],
        },
        {
            "label": "Hard — multi-call",
            "messages": [{"role": "user", "content": "Set an alarm for 7:30 AM and check the weather in New York."}],
            "tools": [
                {"name": "get_weather", "description": "Get current weather for a location",
                 "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}},
                {"name": "set_alarm", "description": "Set an alarm for a given time",
                 "parameters": {"type": "object", "properties": {"hour": {"type": "integer"}, "minute": {"type": "integer"}}, "required": ["hour", "minute"]}},
                {"name": "send_message", "description": "Send a message to a contact",
                 "parameters": {"type": "object", "properties": {"recipient": {"type": "string"}, "message": {"type": "string"}}, "required": ["recipient", "message"]}},
            ],
        },
    ]

    for case in test_cases:
        result = generate_hybrid(case["messages"], case["tools"])
        print_result(case["label"], result)
