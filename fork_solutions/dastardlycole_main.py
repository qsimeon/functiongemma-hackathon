# Source: https://github.com/dastardlycole/functiongemma-hackathon

import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types


# ---------------------------------------------------------------------------
# Persistent handles — init once per process, not once per call
# ---------------------------------------------------------------------------

_model = cactus_init(functiongemma_path)
_gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


# ---------------------------------------------------------------------------
# Multi-call pre-flight detector
# ---------------------------------------------------------------------------
# Each inner set represents one "tool category".
# If ≥ 2 categories appear in the user message → the query almost certainly
# requires multiple distinct tool calls, which FunctionGemma (270 M params)
# cannot reliably chain.  Route those straight to cloud.

_TOOL_CATEGORY_KEYWORDS: list[set[str]] = [
    {"weather", "temperature", "forecast"},
    {"alarm", "wake"},
    {"message", "text", "send"},
    {"remind", "reminder"},
    {"search", "find", "look up", "contact"},
    {"play", "music", "song", "beats", "playlist"},
    {"timer"},
]


def _count_tool_categories(content: str) -> int:
    lc = content.lower()
    return sum(1 for cats in _TOOL_CATEGORY_KEYWORDS if any(kw in lc for kw in cats))


def _is_likely_multi_call(messages: list[dict]) -> bool:
    """Return True when the query references ≥ 2 distinct tool categories."""
    content = " ".join(
        str(m.get("content", "")) for m in messages if m.get("role") == "user"
    )
    return _count_tool_categories(content) >= 2


# ---------------------------------------------------------------------------
# Dynamic confidence threshold (DMind-3 inspired)
# ---------------------------------------------------------------------------
# Informational tools (read-only, low risk of harm if wrong) → lower threshold
# so FunctionGemma's output is trusted more aggressively, maximising on-device ratio.
# Action tools (mutate state, send data) → higher threshold for safety.

_INFORMATIONAL_TOOLS = {"get_weather", "search_contacts", "play_music"}
_ACTION_TOOLS        = {"send_message", "set_alarm", "set_timer", "create_reminder"}

def _confidence_floor(tools: list[dict]) -> float:
    """Return the appropriate local confidence floor for the given tool set."""
    names = {t["name"] for t in tools}
    if names <= _INFORMATIONAL_TOOLS:
        return 0.60   # pure read-only: trust local aggressively
    if names & {"send_message"}:
        return 0.75   # sends data externally: be more careful
    return 0.65       # default for benign action tools


def _strip_trailing_punct(s: str) -> str:
    """Strip sentence-ending punctuation that models sometimes append."""
    return s.rstrip(".,!?")


def _clean_args(function_calls: list[dict]) -> list[dict]:
    """Strip trailing punctuation from all string argument values."""
    for call in function_calls:
        for k, v in call.get("arguments", {}).items():
            if isinstance(v, str):
                call["arguments"][k] = _strip_trailing_punct(v)
    return function_calls


def _local_output_valid(function_calls: list[dict], tools: list[dict]) -> bool:
    """
    Sanity-check FunctionGemma output for obvious hallucinations.

    Catches:
      • Missing required arguments  (e.g. play_music with arguments={})
      • Negative numeric values     (e.g. set_timer with minutes=-5)
      • Recipient name embedded in message body (e.g. message="Hello John"
        when recipient="John" — FunctionGemma conflates the two fields)
    Any condition forces a cloud fallback rather than returning a bad result.
    """
    tool_map = {t["name"]: t for t in tools}
    for call in function_calls:
        tool = tool_map.get(call["name"])
        if not tool:
            return False
        required = tool.get("parameters", {}).get("required", [])
        args = call.get("arguments", {})
        if not all(r in args for r in required):
            return False
        for val in args.values():
            if isinstance(val, (int, float)) and val < 0:
                return False
            if isinstance(val, str) and not val.strip():
                return False  # empty / whitespace-only string
            if isinstance(val, str) and not val.isascii():
                return False  # multilingual hallucination
        # send_message specific: recipient name must not appear inside message body
        if call["name"] == "send_message":
            recipient = args.get("recipient", "").lower().strip(".,!?'\"")
            message = args.get("message", "").lower()
            if recipient and recipient in message:
                return False
        if call["name"] == "set_alarm":
            hour   = args.get("hour", -1)
            minute = args.get("minute", -1)
            if hour == minute and minute != 0:
                return False  # model duplicated the hour digit into minute
    return True


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def generate_cactus(messages, tools):
    """Run function-calling on-device via FunctionGemma + Cactus."""
    cactus_tools = [{"type": "function", "function": t} for t in tools]

    raw_str = cactus_complete(
        _model,
        # Exact trigger phrase required by FunctionGemma to activate function-calling mode.
        # Any other phrasing causes the model to fall back to standard text generation.
        [{"role": "system", "content": "You are a model that can do function calling with the following functions"}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=64,          # function calls are ~20-30 tokens; 64 is plenty and 4x faster
        tool_rag_top_k=2,       # pre-select top-2 relevant tools via RAG — wider net than k=1
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {"function_calls": [], "total_time_ms": 0, "confidence": 0.0, "cloud_handoff": True}

    return {
        "function_calls": _clean_args(raw.get("function_calls", [])),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0.0),
        "cloud_handoff": raw.get("cloud_handoff", False),
    }


def generate_cloud(messages, tools):
    """Escalate to Gemini Flash for multi-call or low-confidence queries."""
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

    contents = [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part(text=str(m.get("content", "")))],
        )
        for m in messages
    ]

    start = time.time()
    resp = _gemini_client.models.generate_content(
        model="gemini-2.5-flash-lite",  # lowest TTFT in Gemini 2.5 family
        contents=contents,
        config=types.GenerateContentConfig(
            tools=gemini_tools,
            thinking_config=types.ThinkingConfig(thinking_budget=0),  # suppress reasoning tokens
            system_instruction=(
                "You are a function-calling assistant. "
                "Call ALL the tools required to fully satisfy the user's request. "
                "If the request mentions multiple actions, return a function call for each one. "
                "Use the minimal exact value from the user's message for each argument — "
                "for example use 'meeting' not 'the meeting', 'San Francisco' not 'San Francisco, CA'. "
                "When the user refers to a person with a pronoun ('him', 'her', 'them'), use that "
                "person's name as the recipient — never use words like 'saying', 'about', or 'that'."
            ),
        ),
    )
    total_time_ms = (time.time() - start) * 1000

    function_calls = []
    for candidate in resp.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                })

    return {"function_calls": _clean_args(function_calls), "total_time_ms": total_time_ms}


# ---------------------------------------------------------------------------
# Public interface — DO NOT change the signature
# ---------------------------------------------------------------------------

def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """
    Hybrid routing: maximise on-device execution while maintaining correctness.

    Routing logic
    ─────────────
    Gate 1  Multi-call detection
            If the user message references ≥ 2 distinct tool categories
            (e.g. "send a message … and check the weather"), skip local
            entirely and go straight to Gemini.  FunctionGemma is a 270 M
            parameter model — it is excellent at single tool calls but
            unreliable at chaining multiple calls.  Skipping local also
            avoids adding wasted local latency to the cloud round-trip.

    Gate 2  On-device attempt
            For single-tool queries, run FunctionGemma locally.

    Gate 3  Confidence / handoff check
            Trust the local result if:
              • it returned at least one function_call, AND
              • the model did NOT signal cloud_handoff, AND
              • confidence >= dynamic floor (0.60 for informational tools,
                0.75 for send_message, 0.65 for other action tools).
            Otherwise fall back to Gemini.

    Scoring impact
    ──────────────
    • Easy / medium benchmarks → local (fast, on-device ratio ↑, F1 good)
    • Hard benchmarks (multi-call) → cloud directly (F1 ↑, no wasted latency)
    """

    # Gate 1: skip local for queries that need multiple tool calls
    if _is_likely_multi_call(messages):
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (multi-call)"
        return cloud

    # Gate 2: attempt local inference
    local = generate_cactus(messages, tools)

    # Gate 3: return local result if the model is confident and productive.
    # Threshold is dynamically set per tool risk level (DMind-3 pattern):
    # informational tools → 0.60, action tools → 0.65-0.75.
    LOCAL_CONFIDENCE_FLOOR = _confidence_floor(tools)
    if (local["function_calls"]
            and not local["cloud_handoff"]
            and local["confidence"] >= LOCAL_CONFIDENCE_FLOOR
            and _local_output_valid(local["function_calls"], tools)):
        local["source"] = "on-device"
        return local

    # Fallback: local produced nothing useful or signalled a cloud handoff
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local["confidence"]
    cloud["total_time_ms"] += local["total_time_ms"]
    return cloud


# ---------------------------------------------------------------------------
# Quick smoke-test — python main.py
# ---------------------------------------------------------------------------

def print_result(label, result):
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


if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"],
        },
    }]

    # Single-call → should go on-device
    single = generate_hybrid(
        [{"role": "user", "content": "What is the weather in San Francisco?"}],
        tools,
    )
    print_result("Single-call (expect on-device)", single)

    # Multi-call → should go to cloud directly
    multi_tools = tools + [{
        "name": "set_alarm",
        "description": "Set an alarm",
        "parameters": {
            "type": "object",
            "properties": {
                "hour": {"type": "integer"},
                "minute": {"type": "integer"},
            },
            "required": ["hour", "minute"],
        },
    }]
    multi = generate_hybrid(
        [{"role": "user", "content": "Check the weather in London and set an alarm for 7 AM."}],
        multi_tools,
    )
    print_result("Multi-call (expect cloud)", multi)

    cactus_destroy(_model)
