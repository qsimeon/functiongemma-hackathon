# Source: https://github.com/nicoA0/functiongemma-hackathon
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time, re
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
from google import genai
from google.genai import types


# ─────────────────────────────────────────────────────────────
# Global model handle – init once, reuse via cactus_reset
# Saves ~30-80ms per call by avoiding repeated model loading
# ─────────────────────────────────────────────────────────────
_model = None

def _get_model():
    """Lazily initialise and return the global FunctionGemma model handle."""
    global _model
    if _model is None:
        _model = cactus_init(functiongemma_path)
    return _model


# ─────────────────────────────────────────────────────────────
# Gemini client – reuse across calls
# ─────────────────────────────────────────────────────────────
_gemini_client = None

def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    return _gemini_client


# ─────────────────────────────────────────────────────────────
# Complexity pre-classifier
# ─────────────────────────────────────────────────────────────
MULTI_CALL_INDICATORS = re.compile(
    r'\b(and|also|then|plus|additionally|as well|after that)\b|,\s*(?=[a-z])',
    re.IGNORECASE,
)

def classify_complexity(messages, tools):
    """
    Heuristically classify query complexity.
    Returns: 'easy', 'medium', or 'hard'
    """
    user_text = " ".join(
        m["content"] for m in messages if m.get("role") == "user"
    ).strip()

    num_tools = len(tools)
    multi_indicators = len(MULTI_CALL_INDICATORS.findall(user_text))

    # If only 1 tool provided → easy (model just needs to extract args)
    if num_tools <= 1 and multi_indicators == 0:
        return "easy"

    # Multiple action hints in the query → hard (likely multi-call)
    if multi_indicators >= 2:
        return "hard"

    # Single "and" with many tools → hard
    if multi_indicators >= 1 and num_tools >= 3:
        return "hard"

    # Single "and" with few tools → medium
    if multi_indicators >= 1:
        return "medium"

    # Multiple tools but single intent → medium
    if num_tools >= 2:
        return "medium"

    return "easy"


def count_expected_calls(messages):
    """
    Estimate how many distinct function calls the user is requesting.
    Uses conjunction / comma counting in the user message.
    """
    user_text = " ".join(
        m["content"] for m in messages if m.get("role") == "user"
    ).strip()

    # Count distinct action segments separated by "and", commas, "then", etc.
    segments = MULTI_CALL_INDICATORS.split(user_text)
    # Filter out empty / whitespace-only segments
    segments = [s.strip() for s in segments if s and s.strip() and len(s.strip()) > 3]
    return max(1, len(segments))


# ─────────────────────────────────────────────────────────────
# Structural validation
# ─────────────────────────────────────────────────────────────
def validate_calls(function_calls, tools):
    """
    Validate that function calls are structurally sound:
    - Function name exists in tools list
    - Required arguments are present
    Returns (is_valid, score 0.0-1.0)
    """
    if not function_calls:
        return False, 0.0

    tool_map = {t["name"]: t for t in tools}
    valid_count = 0

    for call in function_calls:
        name = call.get("name", "")
        args = call.get("arguments", {})

        # Name must match a provided tool
        if name not in tool_map:
            continue

        tool = tool_map[name]
        required = tool.get("parameters", {}).get("required", [])

        # Check required args are present (and not empty)
        has_required = all(
            k in args and args[k] is not None and args[k] != ""
            for k in required
        )
        if has_required:
            valid_count += 1

    if valid_count == 0:
        return False, 0.0

    return True, valid_count / len(function_calls)


# ─────────────────────────────────────────────────────────────
# Composite routing score
# ─────────────────────────────────────────────────────────────
def compute_routing_score(local_result, messages, tools, complexity):
    """
    Multi-signal score (0.0 - 1.0) indicating how confident we are
    in the local result.  Higher → keep on-device.
    """
    function_calls = local_result.get("function_calls", [])
    model_confidence = local_result.get("confidence", 0.0)

    # Signal 1: Model confidence (raw from Cactus)
    conf_score = model_confidence

    # Signal 2: Structural validity
    is_valid, validity_score = validate_calls(function_calls, tools)
    if not is_valid:
        return 0.0  # Immediately disqualify

    # Signal 3: Call count vs expected
    expected_count = count_expected_calls(messages)
    actual_count = len(function_calls)

    if expected_count > 1 and actual_count < expected_count:
        # Model likely missed some calls → penalise heavily
        count_score = actual_count / expected_count * 0.5
    elif actual_count >= expected_count:
        count_score = 1.0
    else:
        count_score = actual_count / max(expected_count, 1)

    # Signal 4: Argument type correctness (basic check)
    type_score = _check_arg_types(function_calls, tools)

    # Weighted combination – structural signals dominate
    if complexity == "easy":
        score = (0.20 * conf_score +
                 0.30 * validity_score +
                 0.20 * count_score +
                 0.30 * type_score)
    elif complexity == "medium":
        score = (0.25 * conf_score +
                 0.25 * validity_score +
                 0.25 * count_score +
                 0.25 * type_score)
    else:  # hard
        score = (0.20 * conf_score +
                 0.20 * validity_score +
                 0.35 * count_score +
                 0.25 * type_score)

    return score


def _check_arg_types(function_calls, tools):
    """
    Lightweight type check: integers should be ints, strings should be strings.
    Returns 0.0 - 1.0 indicating fraction of args with correct types.
    """
    tool_map = {t["name"]: t for t in tools}
    total = 0
    correct = 0

    for call in function_calls:
        name = call.get("name", "")
        args = call.get("arguments", {})
        if name not in tool_map:
            continue

        props = tool_map[name].get("parameters", {}).get("properties", {})
        for k, v in args.items():
            if k in props:
                total += 1
                expected_type = props[k].get("type", "string")
                if expected_type == "integer":
                    if isinstance(v, int) or (isinstance(v, str) and v.isdigit()):
                        correct += 1
                elif expected_type == "string":
                    if isinstance(v, str) and len(v) > 0:
                        correct += 1
                else:
                    correct += 1  # Unknown type, assume OK

    return correct / max(total, 1)


# ─────────────────────────────────────────────────────────────
# Enhanced on-device generation
# ─────────────────────────────────────────────────────────────
def _build_system_prompt(tools):
    """
    Build an optimised system prompt that helps FunctionGemma
    make better tool-calling decisions.
    """
    tool_names = [t["name"] for t in tools]
    prompt = (
        "You are a precise tool-calling assistant. "
        "You MUST call the appropriate function(s) to fulfil the user's request. "
        "Available tools: " + ", ".join(tool_names) + ". "
        "Rules:\n"
        "1. ONLY use the tools listed above.\n"
        "2. If the user asks for multiple actions, call EACH corresponding tool.\n"
        "3. Extract arguments exactly as the user states them.\n"
        "4. For integer arguments, output the number without quotes.\n"
        "5. Match function names EXACTLY as listed.\n"
    )
    return prompt


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = _get_model()
    cactus_reset(model)

    cactus_tools = [{"type": "function", "function": t} for t in tools]

    system_prompt = _build_system_prompt(tools)

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": system_prompt}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=512,
        temperature=0.0,         # Deterministic for tool calling
        top_k=1,                 # Greedy decoding
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
        }

    # Post-process: coerce int args where the schema expects integer
    function_calls = raw.get("function_calls", [])
    _coerce_arg_types(function_calls, tools)

    return {
        "function_calls": function_calls,
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


def _coerce_arg_types(function_calls, tools):
    """
    Post-process function calls to ensure argument types match the schema.
    FunctionGemma sometimes returns "10" (string) instead of 10 (int).
    """
    tool_map = {t["name"]: t for t in tools}
    for call in function_calls:
        name = call.get("name", "")
        if name not in tool_map:
            continue
        props = tool_map[name].get("parameters", {}).get("properties", {})
        args = call.get("arguments", {})
        for k, v in list(args.items()):
            if k in props:
                expected_type = props[k].get("type", "string")
                if expected_type == "integer" and isinstance(v, str):
                    try:
                        args[k] = int(v)
                    except ValueError:
                        # Try to extract number from string like "10 minutes"
                        nums = re.findall(r'\d+', v)
                        if nums:
                            args[k] = int(nums[0])
                elif expected_type == "string" and isinstance(v, (int, float)):
                    args[k] = str(v)


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    client = _get_gemini_client()

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
                fc_args = dict(part.function_call.args)
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": fc_args,
                })

    # Post-process cloud results too for type consistency
    _coerce_arg_types(function_calls, tools)

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


# ─────────────────────────────────────────────────────────────
# HYBRID ROUTER – The core strategy
# ─────────────────────────────────────────────────────────────
# Adaptive thresholds per complexity level
ROUTING_THRESHOLDS = {
    "easy":   0.35,   # Very lenient – FunctionGemma handles easy cases well
    "medium": 0.45,   # Moderate – most single-tool picks work
    "hard":   0.55,   # Stricter – multi-call is hard for small models
}


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """
    Advanced hybrid inference strategy with multi-signal routing.

    Strategy:
    1. Pre-classify query complexity (easy / medium / hard)
    2. Always run FunctionGemma on-device first (fast, ~50ms)
    3. Validate output structurally (correct tool names, required args)
    4. Compute composite routing score from multiple signals
    5. Only fall back to cloud when composite score is below threshold

    Optimised for the hackathon scoring formula:
      60% F1 correctness + 15% speed + 25% on-device ratio
    """

    # Step 1: Classify complexity
    complexity = classify_complexity(messages, tools)

    # Step 2: Run on-device
    local = generate_cactus(messages, tools)

    # Step 3: Quick-reject checks
    function_calls = local.get("function_calls", [])

    # No function calls produced → must go to cloud
    if not function_calls:
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback)"
        cloud["local_confidence"] = local.get("confidence", 0)
        cloud["total_time_ms"] += local["total_time_ms"]
        return cloud

    # Check if any function name is invalid
    valid_names = {t["name"] for t in tools}
    all_names_valid = all(
        call.get("name", "") in valid_names for call in function_calls
    )
    if not all_names_valid:
        # Invalid tool name → cloud
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback)"
        cloud["local_confidence"] = local.get("confidence", 0)
        cloud["total_time_ms"] += local["total_time_ms"]
        return cloud

    # Step 4: Compute composite routing score
    routing_score = compute_routing_score(local, messages, tools, complexity)

    # Step 5: Route based on adaptive threshold
    threshold = ROUTING_THRESHOLDS.get(complexity, 0.45)

    if routing_score >= threshold:
        local["source"] = "on-device"
        return local

    # Below threshold → cloud fallback
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local.get("confidence", 0)
    cloud["total_time_ms"] += local["total_time_ms"]
    return cloud


# ─────────────────────────────────────────────────────────────
# Pretty-print helper
# ─────────────────────────────────────────────────────────────
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
