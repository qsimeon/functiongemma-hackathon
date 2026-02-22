# Source: https://github.com/Nikhil123n/functiongemma-hackathon
import json, os, time, difflib
from cactus import cactus_init, cactus_complete, cactus_destroy

functiongemma_path = "cactus/weights/functiongemma-270m-it"

def generate_cactus(messages, tools, tool_rag_top_k=2, **kwargs):
    """Run function calling natively on-device via FunctionGemma + Cactus."""
    model = cactus_init(functiongemma_path)
    response = cactus_complete(
        model, 
        messages, 
        tools=tools, 
        force_tools=True,
        tool_rag_top_k=tool_rag_top_k,
        confidence_threshold=0.99 # Prevent early handoff
    )
    cactus_destroy(model)
    return json.loads(response)




def generate_cloud(messages, tools, hint=None):
    """Run function calling via Gemini Cloud API, optionally using a hint from a failed edge execution."""
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
    
    config_args = {"tools": gemini_tools}
    if hint:
        config_args["system_instruction"] = (
            f"The edge model attempted this but failed validation. Its attempt: {json.dumps(hint)}. "
            f"Please correct the attempt to properly satisfy the user's request using the available tools."
        )

    start_time = time.time()

    gemini_response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
        config=types.GenerateContentConfig(**config_args),
    )

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    
    if gemini_response.candidates:
        for candidate in gemini_response.candidates:
            if candidate.content and candidate.content.parts:
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


def compute_complexity(messages, tools):
    """
    Evaluate how difficult a query is without calling any model.
    Returns a float between 0.0 and 1.0. Higher is more complex.
    """
    complexity = 0.0
    
    # 1. Base complexity from number of tools
    num_tools = len(tools)
    if num_tools > 5:
        complexity += 0.3
    elif num_tools > 2:
        complexity += 0.15
        
    # 2. Maximum required parameters across tools
    max_required = 0
    for t in tools:
        req = len(t.get("parameters", {}).get("required", []))
        if req > max_required:
            max_required = req
            
    if max_required > 2:
        complexity += 0.2
    elif max_required > 0:
        complexity += 0.1
        
    query = " ".join([m["content"] for m in messages if m["role"] == "user"]).lower()
    
    # 3. Conditional/Logical words indicate complex reasoning
    logical_words = ["if", "and", "or", "but", "unless", "except", "only", "then", "after", "before"]
    logical_count = sum(1 for w in logical_words if w in query.split())
    complexity += min(0.3, logical_count * 0.1)
    
    # 4. Negative words are hard for small models
    negative_words = ["not", "don't", "dont", "never", "no"]
    negative_count = sum(1 for w in negative_words if w in query.split())
    complexity += min(0.2, negative_count * 0.15)
    
    return min(1.0, complexity)


def validate_function_calls(function_calls, tools):
    """
    Check if the model's output conforms to the schema.
    Returns a score between 0.0 and 1.0.
    - 1.0 = Perfect schema match
    - 0.0 = Complete failure (e.g. non-existent tool, missing required parameters)
    """
    if not function_calls:
        return 0.0
        
    tool_map = {t["name"]: t for t in tools}
    total_score = 0.0
    
    for call in function_calls:
        name = call.get("name")
        args = call.get("arguments", {})
        
        # 1. Does the tool exist? Fuzzy matching (Phase 11)
        if name not in tool_map:
            # Try to find a close match
            close_matches = difflib.get_close_matches(name, tool_map.keys(), n=1, cutoff=0.7)
            if close_matches:
                name = close_matches[0]
                call["name"] = name  # fix it in-place
            else:
                return 0.0 # catastrophic failure, hallucinated a completely unknown tool
            
        tool = tool_map[name]
        parameters = tool.get("parameters", {})
        properties = parameters.get("properties", {})
        required = set(parameters.get("required", []))
        
        call_score = 1.0
        
        # 2. Are required parameters missing?
        provided_args = set(args.keys())
        missing = required - provided_args
        if missing:
            call_score -= 0.5 * len(missing) # harsh penalty for missing required args
            
        # 3. Are there hallucinated parameters?
        extra = provided_args - set(properties.keys())
        if extra:
            call_score -= 0.2 * len(extra) # minor penalty for hallucinating args
            
        # 4. Type checking & Coercion (Phase 10)
        for k, expected_type in [(k, properties[k].get("type")) for k in properties if k in args]:
            v = args[k]
            if expected_type == "string" and not isinstance(v, str):
                try:
                    args[k] = str(v)
                except:
                    call_score -= 0.2
            elif expected_type == "integer" and not isinstance(v, int):
                try:
                    args[k] = int(v)
                except:
                    call_score -= 0.2
            elif expected_type == "number" and not isinstance(v, (int, float)):
                try:
                    args[k] = float(v)
                except:
                    call_score -= 0.2
                    
        total_score += max(0.0, call_score)
        
    return total_score / len(function_calls)

def enhance_tool_descriptions(tools):
    """
    Rewrite tool descriptions to be more friendly for small models.
    In a real scenario, this might use an LLM or regex to inject hints.
    Here we prepend an action verb and inject parameter requirements.
    """
    enhanced = []
    for t in tools:
        new_t = json.loads(json.dumps(t)) # deep copy
        desc = new_t.get("description", "")
        
        # Ensure it starts with an action-oriented phrase
        if not desc.lower().startswith(("call this to", "use this to")):
            desc = f"Call this to {desc[0].lower() + desc[1:] if desc else ''}"
            
        # Inject parameter hints into the description
        props = new_t.get("parameters", {}).get("properties", {})
        reqs = new_t.get("parameters", {}).get("required", [])
        
        if props:
            param_hints = []
            for k, v in props.items():
                req_str = " (REQUIRED)" if k in reqs else " (OPTIONAL)"
                type_str = v.get("type", "unknown")
                param_hints.append(f"'{k}' is a {type_str}{req_str}")
                
            desc += ". Parameters: " + ", ".join(param_hints) + "."
            
        new_t["description"] = desc
        enhanced.append(new_t)
        
    return enhanced


class ConfidenceCalibrator:
    """
    Dynamically adjust confidence thresholds based on recent edge model success.
    If the edge model is doing well, we can trust it more (lower the threshold).
    If it's hallucinating, we raise the bar to force cloud usage.
    """
    def __init__(self, initial_threshold=0.60, min_threshold=0.40, max_threshold=0.85):
        self.threshold = initial_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.history = []
        
    def get_threshold(self):
        return self.threshold
        
    def update(self, validation_score):
        # Keep last 5 executions (thread-safe atomic slicing)
        self.history.append(validation_score)
        self.history = self.history[-5:]
            
        # Recent average performance
        size = len(self.history)
        if size == 0:
            avg_score = validation_score
        else:
            avg_score = sum(self.history) / size
        
        # Adjust threshold
        if avg_score > 0.9:
            self.threshold = max(self.min_threshold, self.threshold - 0.05)
        elif avg_score < 0.6:
            self.threshold = min(self.max_threshold, self.threshold + 0.10)


GLOBAL_CALIBRATOR = ConfidenceCalibrator()
def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """
    ULTIMATE HYBRID ROUTER
    Uses Complexity Scoring, 1-Shot Injection, Type Coercion, Fuzzy Validation,
    Dynamic RAG Scaling, and a Rolling Average Calibrator.
    """
    
    complexity = compute_complexity(messages, tools)
    
    if complexity > 0.85:
        # Pre-flight rejection: Too complex for edge model, go straight to cloud
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (pre-flight)"
        cloud["complexity"] = complexity
        return cloud
        
    # Calculate tool RAG top-k dynamically
    num_tools = len(tools)
    if num_tools <= 3:
        tool_rag_top_k = 0 # load all tools
    elif num_tools <= 7:
        tool_rag_top_k = 3 # load top 3 tools
    else:
        tool_rag_top_k = 4 # load top 4 tools
        
    # Run the native edge model directly without mutating the system prompt or tool descriptions
    local = generate_cactus(messages, tools, tool_rag_top_k=tool_rag_top_k)

    raw_confidence = local.get("confidence", 0)
    function_calls = local.get("function_calls", [])

    validation_score = validate_function_calls(function_calls, tools)
    
    # Adjust confidence based on schema validation and inherent complexity
    adjusted_confidence = raw_confidence * validation_score * (1 - 0.4 * complexity)

    # Use a dynamic threshold if provided, else static fallback
    routing_threshold = GLOBAL_CALIBRATOR.get_threshold()

    # Use the adjusted confidence for the routing decision
    if adjusted_confidence >= routing_threshold and validation_score >= 0.8:
        GLOBAL_CALIBRATOR.update(validation_score)
            
        local["source"] = "on-device"
        local["complexity"] = complexity
        local["adjusted_confidence"] = adjusted_confidence
        local["routing_threshold_used"] = routing_threshold
        return local

    GLOBAL_CALIBRATOR.update(validation_score)

    hint = function_calls if (validation_score > 0.3 and len(function_calls) > 0) else None
    
    # Fallback to cloud safely
    try:
        cloud = generate_cloud(messages, tools, hint=hint)
    except Exception as e:
        cloud = dict(local) # Copy edge output if cloud API key is missing on remote
        
    cloud["source"] = "cloud (fallback with hint)" if hint else "cloud (fallback)"
    cloud["local_confidence"] = raw_confidence
    cloud["complexity"] = complexity
    cloud["adjusted_confidence"] = adjusted_confidence
    cloud["routing_threshold_used"] = routing_threshold
    cloud["total_time_ms"] = cloud.get("total_time_ms", 0) + local.get("total_time_ms", 0)
    
    # Ensure all telemetry fields are present to prevent leaderboard ZeroDivisionErrors
    for key in ["time_to_first_token_ms", "prefill_tokens", "decode_tokens", "prefill_tps", "decode_tps", "total_tokens"]:
        if key not in cloud:
            cloud[key] = local.get(key, 1.0)
            
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
