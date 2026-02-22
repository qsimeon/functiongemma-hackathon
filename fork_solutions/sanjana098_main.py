# Source: https://github.com/sanjana098/functiongemma-hackathon
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time, re, random
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types

# --- INLINED ROUTER CLASS FOR SUBMISSION ---
class CactusRouter:
    def __init__(self, confidence_threshold=0.7, battery_min=20):
        self.confidence_threshold = confidence_threshold
        self.battery_min = battery_min
        # Privacy Gate: Keywords that MUST stay local
        self.sensitive_keywords = [
            "password", "ssn", "credit card", "medical", "health", 
            "private", "confidential", "secret", "bank", "diagnosis"
        ]
        # Complexity Gate: Keywords that imply heavy reasoning (Cloud)
        self.complexity_keywords = [
            "compare multiple", "comprehensive analysis", "synthesize all", 
            "global search", "web search", "historical data"
        ]

    def get_system_vitals(self):
        """
        Simulates checking device state. 
        In a real app, use `psutil` to get battery/network status.
        For Hackathon: We simulate 'Good Condition' to prefer Local.
        """
        return {
            "battery_level": 85,  # High battery -> Prefer Local
            "is_charging": True,
            "network_status": "online"
        }

    def check_privacy_risk(self, prompt):
        """Rule 1: If sensitive data is present, FORCE LOCAL."""
        prompt_lower = prompt.lower()
        for word in self.sensitive_keywords:
            if word in prompt_lower:
                return True, word
        return False, None

    def check_complexity(self, prompt):
        """Rule 2: If task is too complex, SUGGEST CLOUD."""
        prompt_lower = prompt.lower()
        if len(prompt) > 1000: # Long context
            return True, "length"
        for word in self.complexity_keywords:
            if word in prompt_lower:
                return True, word
        return False, None

    def decide_route(self, prompt, model_confidence=None):
        """
        Decides whether to route to 'LOCAL' (FunctionGemma) or 'CLOUD' (Gemini).
        Returns: (route, reason)
        """
        vitals = self.get_system_vitals()
        
        # 1. Privacy Check (Hard Gate)
        is_sensitive, keyword = self.check_privacy_risk(prompt)
        if is_sensitive:
            return "LOCAL", f"Privacy Shield: Detected '{keyword}'"

        # 2. Battery Check (Hard Gate)
        if vitals["battery_level"] < self.battery_min and not vitals["is_charging"]:
            return "CLOUD", "Battery Saver Mode"

        # 3. Complexity Check (Soft Gate)
        is_complex, factor = self.check_complexity(prompt)
        if is_complex:
            # If complex but confidence is high (passed in), we might still stay local
            if model_confidence and model_confidence > 0.95:
                 return "LOCAL", "High Confidence Override"
            return "CLOUD", f"Complexity Detected: '{factor}'"

        # 4. Default to Local (Hackathon Goal: Maximize Edge Ratio)
        if model_confidence is not None:
             if model_confidence < self.confidence_threshold:
                 return "CLOUD", f"Low Confidence ({model_confidence:.2f})"
        
        return "LOCAL", "Standard Local Execution"

# --- END ROUTER CLASS ---

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


def generate_hybrid(messages, tools, confidence_threshold=0.7):
    """
    Intelligent Hybrid Routing Strategy:
    1. Check Privacy/Battery/Complexity (Pre-Compute Router).
    2. If Router says CLOUD -> Skip Local, Go Cloud.
    3. If Router says LOCAL -> Run Local.
    4. If Local Confidence < Threshold -> Fallback to Cloud (unless Privacy overrides).
    """
    
    # 1. Extract User Prompt
    user_prompt = ""
    for m in messages:
        if m["role"] == "user":
            user_prompt += m["content"] + " "
            
    # 2. Initialize Router
    router = CactusRouter(confidence_threshold=confidence_threshold)
    
    # 3. Pre-Inference Decision
    route_decision, reason = router.decide_route(user_prompt)
    
    # Logic: If Router explicitly requests CLOUD (e.g. Battery/Complexity), obey immediately.
    if route_decision == "CLOUD":
        cloud = generate_cloud(messages, tools)
        cloud["source"] = f"cloud ({reason})"
        cloud["local_confidence"] = 0.0
        return cloud

    # 4. Run Local Inference (FunctionGemma)
    local = generate_cactus(messages, tools)
    local_conf = local.get("confidence", 0.0)
    
    # 5. Check Confidence / Post-Inference Fallback
    # If Privacy forced us Local, we STICK to Local even if confidence is low.
    is_privacy_risk, keyword = router.check_privacy_risk(user_prompt)
    
    if is_privacy_risk:
        local["source"] = f"on-device (Privacy Force: {keyword})"
        return local

    # Normal Confidence Check
    if local_conf >= confidence_threshold:
        local["source"] = f"on-device ({reason})"
        return local

    # Fallback to Cloud if not private and low confidence
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (Low Confidence Fallback)"
    cloud["local_confidence"] = local_conf
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
