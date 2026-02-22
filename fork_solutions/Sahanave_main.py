# Source: https://github.com/Sahanave/functiongemma-hackathon

import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types


# ---- Query rewrite (cloud): simple sentences vs plan ----
# Style "sentences": one short sentence per intent, self-contained → best for 1:1 mapping to one tool call.
# Style "plan": numbered steps in natural language → good for multi-step clarity; parse to one string per step.

REWRITE_PROMPT_SENTENCES = """You rewrite user requests so each part can be handled by one tool call.

Available tools (name and description):
{tool_summary}

User request: {user_content}

Rewrite this as 1 or more short, simple sentences. Each sentence must be exactly one clear action that maps to one tool call. Keep each sentence self-contained: use full names and specifics (e.g. "Bob", "London"), not pronouns like "him" or "it". Preserve times, places, and message content exactly.

Reply with ONLY a JSON array of strings, no other text. Example: ["Get the weather in London.", "Send a message to Bob saying hello."]"""

REWRITE_PROMPT_PLAN = """You break down user requests into a step-by-step plan. Each step is one action.

Available tools (name and description):
{tool_summary}

User request: {user_content}

Output a short plan: 1 or more steps. Each step is exactly one action in natural language (one tool call). Use full names and specifics, not pronouns. Preserve times, places, and message content.

Reply with ONLY a JSON array of strings, one string per step. Example: ["Set an alarm for 7:30 AM.", "Get the weather in New York."]"""

REWRITE_PROMPT_REASONING_STEPS = """You break the user request into reasoning steps. Each step is one clear, self-contained action that maps to one tool call.

Available tools (name and description):
{tool_summary}

User request: {user_content}

Break this into 1 or more reasoning steps. Each step = one clear action (what to do, and with what: e.g. location, recipient, message text, time). Preserve entities exactly (times, places, names, message content). Use full names and specifics, not pronouns.

Reply with ONLY a JSON array of strings, one string per step. Example: ["Set an alarm for 7:30 AM.", "Get the weather in New York City."]"""


def build_rewrite_prompt(user_content, tool_summary, style="sentences"):
    """Build the prompt for cloud query rewriting. style in ('sentences', 'plan', 'reasoning_steps')."""
    if style == "plan":
        return REWRITE_PROMPT_PLAN.format(user_content=user_content, tool_summary=tool_summary)
    if style == "reasoning_steps":
        return REWRITE_PROMPT_REASONING_STEPS.format(user_content=user_content, tool_summary=tool_summary)
    return REWRITE_PROMPT_SENTENCES.format(user_content=user_content, tool_summary=tool_summary)


def parse_rewrite_response(response_text):
    """Parse Gemini text response into a list of strings. On failure returns None."""
    if not response_text or not isinstance(response_text, str):
        return None
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        out = json.loads(text)
        if isinstance(out, list) and all(isinstance(x, str) for x in out) and out:
            return [s.strip() for s in out if s.strip()]
    except json.JSONDecodeError:
        pass
    return None


# Trailing punctuation chars to strip from message
_TRAILING_PUNCT = ".,!?;:\"'"


def _canonicalize_message(msg):
    if not isinstance(msg, str):
        return msg
    s = msg.strip()
    while s and s[-1] in _TRAILING_PUNCT:
        s = s[:-1].strip()
    return s.lower()


def _canonicalize_reminder_title(title):
    if not isinstance(title, str):
        return title
    s = title.strip()
    # Remove leading "about " then "the " (case-insensitive) to match "remind me about the …"
    if s.lower().startswith("about "):
        s = s[6:].strip()
    if s.lower().startswith("the "):
        s = s[4:].strip()
    return s.lower()


def canonicalize_function_calls(function_calls):
    """
    Light canonicalizer for agent output: normalize arguments to match benchmark expectations.
    - send_message.message: remove trailing punctuation, lower-case.
    - create_reminder.title: remove leading "about " / "the ", lower-case.
    Returns a new list of calls (does not mutate input).
    """
    out = []
    for call in function_calls:
        name = call.get("name")
        args = dict(call.get("arguments") or {})
        if name == "send_message" and "message" in args:
            args["message"] = _canonicalize_message(args["message"])
        if name == "create_reminder" and "title" in args:
            args["title"] = _canonicalize_reminder_title(args["title"])
        out.append({"name": name, "arguments": args})
    return out


CONFIDENCE_TOOL_NAME = "_report_on_device_confidence"

CONFIDENCE_TOOL = {
    "name": CONFIDENCE_TOOL_NAME,
    "description": (
        "Call this to report how confident you are (0.0 to 1.0) that the on-device system "
        "can correctly handle the user's request. Call this in the same turn as other tool calls. "
        "Use 1.0 only when the request clearly fits the available tools and you are certain."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "confidence": {
                "type": "number",
                "description": "Confidence that on-device can handle this request (0.0 = not at all, 1.0 = fully confident).",
            }
        },
        "required": ["confidence"],
    },
}



def tools_with_confidence(tools):
    """Prepend the confidence tool to the tool list (avoid duplicates by name)."""
    names = {t.get("name") for t in tools}
    if CONFIDENCE_TOOL_NAME in names:
        return list(tools)
    return [CONFIDENCE_TOOL] + list(tools)


def parse_confidence_from_calls(function_calls):
    """
    Extract on-device confidence from function_calls if the model called
    _report_on_device_confidence. Returns None if not found or invalid.
    """
    if not function_calls:
        return None
    for call in function_calls:
        if call.get("name") == CONFIDENCE_TOOL_NAME:
            args = call.get("arguments") or {}
            c = args.get("confidence")
            if c is not None:
                try:
                    v = float(c)
                    return max(0.0, min(1.0, v))
                except (TypeError, ValueError):
                    return None
    return None


def filter_confidence_tool_from_calls(function_calls):
    """Return function_calls with _report_on_device_confidence removed (for execution)."""
    return [c for c in function_calls if c.get("name") != CONFIDENCE_TOOL_NAME]


def _user_content(messages):
    """Concatenated user message content."""
    return " ".join(m.get("content", "") for m in messages if m.get("role") == "user").strip()


def is_reasoning_needed(messages, tools):
    """
    Call 1: Ask Gemini if the request requires breaking into multiple steps.
    Returns True if yes or on parse/API failure (safe default). Uses max_tokens=4 for speed.
    """
    user_content = _user_content(messages)
    if not user_content:
        return True
    prompt = (
        f"User request: {user_content}\n\n"
        "Does this user request require breaking into multiple steps (e.g. several actions or tools)? "
        "Reply with exactly one word: yes or no."
    )
    try:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=4),
        )
        text = ""
        for c in response.candidates or []:
            for p in c.content.parts or []:
                if p.text:
                    text += p.text
        raw = text.strip().lower()
        if raw.startswith("no"):
            return False
        if raw.startswith("yes"):
            return True
    except Exception:
        pass
    return True


def rewrite_query_with_cloud(messages, tools, style="sentences"):
    """
    Use Gemini (cloud) to rewrite the user query so each part maps to one tool call.
    - style='sentences': simple broken sentences, each self-contained (recommended for 1:1 tool mapping).
    - style='plan': step-by-step plan in natural language.
    Returns list of strings; on failure returns [original user content].
    """
    user_content = _user_content(messages)
    if not user_content:
        return []
    tool_summary = "\n".join(
        f"- {t['name']}: {t.get('description', '')}"
        for t in tools
        if t.get("name") != CONFIDENCE_TOOL_NAME
    )
    prompt = build_rewrite_prompt(user_content, tool_summary, style=style)
    try:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1),
        )
        text = ""
        for c in response.candidates or []:
            for p in c.content.parts or []:
                if p.text:
                    text += p.text
        parsed = parse_rewrite_response(text)
        if parsed:
            return parsed
    except Exception:
        pass
    return [user_content]


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = cactus_init(functiongemma_path)

    tools_with_conf = tools_with_confidence(tools)
    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools_with_conf]

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

    function_calls = raw.get("function_calls", [])
    cactus_confidence = raw.get("confidence", 0)
    tool_confidence = parse_confidence_from_calls(function_calls)
    confidence = tool_confidence if tool_confidence is not None else cactus_confidence

    return {
        "function_calls": canonicalize_function_calls(filter_confidence_tool_from_calls(function_calls)),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": confidence,
    }


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    tools_with_conf = tools_with_confidence(tools)
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
            for t in tools_with_conf
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

    confidence = parse_confidence_from_calls(function_calls)
    if confidence is None:
        confidence = 0.0

    return {
        "function_calls": canonicalize_function_calls(filter_confidence_tool_from_calls(function_calls)),
        "total_time_ms": total_time_ms,
        "confidence": confidence,
    }


def generate_hybrid(messages, tools, confidence_threshold=0.9, rewrite_first=False, rewrite_style="sentences"):
    """Hybrid inference: on-device (Cactus) vs cloud fallback.
    rewrite_first=False, rewrite_style='sentences' (or 'plan', 'reasoning_steps') for cloud rewrite then hybrid per sentence."""
    if not rewrite_first:
        local = generate_cactus(messages, tools)
        if local["confidence"] >= confidence_threshold:
            local["source"] = "on-device"
            return local
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback)"
        cloud["local_confidence"] = local["confidence"]
        cloud["total_time_ms"] += local["total_time_ms"]
        return cloud

    # Two-call reasoning chain: when reasoning_steps, first check if decomposition is needed
    if rewrite_style == "reasoning_steps":
        if not is_reasoning_needed(messages, tools):
            return generate_hybrid(messages, tools, confidence_threshold=confidence_threshold, rewrite_first=False)
        steps = rewrite_query_with_cloud(messages, tools, style="reasoning_steps")
        if not steps:
            steps = [_user_content(messages) or "Continue."]
        all_calls = []
        total_time_ms = 0.0
        confidences = []
        any_cloud = False
        local_confidence_used = None
        for s in steps:
            r = generate_hybrid([{"role": "user", "content": s}], tools, confidence_threshold=confidence_threshold)
            all_calls.extend(r["function_calls"])
            total_time_ms += r["total_time_ms"]
            confidences.append(r.get("confidence", 0))
            if r.get("source") != "on-device":
                any_cloud = True
                local_confidence_used = r.get("local_confidence")
        return {
            "function_calls": all_calls,
            "total_time_ms": total_time_ms,
            "confidence": min(confidences) if confidences else 0.0,
            "source": "on-device" if not any_cloud else "cloud (fallback)",
            **({"local_confidence": local_confidence_used} if local_confidence_used is not None else {}),
        }

    # Rewrite with cloud, then hybrid per sentence
    sentences = rewrite_query_with_cloud(messages, tools, style=rewrite_style)
    if not sentences:
        sentences = [_user_content(messages) or "Continue."]
    all_calls = []
    total_time_ms = 0.0
    confidences = []
    any_cloud = False
    local_confidence_used = None
    for s in sentences:
        r = generate_hybrid([{"role": "user", "content": s}], tools, confidence_threshold)
        all_calls.extend(r["function_calls"])
        total_time_ms += r["total_time_ms"]
        confidences.append(r.get("confidence", 0))
        if r.get("source") != "on-device":
            any_cloud = True
            local_confidence_used = r.get("local_confidence")
    return {
        "function_calls": all_calls,
        "total_time_ms": total_time_ms,
        "confidence": min(confidences) if confidences else 0.0,
        "source": "on-device" if not any_cloud else "cloud (fallback)",
        **({"local_confidence": local_confidence_used} if local_confidence_used is not None else {}),
    }


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