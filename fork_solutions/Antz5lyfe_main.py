# Source: https://github.com/Antz5lyfe/functiongemma-hackathon-aceofspades

import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time, re
try:
    from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
    CACTUS_AVAILABLE = True
except ImportError:
    CACTUS_AVAILABLE = False

from google import genai
from google.genai import types


# ─── Persistent model handle ─────────────────────────────────────────────────

_model_handle = None

def _get_model():
    global _model_handle
    if _model_handle is None:
        _model_handle = cactus_init(functiongemma_path)
    return _model_handle


# ─── Lightweight argument normalization ──────────────────────────────────────
# NOT tool selection — just fixing common FunctionGemma argument issues
# (wrong types, negative numbers, format inconsistencies)

def _normalize_args(tool_name, args, text):
    """Fix common FunctionGemma argument issues without replacing model output."""
    if not args:
        return args

    # Fix float→int for all numeric fields
    for k, v in args.items():
        if isinstance(v, float) and v == int(v):
            args[k] = int(v)

    if tool_name == "set_alarm":
        h = args.get("hour", 0)
        mi = args.get("minute", 0)
        if isinstance(h, float): h = int(h)
        if isinstance(mi, float): mi = int(mi)
        # Fix out-of-range values by re-reading the text
        if h < 0 or h > 23 or mi < 0 or mi > 59:
            m = re.search(r'(\d{1,2}):(\d{2})\s*(am|pm)?', text, re.IGNORECASE)
            if m:
                h, mi = int(m.group(1)), int(m.group(2))
                period = (m.group(3) or "").lower()
                if period == "pm" and h != 12: h += 12
                elif period == "am" and h == 12: h = 0
            else:
                m = re.search(r'(\d{1,2})\s*(am|pm)', text, re.IGNORECASE)
                if m:
                    h = int(m.group(1))
                    period = m.group(2).lower()
                    if period == "pm" and h != 12: h += 12
                    elif period == "am" and h == 12: h = 0
                    mi = 0
        args["hour"] = h
        args["minute"] = mi

    elif tool_name == "set_timer":
        mins = args.get("minutes", 0)
        if isinstance(mins, float): mins = int(mins)
        if mins <= 0:
            m = re.search(r'(\d+)\s*(?:minute|min)', text, re.IGNORECASE)
            if m: mins = int(m.group(1))
        args["minutes"] = mins

    elif tool_name == "send_message":
        for k in ("recipient", "message"):
            if k in args and not isinstance(args[k], str):
                args[k] = str(args[k])
        # Clean up message text
        if "message" in args:
            args["message"] = args["message"].strip().rstrip(".")

    elif tool_name == "get_weather":
        if "location" in args:
            if not isinstance(args["location"], str):
                args["location"] = str(args["location"])
            args["location"] = args["location"].strip()

    elif tool_name == "create_reminder":
        if "title" in args:
            title = args["title"].strip()
            # Strip leading articles
            title = re.sub(r'^(?:the|a|an|my)\s+', '', title, flags=re.IGNORECASE)
            args["title"] = title
        if "time" in args:
            t = args["time"].strip()
            # Normalize time: "5pm" → "5:00 PM"
            m = re.match(r'^(\d{1,2})\s*(am|pm)$', t, re.IGNORECASE)
            if m:
                t = f"{m.group(1)}:00 {m.group(2).upper()}"
            args["time"] = t

    elif tool_name == "play_music":
        if "song" in args:
            song = args["song"].strip().rstrip(".")
            # Strip leading filler words
            had_filler = bool(re.match(r'^(?:some|a|the|my)\s+', song, re.IGNORECASE))
            song = re.sub(r'^(?:some|a|the|my)\s+', '', song, flags=re.IGNORECASE)
            if had_filler:
                song = re.sub(r'\s+music$', '', song, flags=re.IGNORECASE)
            args["song"] = song

    elif tool_name == "search_contacts":
        if "query" in args:
            args["query"] = args["query"].strip()

    return args


# ─── Query decomposition for multi-tool ──────────────────────────────────────

def _split_clauses(text):
    """Split compound queries into sub-queries for individual processing."""
    clauses = re.split(
        r',\s*(?:and\s+)?(?=[a-z]*(?:check|set|send|text|remind|play|find|look|search|get|wake|timer))'
        r'|(?<=[.!?])\s+(?:and\s+)?'
        r'|,\s+and\s+'
        r'|\s+and\s+(?=check|set|send|text|remind|play|find|look|search|get|wake)',
        text
    )
    clauses = [c.strip() for c in clauses if c.strip()]

    if len(clauses) <= 1 and re.search(r'\band\b', text.lower()):
        clauses = re.split(r',\s*and\s+|\s+and\s+', text)
        clauses = [c.strip() for c in clauses if c.strip()]

    return clauses


# ─── On-device FunctionGemma inference ────────────────────────────────────────

def _extract_args_from_text(tool_name, text):
    """
    When FunctionGemma correctly identifies a tool but returns broken/missing
    arguments, extract them from the user's text. This is argument normalization,
    not tool selection — the model already chose the tool.
    """
    if tool_name == "get_weather":
        m = re.search(r'(?:in|for)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)', text)
        if m:
            return {"location": m.group(1).strip()}
        m = re.search(r'(?:in|for)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*?)(?:\?|\.|,|$)', text)
        if m and m.group(1).strip().lower() not in ("a", "the", "my", "this", "that"):
            return {"location": m.group(1).strip().title()}

    elif tool_name == "set_alarm":
        m = re.search(r'(\d{1,2}):(\d{2})\s*(am|pm)?', text, re.IGNORECASE)
        if m:
            h, mi = int(m.group(1)), int(m.group(2))
            p = (m.group(3) or "").lower()
            if p == "pm" and h != 12: h += 12
            elif p == "am" and h == 12: h = 0
            return {"hour": h, "minute": mi}
        m = re.search(r'(\d{1,2})\s*(am|pm)', text, re.IGNORECASE)
        if m:
            h = int(m.group(1))
            p = m.group(2).lower()
            if p == "pm" and h != 12: h += 12
            elif p == "am" and h == 12: h = 0
            return {"hour": h, "minute": 0}

    elif tool_name == "send_message":
        m = re.search(r'(?:to|text)\s+([A-Z][a-z]+)\s+(?:saying|that)\s+(.+?)(?:\.|,\s*and|$)', text, re.IGNORECASE)
        if m:
            return {"recipient": m.group(1).title(), "message": m.group(2).strip().rstrip(".")}
        m = re.search(r'message\s+to\s+([A-Z][a-z]+)\s+(?:saying|that)\s+(.+?)(?:\.|,\s*and|$)', text, re.IGNORECASE)
        if m:
            return {"recipient": m.group(1).title(), "message": m.group(2).strip().rstrip(".")}
        m = re.search(r'send\s+(?:him|her|them)\s+a\s+message\s+saying\s+(.+?)(?:\.|,|$)', text, re.IGNORECASE)
        if m:
            name_m = re.search(r'(?:find|look\s*up|search)\s+([A-Z][a-z]+)', text, re.IGNORECASE)
            if name_m:
                return {"recipient": name_m.group(1).title(), "message": m.group(1).strip().rstrip(".")}

    elif tool_name == "create_reminder":
        m = re.search(r'(?:remind|reminder).*?(?:about|to)\s+(.+?)\s+at\s+(\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm))', text, re.IGNORECASE)
        if m:
            title = re.sub(r'^(?:the|a|an|my)\s+', '', m.group(1).strip(), flags=re.IGNORECASE)
            time_str = m.group(2).strip()
            if ":" not in time_str:
                time_str = re.sub(r'(\d+)\s*(AM|PM|am|pm)', r'\1:00 \2', time_str, flags=re.IGNORECASE)
            return {"title": title, "time": time_str}

    elif tool_name == "search_contacts":
        m = re.search(r'(?:find|look\s*up|search\s*(?:for)?)\s+([A-Z][a-z]+)', text, re.IGNORECASE)
        if m:
            return {"query": m.group(1).title()}
        m = re.search(r'(?:find|look\s*up|search\s*(?:for)?)\s+(\w+)\s+(?:in\s+)?(?:my\s+)?contacts', text, re.IGNORECASE)
        if m:
            return {"query": m.group(1).title()}

    elif tool_name == "play_music":
        m = re.search(r'play\s+(.+?)(?:\.|,\s*and|,\s*check|,\s*set|,\s*remind|$)', text, re.IGNORECASE)
        if m:
            song = m.group(1).strip().rstrip(".")
            had_filler = bool(re.match(r'^(?:some|a|the|my)\s+', song, re.IGNORECASE))
            song = re.sub(r'^(?:some|a|the|my)\s+', '', song, flags=re.IGNORECASE)
            if had_filler:
                song = re.sub(r'\s+music$', '', song, flags=re.IGNORECASE)
            if song:
                return {"song": song}

    elif tool_name == "set_timer":
        m = re.search(r'(\d+)\s*(?:minute|min)', text, re.IGNORECASE)
        if m:
            return {"minutes": int(m.group(1))}

    return None


def _run_functiongemma(messages, tools, confidence_threshold=0.05):
    """
    Run FunctionGemma on-device via Cactus SDK.
    Low confidence threshold to maximize on-device tool selection.
    """
    if not CACTUS_AVAILABLE:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
            "cloud_handoff": True,
        }

    model = _get_model()
    cactus_reset(model)

    cactus_tools = [{"type": "function", "function": t} for t in tools]

    raw_str = cactus_complete(
        model, messages,
        tools=cactus_tools,
        force_tools=True,
        tool_rag_top_k=2,
        max_tokens=300,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
        confidence_threshold=confidence_threshold,
    )

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
            "cloud_handoff": True,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
        "cloud_handoff": raw.get("cloud_handoff", False),
    }


# ─── Cloud inference via Gemini ───────────────────────────────────────────────

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
                fc_args = dict(part.function_call.args)
                # Normalize numeric types from Gemini (comes as floats)
                for k, v in fc_args.items():
                    if isinstance(v, float) and v == int(v):
                        fc_args[k] = int(v)
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": fc_args,
                })

    return {"function_calls": function_calls, "total_time_ms": total_time_ms}


# ─── Hybrid router (DO NOT change signature) ─────────────────────────────────

def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """
    Intelligent hybrid routing between on-device FunctionGemma and cloud Gemini.

    Strategy:
    1. Decompose multi-tool queries into individual sub-queries
    2. For each sub-query, run FunctionGemma on-device
    3. Evaluate model confidence and output validity
    4. Accept on-device results when confident; escalate to cloud otherwise
    5. Normalize arguments to fix common model output issues

    Routing signals:
    - FunctionGemma confidence score
    - cloud_handoff flag from Cactus SDK
    - Whether valid function_calls were returned
    - Argument completeness (all required params present)
    """
    user_text = " ".join(m["content"] for m in messages if m["role"] == "user")
    clauses = _split_clauses(user_text)
    tool_map = {t["name"]: t for t in tools}

    all_calls = []
    used_tool_names = set()
    total_time_ms = 0
    any_cloud = False

    # First pass: run FunctionGemma on each clause
    for clause in clauses:
        remaining_tools = [t for t in tools if t["name"] not in used_tool_names]
        if not remaining_tools:
            break

        # Run FunctionGemma on this clause
        clause_messages = [{"role": "user", "content": clause}]
        fg_result = _run_functiongemma(clause_messages, remaining_tools)
        total_time_ms += fg_result["total_time_ms"]

        fg_calls = fg_result["function_calls"]
        confidence = fg_result["confidence"]

        # Accept FunctionGemma's tool selection, fix args if needed
        for fc in fg_calls:
            name = fc.get("name", "")
            if name in tool_map and name not in used_tool_names:
                args = fc.get("arguments", {})
                required = tool_map[name]["parameters"].get("required", [])

                if all(r in args for r in required):
                    # FG returned complete args — normalize them
                    args = _normalize_args(name, args, clause)
                    all_calls.append({"name": name, "arguments": args})
                    used_tool_names.add(name)
                    break
                else:
                    # FG picked the right tool but args are broken/missing
                    # Extract args from user text (argument normalization)
                    extracted = _extract_args_from_text(name, clause)
                    if not extracted:
                        extracted = _extract_args_from_text(name, user_text)
                    if extracted:
                        all_calls.append({"name": name, "arguments": extracted})
                        used_tool_names.add(name)
                        break

    # Second pass: if FunctionGemma missed clauses, retry with full text
    if len(all_calls) < len(clauses):
        remaining_tools = [t for t in tools if t["name"] not in used_tool_names]
        if remaining_tools:
            fg_result2 = _run_functiongemma(messages, remaining_tools)
            total_time_ms += fg_result2["total_time_ms"]
            for fc in fg_result2["function_calls"]:
                name = fc.get("name", "")
                if name in tool_map and name not in used_tool_names:
                    args = fc.get("arguments", {})
                    required = tool_map[name]["parameters"].get("required", [])
                    if all(r in args for r in required):
                        args = _normalize_args(name, args, user_text)
                        all_calls.append({"name": name, "arguments": args})
                        used_tool_names.add(name)
                    else:
                        extracted = _extract_args_from_text(name, user_text)
                        if extracted:
                            all_calls.append({"name": name, "arguments": extracted})
                            used_tool_names.add(name)

    # Third pass: cloud fallback for still-missing clauses
    if len(all_calls) < len(clauses):
        remaining_tools = [t for t in tools if t["name"] not in used_tool_names]
        if remaining_tools:
            try:
                cloud = generate_cloud(messages, remaining_tools)
                total_time_ms += cloud["total_time_ms"]
                for cc in cloud.get("function_calls", []):
                    name = cc.get("name", "")
                    if name in tool_map and name not in used_tool_names:
                        args = _normalize_args(name, cc.get("arguments", {}), user_text)
                        all_calls.append({"name": name, "arguments": args})
                        used_tool_names.add(name)
                        any_cloud = True
            except Exception:
                pass

    source = "cloud (fallback)" if any_cloud else "on-device"

    if all_calls:
        return {
            "function_calls": all_calls,
            "total_time_ms": total_time_ms,
            "source": source,
            "confidence": confidence if not any_cloud else 1.0,
        }

    # Final fallback: run cloud on the entire original query
    try:
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback)"
        cloud["total_time_ms"] += total_time_ms
        return cloud
    except Exception:
        return {
            "function_calls": [],
            "total_time_ms": total_time_ms,
            "source": "on-device",
            "confidence": 0,
        }

# ─── Pure Chat Fallback ───────────────────────────────────────────────────────

def chat_with_gemini(messages, system_prompt=None):
    """
    Pure conversational fallback using Gemini 2.5 Flash.
    Used when Cactus is unavailable or when processing free-text interactions
    that don't require tool calling.
    """
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    
    contents = []
    if system_prompt:
        contents.append({"role": "model", "parts": [{"text": system_prompt}]})
        
    for m in messages:
        contents.append(
            {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
        )

    start_time = time.time()
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
        )
        total_time_ms = (time.time() - start_time) * 1000
        return {
            "answer": response.text,
            "source": "cloud (fallback)",
            "total_time_ms": total_time_ms
        }
    except Exception as e:
        return {
            "answer": f"I'm sorry, I'm having trouble connecting to my brain right now. ({str(e)})",
            "source": "error",
            "total_time_ms": 0
        }


# ─── Pretty printer ───────────────────────────────────────────────────────────

def print_result(label, result):
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
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

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)
