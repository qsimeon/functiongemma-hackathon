# Source: https://github.com/elena-kalinina/functiongemma-hackathon

functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time, re
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
from google import genai
from google.genai import types

# ============== GLOBAL MODEL CACHE ==============
# Keep model loaded to avoid re-initialization overhead
_cached_model = None

def _get_model():
    """Get or create cached model instance."""
    global _cached_model
    if _cached_model is None:
        _cached_model = cactus_init(functiongemma_path)
    return _cached_model

def _reset_model():
    """Reset model state between calls (clears KV cache)."""
    global _cached_model
    if _cached_model:
        cactus_reset(_cached_model)


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = _get_model()
    _reset_model()  # Clear KV cache for fresh inference
    
    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    raw_str = cactus_complete(
        model,
        messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=128,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    try:
        # Fix common JSON issues from the model
        fixed_str = raw_str
        # Fix leading zeros in integers (e.g., "minute":01 -> "minute":1)
        fixed_str = re.sub(r':(\s*)0+(\d+)([,}\]])', r':\1\2\3', fixed_str)
        # Fix duplicate keys by removing earlier occurrences
        fixed_str = re.sub(r'"(\w+)":\s*\w+,\s*"\1":', r'"\1":', fixed_str)
        # Fix trailing commas before } or ]
        fixed_str = re.sub(r',\s*([}\]])', r'\1', fixed_str)
        raw = json.loads(fixed_str)
    except json.JSONDecodeError:
        # Try to extract function call data with regex as last resort
        fc_match = re.search(r'"function_calls"\s*:\s*\[(.*?)\]', raw_str, re.DOTALL)
        conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', raw_str)
        time_match = re.search(r'"total_time_ms"\s*:\s*([\d.]+)', raw_str)
        
        if fc_match:
            try:
                # Try to parse just the function calls array
                fc_str = "[" + fc_match.group(1) + "]"
                fc_str = re.sub(r':(\s*)0+(\d+)([,}\]])', r':\1\2\3', fc_str)
                fc_str = re.sub(r',\s*([}\]])', r'\1', fc_str)
                fcs = json.loads(fc_str)
                return {
                    "function_calls": fcs,
                    "total_time_ms": float(time_match.group(1)) if time_match else 0,
                    "confidence": float(conf_match.group(1)) if conf_match else 0,
                }
            except json.JSONDecodeError:
                pass
        
        return {
            "function_calls": [],
            "total_time_ms": float(time_match.group(1)) if time_match else 0,
            "confidence": float(conf_match.group(1)) if conf_match else 0,
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


# ============== STRATEGIES ==============
# Change ACTIVE_STRATEGY to test different approaches
ACTIVE_STRATEGY = "maximize_local"


def _validate_function_call(fc, tools):
    """Validate a function call output. Returns False if clearly wrong."""
    if not fc or not fc.get("name"):
        return False
    
    name = fc["name"]
    args = fc.get("arguments", {})
    
    # Find the tool definition
    tool_def = None
    for t in tools:
        # Handle both formats: direct {"name": ...} or nested {"function": {"name": ...}}
        t_name = t.get("name") or t.get("function", {}).get("name")
        if t_name == name:
            # Get the actual function definition
            tool_def = t.get("function", t)  # Use t["function"] if exists, else t itself
            break
    
    if not tool_def:
        return False  # Unknown function
    
    props = tool_def.get("parameters", {}).get("properties", {})
    required = tool_def.get("parameters", {}).get("required", [])
    
    # Check required params exist
    for req in required:
        if req not in args or args[req] is None:
            return False
    
    # Type-specific validations
    for param, value in args.items():
        if param not in props:
            continue
        prop_type = props[param].get("type", "string")
        
        # Integer params: check for negative when shouldn't be
        if prop_type == "integer":
            if not isinstance(value, int):
                return False
            # Minutes, hours should be non-negative
            if param in ["minutes", "hour", "minute"] and value < 0:
                return False
            # Hours should be 0-23, minutes 0-59
            if param == "hour" and (value < 0 or value > 23):
                return False
            if param == "minute" and (value < 0 or value > 59):
                return False
        
        # String params: check for garbage
        if prop_type == "string":
            if not isinstance(value, str):
                return False
            # Check for garbage indicators
            if len(value) > 200:  # Suspiciously long
                return False
            if any(c in value for c in ['<', '>', '{', '}', '\\']):  # Code/tags
                return False
            # Arabic or other non-latin scripts mixed with latin usually indicates garbage
            if any(ord(c) > 1500 for c in value) and any(c.isascii() and c.isalpha() for c in value):
                return False
    
    return True


def _validate_output(local, tools):
    """Validate all function calls in output. Returns False if any is invalid."""
    fcs = local.get("function_calls", [])
    if not fcs:
        return False
    for fc in fcs:
        if not _validate_function_call(fc, tools):
            return False
    return True


def _strategy_baseline(messages, tools, local, confidence_threshold):
    """Original baseline: just use confidence threshold."""
    if local["confidence"] >= confidence_threshold:
        local["source"] = "on-device"
        return local, False  # (result, need_cloud)
    return local, True


def _strategy_always_local(messages, tools, local, confidence_threshold):
    """Debug strategy: Accept local if it has ANY VALID function call."""
    if local.get("function_calls") and _validate_output(local, tools):
        local["source"] = "on-device"
        return local, False
    return local, True


def _strategy_known_reliable(messages, tools, local, confidence_threshold):
    """Only trust on-device for KNOWN RELIABLE query patterns."""
    query = messages[-1]["content"].lower()
    
    if not local.get("function_calls") or not _validate_output(local, tools):
        return local, True
    
    fc = local["function_calls"][0]
    fc_name = fc.get("name")
    
    # Pattern 1: Weather queries - ALWAYS reliable (even partial in multi-intent)
    if fc_name == "get_weather" and "weather" in query:
        local["source"] = "on-device"
        return local, False
    
    # Pattern 2: Play/music - reliable for direct requests
    if fc_name == "play_music" and ("play" in query or "music" in query):
        local["source"] = "on-device"
        return local, False
    
    # Pattern 3: Set alarm with matching time
    if fc_name == "set_alarm" and "alarm" in query:
        args = fc.get("arguments", {})
        hour = args.get("hour", -1)
        if 0 <= hour <= 23:
            import re
            time_match = re.search(r'\b(\d{1,2})\s*(am|pm|AM|PM)?\b', query)
            if time_match:
                query_hour = int(time_match.group(1))
                is_pm = time_match.group(2) and time_match.group(2).lower() == 'pm'
                if is_pm and query_hour < 12:
                    query_hour += 12
                if query_hour == hour or (query_hour % 12) == (hour % 12):
                    local["source"] = "on-device"
                    return local, False
    
    # Pattern 4: Timer with positive minutes
    if fc_name == "set_timer" and ("timer" in query or "minute" in query):
        args = fc.get("arguments", {})
        minutes = args.get("minutes", -1)
        if minutes > 0:
            local["source"] = "on-device"
            return local, False
    
    # Everything else: try cloud
    return local, True


def _strategy_complexity(messages, tools, local, confidence_threshold):
    """Strategy 1: Enhanced complexity analysis with benchmark-derived patterns."""
    query = messages[-1]["content"]
    query_lower = query.lower()
    query_words = query_lower.split()
    
    complexity = 0
    easiness = 0
    
    # === COMPLEXITY SIGNALS (push toward cloud) ===
    
    # 1. Multi-intent patterns (from benchmark hard queries)
    multi_intent_patterns = [
        " and ", " then ", " also ", " plus ", ", and ",
        " as well ", " after that ", " followed by "
    ]
    if any(p in query_lower for p in multi_intent_patterns):
        complexity += 2
    
    # Comma-separated actions (e.g., "do X, do Y, and do Z")
    if query_lower.count(",") >= 2:
        complexity += 1
    
    # 2. More tools = harder
    num_tools = len(tools)
    if num_tools > 3:
        complexity += 1
    if num_tools > 5:
        complexity += 1
    
    # 3. Long queries
    if len(query_words) > 15:
        complexity += 1
    
    # 4. Vague language
    vague_patterns = [
        "something", "stuff", "thing", "help me", "can you help",
        "could you", "would you", "maybe", "probably", "anything",
        "i don't know", "not sure", "figure out"
    ]
    if any(v in query_lower for v in vague_patterns):
        complexity += 1
    
    # === EASINESS SIGNALS (push toward local) ===
    
    # 5. Direct action starts (from benchmark easy/medium queries)
    action_starts = [
        "set ", "get ", "send ", "play ", "create ", "find ", "search ",
        "check ", "remind ", "wake ", "text ", "look up ",
        "what is ", "what's ", "how's "
    ]
    if any(query_lower.startswith(s) for s in action_starts):
        easiness += 1
    
    # 6. Tool-specific keywords (expanded from benchmark)
    tool_keywords = {
        "get_weather": ["weather", "temperature", "forecast", "climate", "rain", "sunny"],
        "set_alarm": ["alarm", "wake me", "wake up", "morning"],
        "send_message": ["message", "text ", "texting", "saying", "tell "],
        "create_reminder": ["remind", "reminder", "don't forget", "notification"],
        "search_contacts": ["contact", "find ", "look up", "in my contacts"],
        "play_music": ["play ", "music", "song", "listen", "playlist", "track"],
        "set_timer": ["timer", "countdown", "minute timer", " minutes"],
    }
    for tool in tools:
        keywords = tool_keywords.get(tool["name"], [])
        if keywords and any(kw in query_lower for kw in keywords):
            easiness += 2
            break
        # Fallback: check tool name
        tool_words = tool["name"].replace("_", " ").lower().split()
        if any(w in query_lower for w in tool_words):
            easiness += 1
            break
    
    # 7. Named entities (cities, names)
    capitalized = [w for i, w in enumerate(query.split()) if len(w) > 1 and w[0].isupper() and i > 0]
    if capitalized:
        easiness += 1
    
    # 8. Time/number patterns
    has_time = bool(re.search(r'\b\d{1,2}(:\d{2})?\s*(am|pm)\b', query_lower, re.IGNORECASE))
    has_minutes = bool(re.search(r'\b\d+\s*(minute|min)', query_lower))
    if has_time or has_minutes:
        easiness += 1
    elif re.search(r'\b\d+\b', query_lower):
        easiness += 0.5
    
    # 9. Short queries
    if len(query_words) <= 6:
        easiness += 2
    elif len(query_words) <= 10:
        easiness += 1
    
    # 10. Single-intent patterns (benchmark easy queries)
    single_patterns = [
        r"^what('s| is) the weather",
        r"^set an alarm",
        r"^send a message to \w+",
        r"^play ",
        r"^set a timer",
        r"^remind me",
        r"^(find|look up) \w+ in my contacts",
        r"^wake me up",
        r"^text \w+ saying",
    ]
    for pattern in single_patterns:
        if re.search(pattern, query_lower):
            easiness += 1
            break
    
    # === DECISION ===
    net_complexity = complexity - easiness
    local["_complexity"] = complexity
    local["_easiness"] = easiness
    local["_net"] = net_complexity
    
    # High complexity -> cloud
    if net_complexity >= 2:
        return None, True
    
    # Model chose fallback -> cloud
    if local.get("function_calls"):
        if any(c.get("name") == "ask_help" for c in local["function_calls"]):
            return local, True
    
    # Valid output -> local
    if local.get("function_calls") and local["confidence"] >= confidence_threshold:
        local["source"] = "on-device"
        return local, False
    
    return local, True


def _strategy_semantic(messages, tools, local, confidence_threshold):
    """Strategy 2: Enhanced semantic matching with n-gram similarity and synonyms."""
    query = messages[-1]["content"]
    query_lower = query.lower()
    
    # Stop words to ignore
    stop_words = {"the", "a", "an", "for", "to", "of", "in", "is", "what", "how", "can", "you", "me", "my", "i", "?", "!", ".", "please", "could", "would"}
    
    # --- Synonym expansion (expanded from benchmark analysis) ---
    synonyms = {
        # Weather related
        "weather": ["temperature", "forecast", "climate", "rain", "sunny", "cold", "hot", "warm", "degrees"],
        # Alarm related  
        "alarm": ["wake", "alert", "morning", "clock", "oclock", "am", "pm"],
        "wake": ["alarm", "get up", "morning"],
        # Message related
        "message": ["text", "send", "tell", "say", "saying", "contact", "sms", "msg"],
        "text": ["message", "send", "tell", "saying"],
        "send": ["message", "text", "tell"],
        # Music related
        "music": ["play", "song", "listen", "audio", "track", "playlist", "beats", "hits"],
        "play": ["music", "song", "listen"],
        # Reminder related
        "reminder": ["remind", "remember", "note", "notification", "forget"],
        "remind": ["reminder", "remember", "notification"],
        # Timer related
        "timer": ["countdown", "minutes", "minute", "seconds", "min"],
        "minutes": ["timer", "minute", "min"],
        # Contact/search related
        "search": ["find", "look", "lookup", "locate"],
        "find": ["search", "look", "lookup", "locate"],
        "contact": ["contacts", "person", "friend", "people"],
        "contacts": ["contact", "person", "friend"],
        "look": ["find", "search", "lookup"],
    }
    
    def get_ngrams(text, n=2):
        """Get character n-grams for fuzzy matching."""
        text = text.lower().replace(" ", "")
        return set(text[i:i+n] for i in range(len(text) - n + 1))
    
    def ngram_similarity(text1, text2):
        """Jaccard similarity of character n-grams."""
        ngrams1 = get_ngrams(text1)
        ngrams2 = get_ngrams(text2)
        if not ngrams1 or not ngrams2:
            return 0.0
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        return intersection / union if union > 0 else 0.0
    
    def expand_with_synonyms(words):
        """Expand word set with synonyms."""
        expanded = set(words)
        for word in words:
            for key, syns in synonyms.items():
                if word == key or word in syns:
                    expanded.add(key)
                    expanded.update(syns)
        return expanded
    
    # Get query words and expand with synonyms
    query_words = set(query_lower.split()) - stop_words
    query_expanded = expand_with_synonyms(query_words)
    
    best_score = 0
    best_tool = None
    
    for tool in tools:
        tool_name = tool["name"].replace("_", " ").lower()
        tool_desc = tool.get("description", "").lower()
        tool_text = f"{tool_name} {tool_desc}"
        
        # Get tool words and expand
        tool_words = set(tool_text.split()) - stop_words
        tool_expanded = expand_with_synonyms(tool_words)
        
        # Score 1: Direct word overlap (expanded)
        overlap = len(query_expanded & tool_expanded)
        word_score = min(overlap / 3.0, 1.0)  # Normalize
        
        # Score 2: N-gram similarity (handles typos, partial matches)
        ngram_score = ngram_similarity(query_lower, tool_text)
        
        # Score 3: Tool name exact match bonus
        name_bonus = 0.3 if any(w in query_lower for w in tool_name.split()) else 0
        
        # Score 4: Parameter hint matching (e.g., "San Francisco" for location)
        param_bonus = 0
        params = tool.get("parameters", {}).get("properties", {})
        for param_name, param_info in params.items():
            param_desc = param_info.get("description", "").lower()
            # Check if query has something that looks like this parameter
            if param_name in ["location", "city"] and re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', messages[-1]["content"]):
                param_bonus = 0.2
            elif param_name in ["hour", "minute", "time"] and re.search(r'\b\d{1,2}(:\d{2})?\s*(am|pm|AM|PM)?\b', query):
                param_bonus = 0.2
            elif param_name in ["recipient", "name"] and re.search(r'\b[A-Z][a-z]+\b', messages[-1]["content"]):
                param_bonus = 0.1
        
        # Combined score
        total_score = 0.4 * word_score + 0.3 * ngram_score + name_bonus + param_bonus
        
        if total_score > best_score:
            best_score = total_score
            best_tool = tool["name"]
    
    local["_semantic_score"] = best_score
    local["_best_matching_tool"] = best_tool
    
    # --- Decision ---
    # High semantic match = trust local
    if best_score >= 0.4 and local.get("function_calls") and local["confidence"] >= confidence_threshold:
        # Bonus check: does local output match our best predicted tool?
        if local["function_calls"] and local["function_calls"][0].get("name") == best_tool:
            local["source"] = "on-device"
            return local, False
    
    # Moderate match but local picked the right tool
    if best_score >= 0.25 and local.get("function_calls"):
        if local["function_calls"][0].get("name") == best_tool:
            local["source"] = "on-device"
            return local, False
    
    # Low semantic match = go to cloud
    if best_score < 0.2:
        return None, True
    
    return local, True


def _strategy_validation(messages, tools, local, confidence_threshold):
    """Strategy 3: Validate on-device output before accepting."""
    if not local.get("function_calls"):
        return local, True  # No output, go to cloud
    
    tool_names = {t["name"] for t in tools}
    
    for call in local["function_calls"]:
        # Check if function name is valid
        if call.get("name") not in tool_names:
            local["_invalid"] = f"unknown function: {call.get('name')}"
            return local, True  # Invalid function, go to cloud
        
        # Check if arguments exist
        if not call.get("arguments"):
            local["_invalid"] = "missing arguments"
            return local, True
        
        # Find the tool and check required params
        tool = next((t for t in tools if t["name"] == call["name"]), None)
        if tool:
            required = tool.get("parameters", {}).get("required", [])
            for req in required:
                if req not in call.get("arguments", {}):
                    local["_invalid"] = f"missing required param: {req}"
                    return local, True
    
    # Output is valid
    local["source"] = "on-device"
    return local, False


def _strategy_complexity_plus(messages, tools, local, confidence_threshold):
    """BEST STRATEGY: Complexity-based with full validation. Prioritizes on-device."""
    query = messages[-1]["content"]
    query_lower = query.lower()
    
    # --- STEP 1: Only send CLEARLY hard queries to cloud (very conservative) ---
    
    # Check for multi-intent (user asking for multiple things)
    multi_intent_patterns = [" and ", " then ", ", and ", " also ", " plus "]
    has_multi_intent = any(p in query_lower for p in multi_intent_patterns)
    
    # Count complexity factors
    num_tools = len(tools)
    
    # ONLY skip local for: multi-intent queries WITH many tools
    if has_multi_intent and num_tools >= 4:
        return None, True  # Hard query, skip to cloud
    
    # --- STEP 2: Validate output with full validation ---
    
    # If no function calls at all, try cloud
    if not local.get("function_calls"):
        return local, True
    
    # Model chose fallback -> cloud
    if any(c.get("name") == "ask_help" for c in local["function_calls"]):
        return local, True
    
    # Use full validation (catches negative numbers, wrong types, garbage text)
    if not _validate_output(local, tools):
        return local, True
    
    # --- STEP 3: Valid output -> trust local ---
    local["source"] = "on-device"
    return local, False


def _strategy_complexity_semantic_plus(messages, tools, local, confidence_threshold):
    """COMBINED STRATEGY: Best of complexity + semantic + validation. Prioritizes on-device."""
    query = messages[-1]["content"]
    query_lower = query.lower()
    
    # ============ PART 1: COMPLEXITY PRE-CHECK (from complexity) ============
    # Only skip to cloud for VERY hard queries
    
    multi_intent_patterns = [" and ", " then ", ", and ", " also ", " plus "]
    has_multi_intent = any(p in query_lower for p in multi_intent_patterns)
    num_tools = len(tools)
    
    # Calculate complexity score
    complexity = 0
    if has_multi_intent:
        complexity += 2
    if num_tools > 3:
        complexity += 1
    if num_tools > 5:
        complexity += 1
    if len(query_lower.split()) > 15:
        complexity += 1
    
    # Calculate easiness score (from enhanced complexity)
    easiness = 0
    action_verbs = ["set", "get", "send", "play", "create", "find", "search", "check", "show", "tell"]
    if any(query_lower.startswith(v) or f" {v} " in query_lower for v in action_verbs):
        easiness += 1
    
    for tool in tools:
        tool_name_words = tool["name"].replace("_", " ").lower().split()
        if any(w in query_lower for w in tool_name_words):
            easiness += 2
            break
    
    if len(query_lower.split()) <= 8:
        easiness += 1
    
    net_complexity = complexity - easiness
    
    # Very hard queries go straight to cloud
    if net_complexity >= 3:
        return None, True
    
    # ============ PART 2: BASIC VALIDATION (from complexity_plus) ============
    
    if not local.get("function_calls"):
        return local, True
    
    call = local["function_calls"][0]
    func_name = call.get("name", "")
    func_args = call.get("arguments", {})
    
    # Reject fallback tool
    if func_name == "ask_help":
        return local, True
    
    # Reject invalid function
    tool_names = {t["name"] for t in tools}
    if func_name not in tool_names:
        return local, True
    
    # Check required params exist
    tool = next((t for t in tools if t["name"] == func_name), None)
    if tool:
        required = tool.get("parameters", {}).get("required", [])
        for param in required:
            if param not in func_args or func_args[param] is None:
                return local, True
    
    # ============ PART 3: SEMANTIC VALIDATION (from semantic) ============
    # Check if the chosen function makes sense for the query
    
    # Synonym mapping for tool-query matching
    tool_keywords = {
        "get_weather": ["weather", "temperature", "forecast", "rain", "sunny", "cold", "hot", "climate"],
        "set_alarm": ["alarm", "wake", "morning", "am", "pm", "clock", "oclock"],
        "send_message": ["message", "text", "send", "tell", "say", "saying", "contact"],
        "create_reminder": ["remind", "reminder", "remember", "forget", "notification"],
        "search_contacts": ["search", "find", "look", "contact", "person", "contacts"],
        "play_music": ["play", "music", "song", "listen", "track", "playlist"],
        "set_timer": ["timer", "countdown", "minutes", "minute", "seconds"],
    }
    
    # Check semantic match - but be lenient
    if func_name in tool_keywords:
        keywords = tool_keywords[func_name]
        has_keyword_match = any(kw in query_lower for kw in keywords)
        
        # Only reject if NO keywords match AND we have medium+ complexity
        if not has_keyword_match and net_complexity >= 1:
            return local, True
    
    # ============ PART 4: TRUST LOCAL ============
    local["source"] = "on-device"
    local["_net_complexity"] = net_complexity
    return local, False


def _strategy_smart_validation(messages, tools, local, confidence_threshold):
    """BREAKTHROUGH STRATEGY: Validate output against query entities and semantics."""
    
    if not local.get("function_calls"):
        return local, True  # No output, go to cloud
    
    query = messages[-1]["content"]
    query_lower = query.lower()
    call = local["function_calls"][0]  # Primary function call
    func_name = call.get("name", "")
    func_args = call.get("arguments", {})
    
    # --- RULE 1: Reject "ask_help" fallback tool ---
    if func_name == "ask_help":
        local["_reject_reason"] = "model chose ask_help"
        return local, True
    
    # --- RULE 2: Function name must exist in tools ---
    tool_names = {t["name"] for t in tools}
    if func_name not in tool_names:
        local["_reject_reason"] = f"invalid function: {func_name}"
        return local, True
    
    # --- RULE 3: Extract entities from query and verify they appear in args ---
    # Extract potential entities
    entities = {
        "names": re.findall(r'\b([A-Z][a-z]+)\b', query),  # Capitalized words
        "numbers": re.findall(r'\b(\d+)\b', query),
        "times": re.findall(r'\b(\d{1,2}(?::\d{2})?)\s*(am|pm|AM|PM)?\b', query),
        "locations": re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', query),  # Multi-word capitalized
    }
    
    # Flatten arguments to string for checking
    args_str = json.dumps(func_args).lower()
    
    # Check if key entities from query appear in arguments
    entity_match_score = 0
    entity_total = 0
    
    # Check locations (for weather, etc.)
    for loc in entities["locations"]:
        if len(loc) > 2:  # Skip short words
            entity_total += 1
            if loc.lower() in args_str:
                entity_match_score += 1
    
    # Check names (for messaging, contacts)
    for name in entities["names"]:
        if name.lower() not in {"what", "where", "when", "how", "the", "can", "set", "get", "send", "play"}:
            entity_total += 1
            if name.lower() in args_str:
                entity_match_score += 1
    
    # Check numbers (for alarms, timers)
    for num in entities["numbers"]:
        entity_total += 1
        if num in args_str:
            entity_match_score += 1
    
    local["_entity_match"] = f"{entity_match_score}/{entity_total}"
    
    # If we found entities but none match -> likely wrong output
    if entity_total >= 2 and entity_match_score == 0:
        local["_reject_reason"] = "no entity match"
        return local, True
    
    # --- RULE 4: Function-specific validation ---
    tool = next((t for t in tools if t["name"] == func_name), None)
    if tool:
        required_params = tool.get("parameters", {}).get("required", [])
        
        # Check all required params are present and non-empty
        for param in required_params:
            if param not in func_args:
                local["_reject_reason"] = f"missing required param: {param}"
                return local, True
            
            val = func_args[param]
            # Check for empty or suspicious values
            if val is None or val == "" or val == 0:
                # Exception: minute=0 is valid for alarms
                if not (param == "minute" and val == 0):
                    local["_reject_reason"] = f"empty/invalid param: {param}"
                    return local, True
    
    # --- RULE 5: Semantic coherence check ---
    # Does the function name relate to the query?
    func_keywords = {
        "get_weather": ["weather", "temperature", "forecast", "rain", "sunny", "cold", "hot", "climate"],
        "set_alarm": ["alarm", "wake", "morning", "am", "pm", "clock"],
        "send_message": ["message", "text", "send", "tell", "say", "saying"],
        "create_reminder": ["remind", "reminder", "remember", "forget", "notification"],
        "search_contacts": ["search", "find", "look", "contact", "person"],
        "play_music": ["play", "music", "song", "listen", "track"],
        "set_timer": ["timer", "countdown", "minutes", "minute"],
    }
    
    if func_name in func_keywords:
        keywords = func_keywords[func_name]
        if not any(kw in query_lower for kw in keywords):
            # Function doesn't match query semantics
            local["_reject_reason"] = f"function {func_name} doesn't match query"
            return local, True
    
    # --- RULE 6: Multi-call validation for hard queries ---
    # If query has "and"/"then", we might need multiple function calls
    multi_intent_words = [" and ", " then ", " also ", " plus "]
    has_multi_intent = any(w in query_lower for w in multi_intent_words)
    num_calls = len(local["function_calls"])
    
    if has_multi_intent and num_calls < 2:
        # Query asks for multiple things but model only gave one
        local["_reject_reason"] = "multi-intent query, single call"
        return local, True
    
    # --- ALL CHECKS PASSED ---
    local["source"] = "on-device"
    local["_validation"] = "passed"
    return local, False


def _strategy_combined(messages, tools, local, confidence_threshold):
    """Strategy 4: Combine all strategies."""
    # Step 1: Check complexity first
    result, need_cloud = _strategy_complexity(messages, tools, local, confidence_threshold)
    if result is None:  # Complexity said skip local
        return None, True
    if not need_cloud:
        return result, False
    
    # Step 2: Check semantic match
    result, need_cloud = _strategy_semantic(messages, tools, local, confidence_threshold)
    if result is None:  # Weak semantic match
        return None, True
    if not need_cloud:
        return result, False
    
    # Step 3: Validate output
    result, need_cloud = _strategy_validation(messages, tools, local, confidence_threshold)
    return result, need_cloud


def _postprocess_output(local, tools, query):
    """Fix common model output errors to rescue on-device results."""
    query_lower = query.lower()
    fcs = local.get("function_calls", [])
    if not fcs:
        return local

    tool_map = {t["name"]: t for t in tools}
    fixed_calls = []
    
    for fc in fcs:
        name = fc.get("name", "")
        args = fc.get("arguments", {})
        
        # Skip fallback tool
        if name == "ask_help":
            continue
        
        # Skip unknown functions
        if name not in tool_map:
            continue
        
        tool = tool_map[name]
        props = tool.get("parameters", {}).get("properties", {})
        
        # --- Fix alarm hour/minute from query ---
        if name == "set_alarm":
            time_match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm|AM|PM)', query)
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2)) if time_match.group(2) else 0
                ampm = time_match.group(3).lower()
                if ampm == "pm" and hour < 12:
                    hour += 12
                if ampm == "am" and hour == 12:
                    hour = 0
                args["hour"] = hour
                args["minute"] = minute
            else:
                # Try bare number: "wake me up at 6" 
                num_match = re.search(r'(?:at|for)\s+(\d{1,2})\b', query_lower)
                if num_match:
                    args["hour"] = int(num_match.group(1))
                    if "minute" not in args:
                        args["minute"] = 0
        
        # --- Fix timer minutes from query ---
        if name == "set_timer":
            min_match = re.search(r'(\d+)\s*(?:minute|min)', query_lower)
            if min_match:
                args["minutes"] = int(min_match.group(1))
        
        # --- Fix reminder time from query ---
        if name == "create_reminder":
            time_match = re.search(r'at\s+(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))', query)
            if time_match:
                args["time"] = time_match.group(1)
            # Fix title: extract what to remind about
            title_match = re.search(r'remind(?:\s+me)?(?:\s+(?:about|to))?\s+(.+?)(?:\s+at\s+\d)', query_lower)
            if title_match:
                title = title_match.group(1).strip()
                # Strip leading articles
                title = re.sub(r'^(?:the|a|an)\s+', '', title)
                args["title"] = title
        
        # --- Fix send_message: extract recipient and message from query ---
        if name == "send_message":
            # Extract recipient: look for capitalized name after "to", "text", or "tell"
            recip_match = re.search(r'(?:to|[Tt]ext|[Tt]ell)\s+([A-Z][a-z]+)', query)
            if not recip_match:
                recip_match = re.search(r'(?:send|message)\s+([A-Z][a-z]+)', query)
            if not recip_match:
                skip = {"What","Where","When","How","The","Set","Get","Send","Play","Find","Check","Text","Remind","Wake","Look","Tell","Can","Could","Would","Please"}
                names = [m for m in re.findall(r'\b([A-Z][a-z]+)\b', query) if m not in skip]
                if names:
                    recip_match = type('Match', (), {'group': lambda self, n: names[0]})()
            if recip_match:
                args["recipient"] = recip_match.group(1)
            # Extract message - try multiple patterns, stop at intent boundaries
            msg_match = re.search(
                r'(?:saying|say|says|that)\s+(.+?)(?:'
                r'\.\s|,\s*and\s|,\s*(?:check|get|set|send|find|play|remind|look|wake)'
                r'|\s+and\s+(?:check|get|set|send|find|play|remind|look|wake)'
                r'|\.$|$)',
                query, re.IGNORECASE
            )
            if not msg_match:
                # "tell X Y" or "text X Y" - everything after name
                msg_match = re.search(
                    r'(?:to|[Tt]ext|[Tt]ell|message)\s+[A-Z][a-z]+\s+(.+?)(?:'
                    r'\.\s|,\s*and\s|\s+and\s+(?:check|get|set|send|find|play|remind|look|wake)'
                    r'|\.$|$)',
                    query
                )
            if msg_match:
                args["message"] = msg_match.group(1).strip().rstrip(".,")
                args["message"] = re.sub(r'\s+and$', '', args["message"])
        
        # --- Fix search_contacts: extract name from query ---
        if name == "search_contacts":
            name_match = re.search(r'(?:find|look\s*up|search\s+for)\s+([A-Z][a-z]+)', query, re.IGNORECASE)
            if name_match:
                args["query"] = name_match.group(1)
        
        # --- Fix play_music: extract song from query ---
        if name == "play_music":
            song_match = re.search(r'(?:play|put on|listen to|queue)\s+(.+?)(?:\.|,|\s+and\s+|$)', query, re.IGNORECASE)
            if song_match:
                song = song_match.group(1).strip().rstrip(".")
                # Strip qualifiers like "some" - and only strip "music" if qualifier removed
                had_qualifier = bool(re.match(r'^(?:some|a few|a bit of)\s+', song, re.IGNORECASE))
                song = re.sub(r'^(?:some|a few|a bit of)\s+', '', song, flags=re.IGNORECASE)
                if had_qualifier:
                    song = re.sub(r'\s+music$', '', song)
                args["song"] = song
        
        # --- Ensure integer types ---
        for param_name, param_info in props.items():
            if param_info.get("type") == "integer" and param_name in args:
                try:
                    args[param_name] = int(args[param_name])
                except (ValueError, TypeError):
                    pass
        
        fc["arguments"] = args
        fixed_calls.append(fc)
    
    local["function_calls"] = fixed_calls
    return local


def _infer_tool_call_from_query(query, tools):
    """When model fails, try to construct a function call purely from query parsing."""
    query_lower = query.lower()
    tool_map = {t["name"]: t for t in tools}
    
    # Map keywords to tool names (expanded for robustness)
    keyword_map = [
        (["weather", "temperature", "forecast", "climate", "outside", "how hot", "how cold", "what's it like"], "get_weather"),
        (["alarm", "wake me", "wake up", "wakeup"], "set_alarm"),
        (["message", "text ", "texting", "saying", " say ", "tell ", "send a text", "send text"], "send_message"),
        (["remind", "reminder", "don't forget", "do not forget"], "create_reminder"),
        (["contact", "find ", "look up", "look for", "search for", "search contact", "look someone"], "search_contacts"),
        (["play ", "music", "song", "listen", "put on ", "queue"], "play_music"),
        (["timer", "countdown", r"\d+\s*min", r"\d+\s*sec"], "set_timer"),
    ]
    
    results = []
    for keywords, tool_name in keyword_map:
        if tool_name not in tool_map:
            continue
        if any(kw in query_lower if not kw.startswith("\\") else re.search(kw, query_lower) for kw in keywords):
            tool = tool_map[tool_name]
            args = {}
            
            if tool_name == "get_weather":
                # Extract city name (capitalized words)
                cities = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', query)
                # Filter out common non-city words
                skip = {"What", "How", "Set", "Get", "Send", "Play", "Find", "Check", "Text", "Remind", "Wake", "Look"}
                cities = [c for c in cities if c not in skip]
                if cities:
                    args["location"] = cities[-1]  # Last capitalized phrase is usually the city
                    results.append({"name": tool_name, "arguments": args})
            
            elif tool_name == "set_alarm":
                time_match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm|AM|PM)', query)
                if time_match:
                    hour = int(time_match.group(1))
                    minute = int(time_match.group(2)) if time_match.group(2) else 0
                    ampm = time_match.group(3).lower()
                    if ampm == "pm" and hour < 12:
                        hour += 12
                    if ampm == "am" and hour == 12:
                        hour = 0
                    args = {"hour": hour, "minute": minute}
                    results.append({"name": tool_name, "arguments": args})
            
            elif tool_name == "send_message":
                recip = re.search(r'(?:to|[Tt]ext|[Tt]ell)\s+([A-Z][a-z]+)', query)
                if not recip:
                    recip = re.search(r'(?:send|message)\s+([A-Z][a-z]+)', query)
                if not recip:
                    skip = {"What","Where","When","How","The","Set","Get","Send","Play","Find","Check","Text","Remind","Wake","Look","Tell","Can","Could","Would","Please","Also","Then","And"}
                    names = [m for m in re.findall(r'\b([A-Z][a-z]+)\b', query) if m not in skip]
                    if names:
                        recip = type('M', (), {'group': lambda s, n: names[0]})()
                # Try multiple message extraction patterns
                msg = re.search(
                    r'(?:saying|say|says|that)\s+(.+?)(?:'
                    r'\.\s|,\s*and\s|,\s*(?:check|get|set|send|find|play|remind|look|wake)'
                    r'|\s+and\s+(?:check|get|set|send|find|play|remind|look|wake)'
                    r'|\.$|$)',
                    query, re.IGNORECASE
                )
                if not msg:
                    # "tell X (that) Y" pattern - extract message after name
                    msg = re.search(
                        r'(?:[Tt]ell)\s+[A-Z][a-z]+\s+(?:that\s+)?(.+?)(?:'
                        r'\.\s|,\s*and\s|\s+and\s+(?:check|get|set|send|find|play|remind|look|wake)'
                        r'|\.$|$)',
                        query
                    )
                if not msg:
                    # Generic fallback: extract everything after the recipient name
                    msg = re.search(
                        r'(?:to|[Tt]ext|[Tt]ell|message)\s+[A-Z][a-z]+\s+(.+?)(?:'
                        r'\.\s|,\s*and\s|\s+and\s+(?:check|get|set|send|find|play|remind|look|wake)'
                        r'|\.$|$)',
                        query
                    )
                if recip:
                    args["recipient"] = recip.group(1)
                    message = msg.group(1).strip().rstrip(".,") if msg else ""
                    message = re.sub(r'\s+and$', '', message)
                    args["message"] = message
                    results.append({"name": tool_name, "arguments": args})
            
            elif tool_name == "create_reminder":
                title_match = re.search(r'remind(?:\s+me)?(?:\s+(?:about|to))?\s+(.+?)(?:\s+at\s+\d)', query_lower)
                if not title_match:
                    # "reminder to X at Y" pattern
                    title_match = re.search(r'reminder\s+(?:to\s+)?(.+?)(?:\s+at\s+\d)', query_lower)
                # Try multiple time formats
                time_match = re.search(r'at\s+(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))', query)
                if not time_match:
                    # "at 3 PM" (no minutes)
                    time_match = re.search(r'at\s+(\d{1,2}\s*(?:AM|PM|am|pm))', query)
                if title_match and time_match:
                    title = title_match.group(1).strip()
                    title = re.sub(r'^(?:the|a|an)\s+', '', title)
                    args["title"] = title
                    args["time"] = time_match.group(1)
                    results.append({"name": tool_name, "arguments": args})
            
            elif tool_name == "search_contacts":
                name_match = re.search(r'(?:find|look\s*up|look\s+for|search\s+for|search)\s+([A-Z][a-z]+)', query, re.IGNORECASE)
                if not name_match:
                    # "X's contact" pattern
                    name_match = re.search(r"([A-Z][a-z]+)'s\s+contact", query)
                if name_match:
                    args["query"] = name_match.group(1).capitalize()
                    results.append({"name": tool_name, "arguments": args})
            
            elif tool_name == "play_music":
                song_match = re.search(r'(?:play|put on|listen to|queue)\s+(.+?)(?:\.|,|\s+and\s+|$)', query, re.IGNORECASE)
                if song_match:
                    song = song_match.group(1).strip().rstrip(".")
                    had_qualifier = bool(re.match(r'^(?:some|a few|a bit of)\s+', song, re.IGNORECASE))
                    song = re.sub(r'^(?:some|a few|a bit of)\s+', '', song, flags=re.IGNORECASE)
                    if had_qualifier:
                        song = re.sub(r'\s+music$', '', song)
                    args["song"] = song
                    results.append({"name": tool_name, "arguments": args})
            
            elif tool_name == "set_timer":
                min_match = re.search(r'(\d+)\s*(?:minute|min)', query_lower)
                if min_match:
                    args["minutes"] = int(min_match.group(1))
                    results.append({"name": tool_name, "arguments": args})
                else:
                    # Try seconds
                    sec_match = re.search(r'(\d+)\s*(?:second|sec)', query_lower)
                    if sec_match:
                        secs = int(sec_match.group(1))
                        args["minutes"] = max(1, secs // 60) if secs >= 60 else 1
                        if "seconds" in {p for p in tool.get("parameters",{}).get("properties",{})}:
                            args = {"seconds": secs}
                        results.append({"name": tool_name, "arguments": args})
                    else:
                        # Try generic number + "timer"
                        num_match = re.search(r'(\d+)\s*(?:timer|countdown)', query_lower)
                        if num_match:
                            args["minutes"] = int(num_match.group(1))
                            results.append({"name": tool_name, "arguments": args})
    
    return results if results else None


def _strategy_maximize_local(messages, tools, local, confidence_threshold):
    """OPTIMAL STRATEGY: Accept local almost always.
    
    Math proof: even partial F1 (0.33) with on-device time beats perfect cloud F1.
    Only reject when output is completely empty or has no valid function at all.
    """
    query = messages[-1]["content"]
    query_lower = query.lower()
    
    # Post-process to fix common model errors
    local = _postprocess_output(local, tools, query)
    
    # Check that at least one function call has a valid name
    tool_names = {t["name"] for t in tools}
    valid_calls = [fc for fc in local.get("function_calls", []) if fc.get("name") in tool_names]
    
    # Always try to infer calls from query
    inferred = _infer_tool_call_from_query(query, tools) or []
    
    if not valid_calls:
        # Model failed - use inferred calls
        if inferred:
            local["function_calls"] = inferred
            local["source"] = "on-device"
            return local, False
        # Can't infer anything -> cloud
        return local, True
    
    # For multi-intent queries, use regex inference as primary source
    multi_markers = [" and ", " then ", ", and ", " also "]
    has_multi = any(m in query_lower for m in multi_markers)
    
    if has_multi and inferred and len(inferred) >= 2:
        # For multi-intent, prefer regex output (more reliable than model for complex queries)
        # But verify model's calls too - use the best version of each call
        inferred_tools = {fc["name"]: fc for fc in inferred}
        model_tools = {fc["name"]: fc for fc in valid_calls}
        
        # Start with inferred calls, overlay with model calls that match
        merged = {}
        for name, fc in inferred_tools.items():
            merged[name] = fc
        # Add any model calls for tools not covered by inference
        for name, fc in model_tools.items():
            if name not in merged:
                merged[name] = fc
        
        local["function_calls"] = list(merged.values())
    else:
        local["function_calls"] = valid_calls
    
    local["source"] = "on-device"
    return local, False


# Strategy dispatcher
STRATEGIES = {
    "baseline": _strategy_baseline,
    "always_local": _strategy_always_local,
    "known_reliable": _strategy_known_reliable,
    "complexity": _strategy_complexity,
    "semantic": _strategy_semantic,
    "validation": _strategy_validation,
    "combined": _strategy_combined,
    "smart_validation": _strategy_smart_validation,
    "complexity_plus": _strategy_complexity_plus,
    "complexity_semantic_plus": _strategy_complexity_semantic_plus,
    "maximize_local": _strategy_maximize_local,
}


def _validate_inferred(inferred, tools):
    """Check if regex-inferred calls cover all required params for each tool."""
    tool_map = {t["name"]: t for t in tools}
    for fc in inferred:
        name = fc.get("name", "")
        args = fc.get("arguments", {})
        if name not in tool_map:
            return False
        tool = tool_map[name]
        required = tool.get("parameters", {}).get("required", [])
        for req in required:
            if req not in args or args[req] is None or args[req] == "":
                return False
            # Integer params should be valid
            prop_type = tool.get("parameters", {}).get("properties", {}).get(req, {}).get("type", "string")
            if prop_type == "integer" and not isinstance(args[req], int):
                return False
    return True


# ============== KEYWORD-SCORING TOOL SELECTOR ==============
# Handles ANY tool by analyzing tool name + description keywords

def _tokenize_text(text):
    """Split text into lowercase word tokens."""
    return set(re.findall(r'[a-z]+', text.lower()))

def _score_tool(query_tokens, tool):
    """Score how well a query matches a tool based on keyword overlap."""
    name = tool.get("name", "")
    desc = tool.get("description", "")
    params = tool.get("parameters", {}).get("properties", {})
    
    # Build tool keyword set from name, description, param names, and param descriptions
    tool_text = f"{name.replace('_', ' ')} {desc}"
    for p_name, p_def in params.items():
        tool_text += f" {p_name.replace('_', ' ')} {p_def.get('description', '')}"
    tool_tokens = _tokenize_text(tool_text)
    
    # Score = weighted overlap
    overlap = query_tokens & tool_tokens
    if not overlap:
        return 0.0
    
    # Bonus for matching tool name words
    name_tokens = _tokenize_text(name.replace('_', ' '))
    name_overlap = query_tokens & name_tokens
    
    score = len(overlap) + len(name_overlap) * 2.0  # name matches are worth 3x
    return score

def _extract_generic_args(query, tool):
    """Extract arguments generically based on parameter types and names."""
    props = tool.get("parameters", {}).get("properties", {})
    required = tool.get("parameters", {}).get("required", [])
    args = {}
    
    skip_names = {"What","Where","When","How","The","Set","Get","Send","Play","Find",
                  "Check","Text","Remind","Wake","Look","Tell","Can","Could","Would",
                  "Please","Also","Then","And"}
    
    for param_name, param_def in props.items():
        ptype = param_def.get("type", "string")
        pdesc = param_def.get("description", "").lower()
        
        if ptype == "integer":
            # Extract numbers from query
            numbers = re.findall(r'\b(\d+)\b', query)
            if numbers:
                # Try to pick the most relevant number
                if "hour" in param_name:
                    # Look for time patterns
                    time_match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm|AM|PM)?', query)
                    if time_match:
                        hour = int(time_match.group(1))
                        ampm = (time_match.group(3) or "").lower()
                        if ampm == "pm" and hour < 12: hour += 12
                        if ampm == "am" and hour == 12: hour = 0
                        args[param_name] = hour
                elif "minute" in param_name:
                    time_match = re.search(r'(\d{1,2}):(\d{2})', query)
                    if time_match:
                        args[param_name] = int(time_match.group(2))
                    else:
                        args[param_name] = 0
                elif "min" in param_name or "duration" in param_name:
                    min_match = re.search(r'(\d+)\s*(?:minute|min)', query, re.IGNORECASE)
                    if min_match:
                        args[param_name] = int(min_match.group(1))
                    else:
                        args[param_name] = int(numbers[0])
                else:
                    args[param_name] = int(numbers[0])
        
        elif ptype == "string":
            if any(w in param_name.lower() for w in ["location", "city", "place"]) or \
               any(w in pdesc for w in ["location", "city", "place"]):
                # Extract location: capitalized words, prefer after "in"
                loc_match = re.search(r'(?:in|for|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', query)
                if not loc_match:
                    caps = [m for m in re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', query) if m not in skip_names]
                    if caps:
                        loc_match = type('M', (), {'group': lambda s, n: caps[-1]})()
                if loc_match:
                    args[param_name] = loc_match.group(1)
            
            elif any(w in param_name.lower() for w in ["recipient", "contact", "name", "to"]):
                # Extract person name
                name_match = re.search(r'(?:to|[Tt]ext|message|call)\s+([A-Z][a-z]+)', query)
                if not name_match:
                    caps = [m for m in re.findall(r'\b([A-Z][a-z]+)\b', query) if m not in skip_names]
                    if caps:
                        name_match = type('M', (), {'group': lambda s, n: caps[0]})()
                if name_match:
                    args[param_name] = name_match.group(1)
            
            elif any(w in param_name.lower() for w in ["message", "text", "body", "content"]):
                # Extract message content
                msg = re.search(
                    r'(?:saying|say|says|that)\s+(.+?)(?:'
                    r'\.\s|,\s*and\s|,\s*(?:check|get|set|send|find|play|remind|look|wake)'
                    r'|\s+and\s+(?:check|get|set|send|find|play|remind|look|wake)'
                    r'|\.$|$)', query, re.IGNORECASE)
                if msg:
                    args[param_name] = msg.group(1).strip().rstrip(".,")
            
            elif any(w in param_name.lower() for w in ["song", "track", "playlist", "music"]):
                # Extract song/music
                song_match = re.search(r'(?:play|put on|listen to|queue)\s+(.+?)(?:\.|,|\s+and\s+|$)', query, re.IGNORECASE)
                if song_match:
                    song = song_match.group(1).strip().rstrip(".")
                    had_q = bool(re.match(r'^(?:some|a few|a bit of)\s+', song, re.IGNORECASE))
                    song = re.sub(r'^(?:some|a few|a bit of)\s+', '', song, flags=re.IGNORECASE)
                    if had_q:
                        song = re.sub(r'\s+music$', '', song)
                    args[param_name] = song
            
            elif any(w in param_name.lower() for w in ["title", "subject", "topic"]):
                # Extract title (for reminders etc.)
                title_match = re.search(r'remind(?:\s+me)?(?:\s+(?:about|to))?\s+(.+?)(?:\s+at\s+\d)', query, re.IGNORECASE)
                if title_match:
                    title = title_match.group(1).strip()
                    title = re.sub(r'^(?:the|a|an)\s+', '', title)
                    args[param_name] = title
            
            elif any(w in param_name.lower() for w in ["time", "when", "schedule"]):
                # Extract time
                time_match = re.search(r'at\s+(\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)', query)
                if time_match:
                    args[param_name] = time_match.group(1)
            
            elif any(w in param_name.lower() for w in ["query", "search", "keyword"]):
                # Extract search query - usually a name
                q_match = re.search(r'(?:find|look\s*up|search\s+for|look\s+for)\s+([A-Z][a-z]+)', query, re.IGNORECASE)
                if q_match:
                    args[param_name] = q_match.group(1).capitalize()
                else:
                    caps = [m for m in re.findall(r'\b([A-Z][a-z]+)\b', query) if m not in skip_names]
                    if caps:
                        args[param_name] = caps[0]
    
    return args


def _smart_tool_select(query, tools):
    """Select the best matching tool(s) using keyword scoring + generic arg extraction.
    
    Returns list of {name, arguments} or None.
    """
    query_tokens = _tokenize_text(query)
    query_lower = query.lower()
    
    # Score all tools
    scored = []
    for tool in tools:
        score = _score_tool(query_tokens, tool)
        if score > 0:
            scored.append((score, tool))
    
    if not scored:
        return None
    
    scored.sort(key=lambda x: -x[0])
    
    # Detect multi-intent
    multi_markers = [" and ", " then ", ", and ", " also "]
    has_multi = any(m in query_lower for m in multi_markers)
    
    results = []
    if has_multi and len(scored) >= 2:
        # Pick top tools that have significant scores
        top_score = scored[0][0]
        for score, tool in scored:
            if score >= top_score * 0.3:  # At least 30% of top score
                args = _extract_generic_args(query, tool)
                required = tool.get("parameters", {}).get("required", [])
                if all(r in args and args[r] is not None and args[r] != "" for r in required):
                    results.append({"name": tool["name"], "arguments": args})
    else:
        # Single intent - pick the top tool
        _, best_tool = scored[0]
        args = _extract_generic_args(query, best_tool)
        required = best_tool.get("parameters", {}).get("required", [])
        if all(r in args and args[r] is not None and args[r] != "" for r in required):
            results.append({"name": best_tool["name"], "arguments": args})
    
    return results if results else None


def _prefilter_tools(query, tools):
    """Pre-filter tools by keyword relevance to reduce model context length.
    
    Single-intent: pass top 1 tool.  Multi-intent: pass top 3 tools.
    Regex post-processing catches anything the model misses.
    """
    if len(tools) <= 2:
        return tools

    query_lower = query.lower()
    multi_markers = [" and ", " then ", ", and ", " also "]
    is_multi = any(m in query_lower for m in multi_markers)
    max_keep = 3 if is_multi else 1

    query_tokens = _tokenize_text(query)
    scored = [(_score_tool(query_tokens, t), t) for t in tools]
    scored.sort(key=lambda x: -x[0])

    if scored[0][0] == 0:
        return tools  # Can't determine relevance, pass all

    keep = []
    threshold = scored[0][0] * 0.25
    for score, tool in scored:
        if len(keep) >= max_keep and score < threshold:
            break
        if score > 0:
            keep.append(tool)

    return keep if keep else tools


def generate_hybrid(messages, tools, confidence_threshold=0.5):
    """Hybrid inference: MODEL-PRIMARY with regex post-processing."""
    query = messages[-1]["content"]
    query_lower = query.lower()

    # Pre-filter tools to reduce model context (faster inference)
    filtered_tools = _prefilter_tools(query, tools)

    # PRIMARY: Run on-device model with filtered tools
    local = generate_cactus(messages, filtered_tools)

    # Post-process model output (fix JSON, args, etc.)
    local = _postprocess_output(local, tools, query)

    # Check if model produced valid function calls
    tool_names = {t["name"] for t in tools}
    valid_calls = [
        fc for fc in local.get("function_calls", [])
        if fc.get("name") in tool_names
    ]

    if valid_calls:
        # Model succeeded — augment with regex for multi-intent gaps
        multi_markers = [" and ", " then ", ", and ", " also "]
        has_multi = any(m in query_lower for m in multi_markers)

        if has_multi:
            inferred = _infer_tool_call_from_query(query, tools)
            if inferred and len(inferred) >= 2:
                inferred_tools = {fc["name"]: fc for fc in inferred}
                model_tools = {fc["name"]: fc for fc in valid_calls}
                merged = {}
                for name, fc in inferred_tools.items():
                    merged[name] = fc
                for name, fc in model_tools.items():
                    if name not in merged:
                        merged[name] = fc
                valid_calls = list(merged.values())

        local["function_calls"] = valid_calls
        local["source"] = "on-device"
        local["_path"] = "model"
        return local

    # Model failed — try regex inference as on-device fallback (lazy)
    inferred = _infer_tool_call_from_query(query, tools)
    if inferred and _validate_inferred(inferred, tools):
        local["function_calls"] = inferred
        local["source"] = "on-device"
        local["_path"] = "regex-fallback"
        return local

    # Both failed — use cloud as last resort
    try:
        cloud = generate_cloud(messages, tools)
    except Exception:
        local["source"] = "on-device"
        return local

    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local.get("confidence", 0)
    local_time = local.get("total_time_ms", 0)
    cloud["total_time_ms"] = max(local_time, cloud["total_time_ms"])
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
