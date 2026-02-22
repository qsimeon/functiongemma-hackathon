# Source: https://github.com/wnwoghd22/functiongemma-hackathon
"""On-device only strategy v3.1: Hybrid cactus + rule-based extraction.

Server cloud fallback doesn't work → on-device 100%.
Key insight: The 270M model is good at tool SELECTION but bad at argument EXTRACTION.
Strategy: Use cactus to pick the right tool, then extract arguments with rules.

Techniques:
1. Tool pruning: keyword-based filtering to reduce confusion
2. Multi-action splitting: split compound requests
3. Rule-based argument extraction from user text
4. Cactus as tool selector + fallback for argument values
5. Parse response field when function_calls is empty
"""

import json
import re
import sys

sys.path.insert(0, "cactus/python/src")
from cactus import cactus_init, cactus_complete, cactus_destroy

_FUNCTIONGEMMA_PATH = "cactus/weights/functiongemma-270m-it"

_LOCAL_SYSTEM_PROMPT = (
    "You are a function-calling assistant. You MUST call functions. "
    "Never apologize. Never ask questions. Never refuse. "
    "Output ONLY valid function calls in JSON.\n"
    "Example: User: \"Play jazz\" → {\"name\": \"play_music\", \"arguments\": {\"song\": \"jazz\"}}\n"
    "Example: User: \"Set alarm for 7 AM\" → {\"name\": \"set_alarm\", \"arguments\": {\"hour\": 7, \"minute\": 0}}\n"
    "Example: User: \"Send hi to Bob\" → {\"name\": \"send_message\", \"arguments\": {\"recipient\": \"Bob\", \"message\": \"hi\"}}"
)

# Keyword → tool name mapping for tool pruning
_TOOL_KEYWORDS = {
    "get_weather": ["weather", "temperature", "forecast"],
    "set_alarm": ["alarm", "wake me", "wake up"],
    "send_message": ["message", "text ", "send ", "tell ", "saying", "msg"],
    "create_reminder": ["remind", "reminder"],
    "search_contacts": ["contact", "look up", "find "],
    "play_music": ["play ", "music", "song", "listen"],
    "set_timer": ["timer", "countdown"],
}

# Enhanced descriptions
_ENHANCED_DESCRIPTIONS = {
    "get_weather": "Get current weather for a location. Use when user asks about weather, temperature, or forecast.",
    "set_alarm": "Set an alarm clock. Use when user says alarm or wake me up.",
    "send_message": "Send a text message to a person. Use when user says send, text, message, or tell someone.",
    "create_reminder": "Create a reminder. Use when user says remind me or reminder.",
    "search_contacts": "Search for a contact by name. Use when user says find, look up, or search contacts.",
    "play_music": "Play a song or playlist. Use when user says play or listen to music.",
    "set_timer": "Set a countdown timer. Use when user says timer or countdown.",
}


# ============ Rule-based argument extraction ============

def _extract_location(text):
    """Extract location from text like 'weather in San Francisco'."""
    # "in <Location>" pattern
    m = re.search(r'\bin\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)', text)
    if m:
        return m.group(1)
    # "in <location>" case-insensitive fallback
    m = re.search(r'\bin\s+(\w+(?:\s+\w+)?)\s*[?.!]?\s*$', text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip('?.!')
    # "weather <location>" pattern
    m = re.search(r'weather\s+(?:in\s+|for\s+)?(.+?)(?:\?|$)', text, re.IGNORECASE)
    if m:
        loc = m.group(1).strip().rstrip('?.!')
        if loc:
            return loc
    return None


def _extract_time_hours_minutes(text):
    """Extract hour and minute from text like '7:30 AM' or '6 AM'."""
    # "H:MM AM/PM"
    m = re.search(r'(\d{1,2}):(\d{2})\s*(AM|PM|am|pm|a\.m\.|p\.m\.)', text)
    if m:
        h, mi, ampm = int(m.group(1)), int(m.group(2)), m.group(3).upper().replace('.', '')
        if ampm == 'PM' and h != 12:
            h += 12
        if ampm == 'AM' and h == 12:
            h = 0
        return h, mi

    # "H AM/PM"
    m = re.search(r'(\d{1,2})\s*(AM|PM|am|pm|a\.m\.|p\.m\.)', text)
    if m:
        h, ampm = int(m.group(1)), m.group(2).upper().replace('.', '')
        if ampm == 'PM' and h != 12:
            h += 12
        if ampm == 'AM' and h == 12:
            h = 0
        return h, 0

    # Just a number after "at" or "for"
    m = re.search(r'(?:at|for)\s+(\d{1,2})(?::(\d{2}))?\b', text)
    if m:
        h = int(m.group(1))
        mi = int(m.group(2)) if m.group(2) else 0
        return h, mi

    return None, None


def _extract_time_string(text):
    """Extract time as a string like '3:00 PM' from text."""
    m = re.search(r'(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))', text)
    if m:
        return m.group(1)
    m = re.search(r'(\d{1,2}\s*(?:AM|PM|am|pm))', text)
    if m:
        return m.group(1)
    m = re.search(r'at\s+(\d{1,2}:\d{2})', text)
    if m:
        return m.group(1)
    return None


def _extract_minutes(text):
    """Extract number of minutes from text."""
    m = re.search(r'(\d+)\s*(?:minute|min)', text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r'(?:for|timer)\s+(\d+)', text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _extract_recipient_and_message(text):
    """Extract recipient and message from text like 'Send a message to Alice saying good morning'."""
    # "to <Name> saying <message>"
    m = re.search(r'(?:to|tell)\s+(\w+)\s+(?:saying|that|:)\s+(.+?)(?:\.|$)', text, re.IGNORECASE)
    if m:
        name = m.group(1).strip()
        if name.lower() not in ('him', 'her', 'them'):
            return name, m.group(2).strip().rstrip('.')

    # "<Name> a message saying <message>"
    m = re.search(r'(\w+)\s+a\s+message\s+saying\s+(.+?)(?:\.|$)', text, re.IGNORECASE)
    if m:
        name = m.group(1).strip()
        if name.lower() not in ('send', 'him', 'her', 'them', 'a', 'the'):
            return name, m.group(2).strip().rstrip('.')

    # "<verb> <Name> saying <message>"
    m = re.search(r'(?:text|message|msg)\s+(\w+)\s+(?:saying|that|:)\s+(.+?)(?:\.|$)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip().rstrip('.')

    # "send <message> to <Name>"
    m = re.search(r'send\s+(.+?)\s+to\s+(\w+)', text, re.IGNORECASE)
    if m:
        return m.group(2).strip(), m.group(1).strip()

    # "text <Name> <message>" (no "saying")
    m = re.search(r'(?:text|message)\s+(\w+)\s+(.+?)(?:\.|$)', text, re.IGNORECASE)
    if m:
        name = m.group(1).strip()
        msg = m.group(2).strip().rstrip('.')
        # Filter out common non-name words
        if name.lower() not in ('a', 'the', 'my', 'to', 'saying'):
            return name, msg

    # "send a message to <Name> saying <msg>"  (broader)
    m = re.search(r'to\s+(\w+)\s+saying\s+(.+?)(?:\.|$)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip().rstrip('.')

    # "tell <Name> <message>"
    m = re.search(r'tell\s+(\w+)\s+(.+?)(?:\.|$)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip().rstrip('.')

    return None, None


def _extract_reminder_title_and_time(text):
    """Extract reminder title and time from text."""
    time_str = _extract_time_string(text)

    # "remind me about/to <title> at <time>"
    m = re.search(r'remind\s+(?:me\s+)?(?:about|to)\s+(.+?)\s+at\s+', text, re.IGNORECASE)
    if m:
        title = m.group(1).strip()
        # Strip leading articles
        title = re.sub(r'^(?:the|a|an)\s+', '', title, flags=re.IGNORECASE)
        return title, time_str

    # "remind me <title> at <time>"
    m = re.search(r'remind\s+(?:me\s+)?(.+?)\s+at\s+', text, re.IGNORECASE)
    if m:
        title = m.group(1).strip()
        title = re.sub(r'^(?:the|a|an)\s+', '', title, flags=re.IGNORECASE)
        if title.lower() not in ('', 'to', 'about'):
            return title, time_str

    return None, time_str


def _extract_song(text):
    """Extract song/playlist name from text."""
    # "play <song>"
    m = re.search(r'play\s+(.+?)(?:\.|$)', text, re.IGNORECASE)
    if m:
        song = m.group(1).strip().rstrip('.')
        # Remove filler words
        song = re.sub(r'^(?:some|a|the|my)\s+', '', song, flags=re.IGNORECASE)
        # Strip trailing "music" only for single-word genres (e.g. "jazz music" → "jazz")
        # Keep it for multi-word or adjective+music combos (e.g. "classical music", "lo-fi music")
        words = song.split()
        if len(words) == 2 and words[-1].lower() == "music":
            # Only strip if the first word is a standalone genre name, not an adjective
            adjective_genres = {"classical", "electronic", "ambient", "indie", "acoustic", "instrumental"}
            if words[0].lower() not in adjective_genres:
                song = words[0]
        return song
    # "listen to <song>"
    m = re.search(r'listen\s+to\s+(.+?)(?:\.|$)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip('.')
    return None


def _extract_contact_query(text):
    """Extract contact name from text."""
    # "find <Name> in my contacts"
    m = re.search(r'(?:find|look\s*up|search\s+for)\s+(\w+)', text, re.IGNORECASE)
    if m:
        name = m.group(1).strip()
        if name.lower() not in ('a', 'the', 'my', 'in', 'for'):
            return name
    return None


def _extract_args_for_tool(tool_name, user_text, cactus_args):
    """Extract arguments using rules, falling back to cactus output."""
    if tool_name == "get_weather":
        location = _extract_location(user_text)
        if location:
            return {"location": location}
        if cactus_args.get("location"):
            return cactus_args
        return {"location": user_text.split("in")[-1].strip().rstrip("?.!") if "in" in user_text.lower() else ""}

    elif tool_name == "set_alarm":
        h, mi = _extract_time_hours_minutes(user_text)
        if h is not None:
            return {"hour": h, "minute": mi}
        if isinstance(cactus_args.get("hour"), int) and cactus_args["hour"] > 0:
            return cactus_args
        return {"hour": 0, "minute": 0}

    elif tool_name == "send_message":
        recipient, message = _extract_recipient_and_message(user_text)
        if recipient and message:
            return {"recipient": recipient, "message": message}
        result = {}
        result["recipient"] = recipient or cactus_args.get("recipient", "")
        result["message"] = message or cactus_args.get("message", "")
        return result

    elif tool_name == "create_reminder":
        title, time_str = _extract_reminder_title_and_time(user_text)
        result = {}
        result["title"] = title or cactus_args.get("title", "")
        result["time"] = time_str or cactus_args.get("time", "")
        return result

    elif tool_name == "search_contacts":
        query = _extract_contact_query(user_text)
        if query:
            return {"query": query}
        if cactus_args.get("query"):
            return cactus_args
        return {"query": ""}

    elif tool_name == "play_music":
        song = _extract_song(user_text)
        if song:
            return {"song": song}
        if cactus_args.get("song"):
            s = cactus_args["song"].replace("_", " ")
            return {"song": s}
        return {"song": ""}

    elif tool_name == "set_timer":
        minutes = _extract_minutes(user_text)
        if minutes is not None:
            return {"minutes": abs(minutes)}
        if isinstance(cactus_args.get("minutes"), (int, float)):
            return {"minutes": abs(int(cactus_args["minutes"]))}
        return {"minutes": 0}

    # Unknown tool: use cactus args as-is
    return cactus_args


# ============ Cactus interaction ============

def _repair_json_payload(raw_str):
    s = re.sub(r':\s*0(\d+)(?=\s*[,}\]])', r':\1', raw_str)
    s = re.sub(r',\s*([}\]])', r'\1', s)
    s = s.replace("'", '"')
    return s


def _enhance_tools(tools):
    enhanced = []
    for t in tools:
        t_copy = dict(t)
        name = t_copy.get("name", "")
        if name in _ENHANCED_DESCRIPTIONS:
            t_copy["description"] = _ENHANCED_DESCRIPTIONS[name]
        enhanced.append(t_copy)
    return enhanced


def _prune_tools(user_text, tools):
    text_lower = user_text.lower()
    matched = set()
    for tool_name, keywords in _TOOL_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                matched.add(tool_name)
                break
    if not matched:
        return tools
    pruned = [t for t in tools if t.get("name") in matched]
    return pruned if pruned else tools


def _call_cactus(messages, tools):
    model = cactus_init(_FUNCTIONGEMMA_PATH)
    cactus_tools = [{"type": "function", "function": t} for t in tools]
    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": _LOCAL_SYSTEM_PROMPT}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=512,
        temperature=0.2,
        confidence_threshold=0.0,
        tool_rag_top_k=0,
        stop_sequences=["<end_of_turn>"],
    )
    cactus_destroy(model)
    return raw_str


def _parse_tool_name_from_response(response_text, available_tools):
    """Try to extract a tool name from the response text when function_calls is empty."""
    available_names = {t.get("name") for t in available_tools}
    for name in available_names:
        if name in response_text:
            return name
    # Try "call:<tool_name>" pattern
    m = re.search(r'call[:\s]+(\w+)', response_text)
    if m and m.group(1) in available_names:
        return m.group(1)
    return None


def _parse_cactus_output(raw_str, tools):
    """Parse cactus output, including from response field."""
    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        try:
            raw = json.loads(_repair_json_payload(raw_str))
        except json.JSONDecodeError:
            return None

    calls = raw.get("function_calls", [])

    # If function_calls is empty but response has content, try to parse tool name from response
    if not calls and raw.get("response"):
        tool_name = _parse_tool_name_from_response(raw["response"], tools)
        if tool_name:
            calls = [{"name": tool_name, "arguments": {}}]

    return {
        "function_calls": calls,
        "total_time_ms": raw.get("total_time_ms", 0),
    }


# ============ Tool selection heuristic ============

def _select_tool_by_keywords(user_text, tools):
    """Heuristic tool selection based on keywords when cactus fails."""
    text_lower = user_text.lower()
    available_names = {t.get("name") for t in tools}

    for tool_name, keywords in _TOOL_KEYWORDS.items():
        if tool_name not in available_names:
            continue
        for kw in keywords:
            if kw in text_lower:
                return tool_name
    return None


# ============ Multi-action ============

def _is_multi_action(user_text, tools):
    text_lower = user_text.lower()
    markers = (" and ", " then ", " also ", " plus ", ", and ")
    has_marker = any(m in text_lower for m in markers) or text_lower.count(",") >= 2
    if not has_marker:
        return False
    matched = set()
    available_names = {t.get("name") for t in tools}
    for tool_name, keywords in _TOOL_KEYWORDS.items():
        if tool_name not in available_names:
            continue
        for kw in keywords:
            if kw in text_lower:
                matched.add(tool_name)
                break
    return len(matched) >= 2


def _split_multi_action(user_text):
    """Split compound requests. Returns list of sub-request strings."""
    # First try combined comma + and splitting for patterns like "A, B, and C"
    # Replace ", and " with a unique delimiter first
    normalized = re.sub(r',\s+and\s+', ' |SPLIT| ', user_text, flags=re.IGNORECASE)
    # Then replace remaining ", "
    normalized = re.sub(r',\s+', ' |SPLIT| ', normalized)
    # Then replace remaining " and "
    normalized = re.sub(r'\s+and\s+', ' |SPLIT| ', normalized, flags=re.IGNORECASE)

    parts = [p.strip() for p in normalized.split('|SPLIT|') if p.strip()]
    if len(parts) <= 1:
        return [user_text]
    return parts


def _extract_names_from_text(text):
    """Extract capitalized names from text."""
    # Find capitalized words that look like names (not at sentence start)
    words = text.split()
    names = []
    for i, w in enumerate(words):
        clean = w.strip('.,!?')
        if clean and clean[0].isupper() and clean.isalpha() and len(clean) > 1:
            # Skip words at absolute start or common non-names
            if clean.lower() not in ('set', 'play', 'find', 'look', 'send', 'text', 'check',
                                      'get', 'what', 'how', 'the', 'and', 'remind', 'i', 'am', 'pm'):
                names.append(clean)
    return names


def _resolve_pronouns_in_subrequests(sub_requests, full_text):
    """Replace pronouns like 'him', 'her' with actual names from the full text."""
    names = _extract_names_from_text(full_text)
    if not names:
        return sub_requests

    resolved = []
    for req in sub_requests:
        # Check if this sub-request has a pronoun that needs resolving
        if re.search(r'\b(him|her|them)\b', req, re.IGNORECASE):
            # Find a name from the full text that's not in this sub-request
            for name in names:
                if name.lower() not in req.lower():
                    req = re.sub(r'\bhim\b', name, req, flags=re.IGNORECASE)
                    req = re.sub(r'\bher\b', name, req, flags=re.IGNORECASE)
                    req = re.sub(r'\bthem\b', name, req, flags=re.IGNORECASE)
                    break
        resolved.append(req)
    return resolved


# ============ Main entry point ============

def _process_single(user_text, tools):
    """Process a single (non-compound) request."""
    messages = [{"role": "user", "content": user_text}]
    enhanced = _enhance_tools(tools)
    pruned = _prune_tools(user_text, enhanced)

    raw_str = _call_cactus(messages, pruned)
    result = _parse_cactus_output(raw_str, pruned)

    total_time = result["total_time_ms"] if result else 0

    tool_name = None
    cactus_args = {}

    if result and result["function_calls"]:
        call = result["function_calls"][0]
        if isinstance(call, dict) and isinstance(call.get("name"), str):
            tool_name = call["name"]
            cactus_args = call.get("arguments", {})
            if not isinstance(cactus_args, dict):
                cactus_args = {}

    # If cactus didn't pick a tool, use keyword heuristic
    if not tool_name:
        tool_name = _select_tool_by_keywords(user_text, tools)

    if not tool_name:
        return {"function_calls": [], "total_time_ms": total_time}

    # Validate tool_name exists in available tools
    available_names = {t.get("name") for t in tools}
    if tool_name not in available_names:
        tool_name = _select_tool_by_keywords(user_text, tools)
        if not tool_name:
            return {"function_calls": [], "total_time_ms": total_time}

    # Extract arguments using rules (with cactus args as fallback)
    args = _extract_args_for_tool(tool_name, user_text, cactus_args)

    return {
        "function_calls": [{"name": tool_name, "arguments": args}],
        "total_time_ms": total_time,
    }


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """On-device only generation with tool pruning and rule-based extraction."""
    user_text = " ".join(
        str(m.get("content", ""))
        for m in messages
        if m.get("role") == "user"
    )

    if _is_multi_action(user_text, tools):
        # Split and process each sub-request independently
        sub_requests = _split_multi_action(user_text)
        sub_requests = _resolve_pronouns_in_subrequests(sub_requests, user_text)
        all_calls = []
        total_time = 0
        seen_tools = set()

        for sub_req in sub_requests:
            sub_result = _process_single(sub_req, tools)
            for call in sub_result.get("function_calls", []):
                # Avoid duplicate tool calls
                if call["name"] not in seen_tools:
                    all_calls.append(call)
                    seen_tools.add(call["name"])
            total_time += sub_result.get("total_time_ms", 0)

        return {
            "function_calls": all_calls,
            "total_time_ms": total_time,
            "source": "on-device",
        }
    else:
        result = _process_single(user_text, tools)
        result["source"] = "on-device"
        return result
