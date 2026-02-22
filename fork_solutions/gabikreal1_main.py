# Source: https://github.com/gabikreal1/functiongemma-hackathon
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time, re, copy
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset, cactus_tokenize, cactus_score_window
from google import genai
from google.genai import types

# ── Persistent model ─────────────────────────────────────────────────
_model = None
def _get_model():
    global _model
    if _model is None:
        _model = cactus_init(functiongemma_path)
    return _model

# ── Persistent Gemini client ─────────────────────────────────────────
_gemini_client = None
def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    return _gemini_client


# ══════════════════════════════════════════════════════════════════════
# VERB SYNONYMS — for keyword detection generalization
# ══════════════════════════════════════════════════════════════════════

_VERB_SYNONYMS = {
    "send": {"send", "text", "message", "write", "tell", "notify"},
    "get": {"get", "check", "show", "display", "what", "how", "fetch", "retrieve"},
    "set": {"set", "create", "make", "configure", "enable", "activate", "adjust"},
    "play": {"play", "listen", "start", "put", "queue", "stream"},
    "find": {"find", "search", "look", "lookup", "locate", "discover"},
    "remind": {"remind", "reminder", "remember"},
    "alarm": {"alarm", "wake", "wakeup"},
    "timer": {"timer", "countdown", "stopwatch"},
    "weather": {"weather", "forecast", "climate"},
    "music": {"music", "song", "playlist", "track", "album", "tune"},
    "contact": {"contact", "person", "people", "friend", "colleague"},
    "call": {"call", "phone", "dial", "ring"},
    "turn": {"turn", "switch", "toggle"},
    "light": {"light", "lamp", "bulb"},
    "thermostat": {"thermostat", "heating", "cooling"},
    "direction": {"direction", "directions", "navigate", "navigation", "route", "driving"},
    "restaurant": {"restaurant", "dining", "eatery", "eat"},
    "workout": {"workout", "exercise", "log", "fitness", "gym"},
    "event": {"event", "meeting", "appointment", "calendar", "schedule"},
    "translate": {"translate", "translation", "interpret"},
    "cart": {"cart", "basket", "shopping", "buy", "purchase", "add"},
    "order": {"order", "delivery", "shipment", "tracking", "status"},
    "book": {"book", "reserve", "ride", "taxi", "uber"},
    "volume": {"volume", "speaker", "sound", "audio"},
    "lock": {"lock", "secure", "bolt"},
    "news": {"news", "headlines", "read", "article"},
    "note": {"note", "memo", "jot", "take"},
    "convert": {"convert", "exchange", "currency"},
}


# ══════════════════════════════════════════════════════════════════════
# QUERY NORMALIZATION — strip fillers, expand bare times
# ══════════════════════════════════════════════════════════════════════

_TIME_NORM = re.compile(r'(?<!\d:)(?<!\d)\b(\d{1,2})\s+(AM|PM|am|pm)\b')

_FILLER_RE = re.compile(
    r'\b(please|can you|could you|would you|i need you to|i want you to|'
    r'i\'d like you to|go ahead and|just|kindly)\b',
    re.IGNORECASE
)

def _normalize_query(text):
    """Expand bare times and strip politeness fillers."""
    def _expand(m):
        hour = m.group(1)
        ampm = m.group(2)
        start = m.start()
        if start > 0 and text[start-1] == ':':
            return m.group(0)
        return f"{hour}:00 {ampm}"
    text = _TIME_NORM.sub(_expand, text)
    text = _FILLER_RE.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.strip('.,!? ')
    return text


# ══════════════════════════════════════════════════════════════════════
# ENHANCED TOOL DESCRIPTIONS — dynamically generated from schema
# ══════════════════════════════════════════════════════════════════════

def _generate_enhanced_description(tool):
    """
    Generate a richer description from tool schema by adding
    verb synonyms, trigger words, and parameter hints.
    Generalizable — works with any tool schema.
    """
    name = tool["name"]
    desc = tool.get("description", "")
    props = tool.get("parameters", {}).get("properties", {})

    # Extract verb and object from tool name (e.g., "set_alarm" → "set", "alarm")
    parts = name.lower().split("_")

    # Find synonyms for the verb
    triggers = set()
    for part in parts:
        for verb, syns in _VERB_SYNONYMS.items():
            if part == verb or part in syns:
                triggers |= syns
                break
        triggers.add(part)

    # Add parameter format hints
    param_hints = []
    for pname, pschema in props.items():
        ptype = pschema.get("type", "string")
        pdesc = pschema.get("description", "")
        hint = f"{pname}: {pdesc}" if pdesc else f"{pname} ({ptype})"
        param_hints.append(hint)

    # Build enhanced description
    trigger_str = ", ".join(sorted(triggers - {'a', 'an', 'the'}))
    enhanced = f"{desc}. Triggers: {trigger_str}."
    if param_hints:
        enhanced += " Params: " + "; ".join(param_hints) + "."

    return enhanced


def _enrich_tools(tools):
    """Deep-copy tools and enhance descriptions + add format hints."""
    enriched = []
    for t in tools:
        t2 = copy.deepcopy(t)

        # Generate enhanced description
        t2["description"] = _generate_enhanced_description(t)

        props = t2.get("parameters", {}).get("properties", {})
        if "time" in props:
            desc = props["time"].get("description", "")
            if "format" not in desc.lower():
                props["time"]["description"] = desc + " (format like '3:00 PM', not ISO)"
        if "minute" in props:
            desc = props["minute"].get("description", "")
            if "0-59" not in desc:
                props["minute"]["description"] = desc + " (0-59, use 0 for on-the-hour)"
        if "hour" in props:
            desc = props["hour"].get("description", "")
            if "0-23" not in desc:
                props["hour"]["description"] = desc + " (0-23)"
        enriched.append(t2)
    return enriched


# ══════════════════════════════════════════════════════════════════════
# PHASE 1: GUIDED INFERENCE — keyword → single tool
# ══════════════════════════════════════════════════════════════════════

def _build_keyword_index(tools):
    """
    Build keyword→tool mapping using tool name parts, description words,
    and verb synonyms. Fully generalizable.
    """
    index = {}
    for t in tools:
        name = t["name"]
        name_parts = name.lower().split("_")
        desc_words = set(re.findall(r'[a-z]+', t.get("description", "").lower()))

        keywords = set(name_parts) | desc_words

        # Add verb synonyms for each name part
        for part in name_parts:
            for verb, syns in _VERB_SYNONYMS.items():
                if part == verb or part in syns:
                    keywords |= syns

        # Add param names and description keywords
        for pname, pschema in t.get("parameters", {}).get("properties", {}).items():
            keywords |= set(pname.lower().split("_"))
            keywords |= set(re.findall(r'[a-z]+', pschema.get("description", "").lower()))

        # Remove very generic words
        generic = {'a', 'an', 'the', 'for', 'to', 'and', 'or', 'of', 'in', 'by',
                   'with', 'is', 'it', 'current', 'given', 'type', 'string',
                   'integer', 'number', 'object', 'name', 'description', 'value'}
        keywords -= generic

        for kw in keywords:
            if len(kw) > 2:
                if kw not in index:
                    index[kw] = []
                if t not in index[kw]:
                    index[kw].append(t)

    return index


def _identify_single_tool(query_text, tools):
    """
    Try to identify a single best tool from query keywords.
    Uses verb synonyms and schema-derived keywords.
    Returns the tool if confident, None if ambiguous.
    """
    if len(tools) <= 1:
        return tools[0] if tools else None

    query_lower = query_text.lower()
    query_words = set(re.findall(r'[a-z]+', query_lower))

    keyword_index = _build_keyword_index(tools)

    tool_scores = {}
    for t in tools:
        tool_scores[t["name"]] = 0

    for word in query_words:
        if word in keyword_index:
            matched_tools = keyword_index[word]
            if len(matched_tools) == 1:
                tool_scores[matched_tools[0]["name"]] += 3
            else:
                for mt in matched_tools:
                    tool_scores[mt["name"]] += 1

    if not tool_scores:
        return None

    sorted_tools = sorted(tool_scores.items(), key=lambda x: x[1], reverse=True)
    best_name, best_score = sorted_tools[0]
    second_score = sorted_tools[1][1] if len(sorted_tools) > 1 else 0

    if best_score >= 3 and best_score >= second_score * 2:
        return next(t for t in tools if t["name"] == best_name)

    return None


# ══════════════════════════════════════════════════════════════════════
# PARAM HINT CLASSIFICATION — data-driven approach from mate's arch
# ══════════════════════════════════════════════════════════════════════

_PARAM_HINT_PATTERNS = {
    "location": {"location", "city", "place", "area", "town", "address", "where"},
    "destination": {"destination", "direction"},
    "room": {"room", "bedroom", "kitchen", "bathroom", "living"},
    "door": {"door", "gate", "entrance"},
    "person": {"recipient", "person", "contact", "sender", "who"},
    "message": {"message", "body"},
    "time_string": {"time", "when", "schedule"},
    "time_hour": {"hour"},
    "time_minute": {"minute"},
    "duration": {"duration", "minutes", "seconds", "length", "timeout", "period"},
    "title": {"title", "subject", "heading", "label", "reminder"},
    "query": {"query", "search", "term", "keyword", "filter", "lookup"},
    "media": {"song", "music", "track", "playlist", "album", "video", "media", "tune"},
    "topic": {"topic", "category", "genre"},
    "numeric_value": {"temperature", "level", "volume", "brightness", "speed", "amount", "quantity", "count", "number"},
    "cuisine": {"cuisine", "food", "dish"},
    "activity": {"activity", "exercise", "sport", "workout"},
    "item": {"item", "product", "goods"},
    "ride_type": {"ride_type", "ride"},
    "language": {"language", "lang", "locale"},
    "content_text": {"content", "text", "note", "description"},
    "currency": {"currency", "from_currency", "to_currency"},
    "order_id": {"order_id", "order", "tracking"},
}


def _classify_param(param_name, param_desc, param_type):
    """Classify a parameter into a hint category based on name and description."""
    name_lower = param_name.lower()
    desc_lower = param_desc.lower()

    # ── EXACT NAME MATCH — highest priority ──
    for category, trigger_words in _PARAM_HINT_PATTERNS.items():
        if name_lower in trigger_words:
            # Special: integer type overrides
            if param_type in ("integer", "number"):
                if name_lower in ("minutes", "duration", "seconds", "length", "timeout", "period"):
                    return "duration"
                if name_lower == "hour":
                    return "time_hour"
                if name_lower == "minute":
                    return "time_minute"
                if name_lower in ("temperature", "level", "volume", "brightness", "speed", "amount", "quantity", "count", "number"):
                    return "numeric_value"
            return category

    # ── Special integer handling ──
    if param_type in ("integer", "number"):
        if "minutes" in name_lower or "minutes" in desc_lower:
            return "duration"
        if "hour" in name_lower:
            return "time_hour"
        if "minute" in name_lower:
            return "time_minute"
        # Generic numeric (temperature, level, quantity, etc.)
        return "numeric_value"

    # ── FUZZY: check description words for best match ──
    best_category = None
    best_score = 0
    for category, trigger_words in _PARAM_HINT_PATTERNS.items():
        score = 0
        for tw in trigger_words:
            if tw in name_lower:
                score += 3
            if tw in desc_lower:
                score += 1
        if score > best_score:
            best_score = score
            best_category = category

    return best_category


# ══════════════════════════════════════════════════════════════════════
# GENERAL-PURPOSE ARG EXTRACTORS — one per param category
# ══════════════════════════════════════════════════════════════════════

_GARBLED_RE = re.compile(r'[\u0400-\u04FF\u0500-\u052F\u0590-\u05FF\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')

_WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "forty-five": 45, "fifty": 50, "half": 30, "quarter": 15,
}


def _extract_location(text):
    m = re.search(r'\b(?:in|for|at|near)\s+([A-Z][a-zA-Z\s]+?)(?:\s*[.?!,]|\s+and\s|\s+then\s|$)', text)
    return m.group(1).strip() if m else None


def _extract_destination(text):
    """Extract destination after 'to' or 'to the'."""
    m = re.search(r'\bto\s+(?:the\s+)?(.+?)(?:\s*[.?!,]|\s+and\s|\s+then\s|$)', text, re.IGNORECASE)
    if m:
        dest = m.group(1).strip().rstrip('.')
        # Don't return if it's a person name pattern (single capitalized word after 'to')
        if re.match(r'^[A-Z][a-z]+$', dest):
            return None
        return dest
    return None


def _extract_room(text):
    """Extract room name from 'the X light/room' or 'in the X' patterns."""
    m = re.search(r'\bthe\s+(.+?)\s+light\b', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r'\b(?:in|of)\s+(?:the\s+)?(.+?)\s*(?:room)?\s*(?:[.?!,]|$)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _extract_door(text):
    """Extract door name like 'front door', 'back door'."""
    m = re.search(r'\b((?:front|back|side|garage|main|rear)\s+door)\b', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r'\block\s+(?:the\s+)?(.+?)\s*(?:[.?!,]|$)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip('.')
    return None


def _extract_topic(text):
    """Extract topic/category from text."""
    m = re.search(r'\b(?:the\s+)?(?:latest\s+)?(\w+)\s+news\b', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r'\bnews\s+(?:about|on|for)\s+(.+?)(?:\s*[.?!,]|\s+and\s|$)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _extract_numeric_value(text):
    """Extract a standalone number from text (for temperature, volume, etc.)."""
    m = re.search(r'\bto\s+(\d+)\b', text)
    if m:
        return int(m.group(1))
    nums = re.findall(r'\b(\d+)\b', text)
    if nums:
        return int(nums[0])
    for word, num in _WORD_TO_NUM.items():
        if re.search(r'\b' + word + r'\b', text, re.IGNORECASE):
            return num
    return None


def _extract_cuisine(text):
    """Extract cuisine type."""
    m = re.search(r'\b(?:an?\s+)?(\w+)\s+restaurant\b', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r'\b(\w+)\s+(?:food|cuisine|dish)\b', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _extract_activity(text):
    """Extract activity/exercise type."""
    m = re.search(r'\b(?:minute|min)\s+(\w+)\s+(?:workout|session|exercise)\b', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r'\b(\w+)\s+(?:workout|session|exercise)\b', text, re.IGNORECASE)
    if m:
        val = m.group(1).strip()
        if val.lower() not in ('a', 'the', 'my', 'this'):
            return val
    return None


def _extract_item(text):
    """Extract product/item name."""
    m = re.search(r'\b(?:add)\s+(?:\d+\s+)?(.+?)\s+(?:to\s+(?:the\s+)?cart|to\s+(?:the\s+)?basket)\b', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _extract_quantity(text):
    """Extract quantity number."""
    m = re.search(r'\badd\s+(\d+)\b', text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _extract_ride_type(text):
    """Extract ride type (economy, premium, etc.)."""
    m = re.search(r'\b(economy|premium|standard|luxury|shared|pool)\b', text, re.IGNORECASE)
    if m:
        return m.group(1).strip().lower()
    return None


def _extract_language(text):
    """Extract target language."""
    m = re.search(r'\bto\s+([A-Z][a-z]+)\b', text)
    if m:
        lang = m.group(1)
        if lang.lower() in ('spanish', 'french', 'german', 'italian', 'portuguese',
                            'japanese', 'chinese', 'korean', 'russian', 'arabic',
                            'hindi', 'dutch', 'swedish', 'norwegian', 'danish',
                            'polish', 'turkish', 'greek', 'hebrew', 'thai'):
            return lang
    return None


def _extract_translate_text(text):
    """Extract text to translate (between 'translate' and 'to Language')."""
    m = re.search(r'\btranslate\s+(.+?)\s+(?:to|into)\s+[A-Z]', text, re.IGNORECASE)
    if m:
        return m.group(1).strip().strip('"\'')
    return None


def _extract_content_text(text):
    """Extract content/body text after 'content' or 'with' keyword."""
    m = re.search(r'\b(?:content|body|with\s+content)\s+(.+?)(?:\s+and\s+|\s+then\s+|[.!]?\s*$)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip('.')
    return None


def _extract_order_id(text):
    """Extract order ID."""
    m = re.search(r'\border\s+(?:id\s+|#?\s*)?(\w+)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _extract_currency(text):
    """Extract currency code."""
    codes = re.findall(r'\b([A-Z]{3})\b', text)
    return codes[0] if codes else None


def _extract_person(text):
    # Match person name after action verbs/prepositions
    names = re.findall(r'(?:to|text|message|send|tell|notify)\s+([A-Z][a-z]+)', text)
    if not names:
        names = re.findall(r'(?:[Tt]o|[Tt]ext|[Mm]essage|[Ss]end|[Tt]ell|[Nn]otify)\s+([A-Z][a-z]+)', text)
    return names[0] if names else None


def _extract_query_term(text):
    names = re.findall(r'(?:[Ff]ind|[Ll]ook\s+up|[Ss]earch\s+for?)\s+([A-Z][a-z]+)', text)
    return names[0] if names else None


def _extract_message_text(text):
    m = re.search(r'(?:saying|says?|that)\s+(.+?)(?:\s+and\s+|\s+then\s+|[.!]?\s*$)', text, re.IGNORECASE)
    return m.group(1).strip().rstrip('.') if m else None


def _extract_time_string(text):
    m = re.search(r'\b(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))\b', text)
    if m:
        return m.group(1).strip()
    m = re.search(r'\b(\d{1,2})\s+(AM|PM|am|pm)\b', text)
    if m:
        return f"{m.group(1)}:00 {m.group(2).upper()}"
    return None


def _extract_time_hour(text):
    text_lower = text.lower()
    m = re.search(r'\b(\d{1,2}):(\d{2})\s*(am|pm)\b', text_lower)
    if m:
        h = int(m.group(1))
        ampm = m.group(3)
        if ampm == 'pm' and h < 12: h += 12
        if ampm == 'am' and h == 12: h = 0
        return h
    m = re.search(r'\b(\d{1,2})\s*:?\s*(?:00\s*)?(am|pm)\b', text_lower)
    if m:
        h = int(m.group(1))
        ampm = m.group(2)
        if ampm == 'pm' and h < 12: h += 12
        if ampm == 'am' and h == 12: h = 0
        return h
    return None


def _extract_time_minute(text):
    text_lower = text.lower()
    m = re.search(r'\b(\d{1,2}):(\d{2})\s*(am|pm)\b', text_lower)
    if m:
        return int(m.group(2))
    m = re.search(r'\b\d{1,2}\s*:?\s*(?:00\s*)?(am|pm)\b', text_lower)
    if m:
        return 0
    return None


def _extract_duration(text):
    text_lower = text.lower()
    m = re.search(r'(\d+)\s*(?:-?\s*)?minute', text_lower)
    if m:
        return int(m.group(1))
    for word, num in _WORD_TO_NUM.items():
        if re.search(r'\b' + word + r'\b.*minute', text_lower):
            return num
    return None


def _extract_title(text):
    # "called X at TIME" pattern — for events, notes, etc.
    m = re.search(r'\b(?:called|titled|named)\s+(.+?)(?:\s+at\s+\d|\s+and\s+|[.!]?\s*$)', text, re.IGNORECASE)
    if m:
        # If there's a "with content" after, strip it
        title = re.sub(r'\s+with\s+content\s+.*$', '', m.group(1), flags=re.IGNORECASE)
        return title.strip().rstrip('.')
    # "remind me about X" pattern
    m = re.search(r'remind\s+(?:me\s+)?about\s+(?:the\s+)?(.+?)(?:\s+at\s+|\s+and\s+|[.!]?\s*$)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip('.')
    # "remind me to X" pattern
    m = re.search(r'remind\s+(?:me\s+)?to\s+(.+?)(?:\s+at\s+|\s+and\s+|[.!]?\s*$)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip('.')
    # "event/note X at TIME" pattern
    m = re.search(r'\b(?:event|note|reminder)\s+(.+?)(?:\s+at\s+\d|\s+and\s+|[.!]?\s*$)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip('.')
    return None


def _extract_media(text):
    m = re.search(r'(?:play|listen\s+to|put\s+on)\s+(?:some\s+)?(.+?)(?:\s+and\s+|\s+then\s+|[.!]?\s*$)', text, re.IGNORECASE)
    if m:
        song = m.group(1).strip().rstrip('.')
        if re.search(r'\bsome\b', text, re.IGNORECASE):
            song = re.sub(r'\s+music\s*$', '', song, flags=re.IGNORECASE)
        return song
    return None


# Map category → extractor function
_EXTRACTORS = {
    "location": _extract_location,
    "destination": _extract_destination,
    "room": _extract_room,
    "door": _extract_door,
    "person": _extract_person,
    "query": _extract_query_term,
    "message": _extract_message_text,
    "time_string": _extract_time_string,
    "time_hour": _extract_time_hour,
    "time_minute": _extract_time_minute,
    "duration": _extract_duration,
    "title": _extract_title,
    "media": _extract_media,
    "topic": _extract_topic,
    "numeric_value": _extract_numeric_value,
    "cuisine": _extract_cuisine,
    "activity": _extract_activity,
    "item": _extract_item,
    "ride_type": _extract_ride_type,
    "language": _extract_language,
    "content_text": _extract_content_text,
    "order_id": _extract_order_id,
}


def _extract_numbers(text):
    """Extract all numbers from text."""
    text_lower = text.lower()
    nums = set()
    for match in re.finditer(r'\b\d+\b', text):
        nums.add(int(match.group()))
    for match in re.finditer(r'\b(\d{1,2}):(\d{2})\b', text):
        nums.add(int(match.group(1)))
        nums.add(int(match.group(2)))
    for word, num in _WORD_TO_NUM.items():
        if re.search(r'\b' + word + r'\b', text_lower):
            nums.add(num)
    return nums


def _extract_words(text):
    """Extract meaningful words from text (lowercased, no stopwords)."""
    stopwords = {'a', 'an', 'the', 'in', 'at', 'for', 'to', 'and', 'or', 'my',
                 'me', 'i', 'is', 'what', 'how', 'set', 'get', 'play', 'send',
                 'find', 'check', 'look', 'up', 'some', 'of', 'about', 'saying',
                 'remind', 'text', 'message', 'contacts', 'contact', 'weather',
                 'alarm', 'timer', 'minute', 'minutes', 'hour', 'am', 'pm',
                 'wake', 'like', 'him', 'her', 'them', 'it', 'that', 'this',
                 'please', 'can', 'you', 'will', 'just', 'could', 'would',
                 'turn', 'on', 'off', 'lock', 'book', 'read', 'take', 'log',
                 'translate', 'convert', 'add', 'create', 'make', 'called',
                 'with', 'content', 'from', 'into', 'latest', 'specific',
                 'be', 'do', 'go', 'has', 'have', 'had', 'was', 'were',
                 'not', 'but', 'so', 'if', 'then', 'also', 'too', 'its'}
    words = set()
    for w in re.findall(r"[a-zA-Z'-]+", text.lower()):
        if w not in stopwords and len(w) > 1:
            words.add(w)
    return words


# ══════════════════════════════════════════════════════════════════════
# SCHEMA-DRIVEN ARG EXTRACTION — builds extractor from tool schema
# ══════════════════════════════════════════════════════════════════════

# Cache for built extractors
_extractor_cache = {}

def _build_generic_extractor(tool):
    """
    Build an arg extractor function for a tool based on its schema.
    Each parameter is classified into a category and assigned the appropriate
    extractor function. Cached per tool name.
    """
    name = tool["name"]
    if name in _extractor_cache:
        return _extractor_cache[name]

    props = tool.get("parameters", {}).get("properties", {})
    param_extractors = {}

    for pname, pschema in props.items():
        ptype = pschema.get("type", "string").lower()
        pdesc = pschema.get("description", "")
        category = _classify_param(pname, pdesc, ptype)

        # Special: translate_text tool "text" param → use translate_text extractor
        if name == "translate_text" and pname == "text":
            param_extractors[pname] = _extract_translate_text
            continue

        # Special: quantity param → use quantity extractor
        if pname == "quantity":
            param_extractors[pname] = _extract_quantity
            continue

        if category and category in _EXTRACTORS:
            param_extractors[pname] = _EXTRACTORS[category]

    def extractor(text):
        result = {}
        for pname, extract_fn in param_extractors.items():
            val = extract_fn(text)
            if val is not None:
                result[pname] = val
        return result if result else None

    _extractor_cache[name] = extractor
    return extractor


def _extract_args_from_text(tool_def, user_text):
    """
    FUSION: Extract argument values from user text using schema-driven extractors.
    General-purpose — classifies each param and uses appropriate extractor.
    """
    extractor = _build_generic_extractor(tool_def)
    return extractor(user_text)


# ══════════════════════════════════════════════════════════════════════
# ARGUMENT SANITIZATION
# ══════════════════════════════════════════════════════════════════════

def _sanitize_args(args, tool_def, user_text=""):
    """Fix hallucinations in a general way — works for any tool schema."""
    if not isinstance(args, dict):
        return args

    props = tool_def.get("parameters", {}).get("properties", {})
    cleaned = {}

    for key, value in args.items():
        if value is None or value == "None" or value == "null":
            continue
        if isinstance(value, str):
            if _GARBLED_RE.search(value):
                continue
            value = value.strip().strip("'").strip('"')
            if not value:
                continue
        if key in props:
            expected_type = props[key].get("type", "").lower()
            if expected_type in ("integer", "number") and isinstance(value, str):
                try:
                    value = int(value) if expected_type == "integer" else float(value)
                except (ValueError, TypeError):
                    pass
        if isinstance(value, (int, float)) and value < 0:
            value = abs(int(value))
        if key == "minute" and isinstance(value, (int, float)):
            value = int(value)
            if value > 59:
                value = 0
        if key == "hour" and isinstance(value, (int, float)):
            value = int(value) % 24
        cleaned[key] = value

    return cleaned


# ══════════════════════════════════════════════════════════════════════
# CORE LOCAL INFERENCE
# ══════════════════════════════════════════════════════════════════════

def _run_local(messages, tools, rag_top_k=0, temperature=0.1, sampling_top_k=1):
    """Run FunctionGemma and return sanitized result."""
    model = _get_model()
    cactus_reset(model)

    enriched = _enrich_tools(tools)
    cactus_tools = [{"type": "function", "function": t} for t in enriched]

    user_text = ""
    processed_msgs = []
    for m in messages:
        if m.get("role") == "user":
            normalized = _normalize_query(m["content"])
            user_text = normalized
            processed_msgs.append({"role": "user", "content": normalized})
        else:
            processed_msgs.append(m)

    start = time.time()
    try:
        raw_str = cactus_complete(
            model,
            processed_msgs,
            tools=cactus_tools,
            force_tools=True,
            max_tokens=256,
            temperature=temperature,
            top_k=sampling_top_k,
            tool_rag_top_k=rag_top_k,
            confidence_threshold=0.0,
            stop_sequences=["<|im_end|>", "<end_of_turn>"],
        )
        elapsed = (time.time() - start) * 1000
    except Exception:
        return {
            "function_calls": [],
            "total_time_ms": (time.time() - start) * 1000,
            "confidence": 0,
        }

    try:
        raw = json.loads(raw_str)
    except (json.JSONDecodeError, TypeError):
        return {"function_calls": [], "total_time_ms": elapsed, "confidence": 0}

    calls = raw.get("function_calls", [])
    tool_map = {t["name"]: t for t in tools}
    sanitized = []
    for call in calls:
        name = call.get("name", "")
        args = call.get("arguments", {})
        if name in tool_map:
            args = _sanitize_args(args, tool_map[name], user_text)
        sanitized.append({"name": name, "arguments": args})

    return {
        "function_calls": sanitized,
        "total_time_ms": raw.get("total_time_ms", elapsed),
        "confidence": raw.get("confidence", 0),
    }


# ══════════════════════════════════════════════════════════════════════
# MULTI-SIGNAL CONFIDENCE SCORING
# ══════════════════════════════════════════════════════════════════════

def _is_confident(result, tools, user_text=""):
    """Multi-signal confidence scoring. General-purpose."""
    calls = result.get("function_calls", [])
    if not calls:
        return False

    valid_names = {t["name"] for t in tools}
    tool_map = {t["name"]: t for t in tools}
    allowed_nums = _extract_numbers(user_text) if user_text else set()
    user_words = _extract_words(user_text) if user_text else set()

    for call in calls:
        name = call.get("name", "")
        args = call.get("arguments", {})

        # ── STRUCTURAL checks ──
        if name not in valid_names:
            return False

        required = tool_map[name].get("parameters", {}).get("required", [])
        for param in required:
            if param not in args:
                return False
            val = args[param]
            if val is None or val == "" or val == "None":
                return False

        # ── NUMERIC GROUNDING ──
        for key, val in args.items():
            if isinstance(val, (int, float)):
                v = int(val)
                if v != 0 and v not in allowed_nums:
                    return False
            if isinstance(val, str):
                for d in re.findall(r'\b\d+\b', val):
                    d_int = int(d)
                    if d_int != 0 and d_int not in allowed_nums:
                        return False

        # ── STRING QUALITY ──
        for key, val in args.items():
            if not isinstance(val, str):
                continue
            if _GARBLED_RE.search(val):
                return False
            if re.match(r'\d{4}-\d{2}-\d{2}T', val):
                return False
            if '@' in val:
                return False
            if re.search(r'(person_name|example|placeholder|unknown|user_name|name_here|_to_be_)', val, re.IGNORECASE):
                return False
            if len(val) > 100:
                return False

        # ── VALUE GROUNDING ──
        for key, val in args.items():
            if not isinstance(val, str) or len(val) < 3:
                continue
            val_words = {w.lower() for w in re.findall(r"[a-zA-Z'-]+", val) if len(w) > 2}
            if val_words and user_words:
                overlap = val_words & user_words
                if len(overlap) == 0:
                    return False
                extra_words = val_words - user_words
                if len(extra_words) > 1 and len(extra_words) > len(overlap):
                    return False

    # ── CACTUS LOGIT CONFIDENCE ──
    if result.get("confidence", 0) < 0.80:
        return False

    return True


def _validate_with_fusion(result, tool_def, user_text):
    """
    FUSION: Model picks the tool, extraction provides the args.
    When extraction has all required params, ALWAYS use extracted args.
    """
    calls = result.get("function_calls", [])
    if not calls:
        return result, False

    call = calls[0]
    extracted = _extract_args_from_text(tool_def, user_text)

    if not extracted:
        return result, True

    required = tool_def.get("parameters", {}).get("required", [])
    all_extracted = all(param in extracted for param in required)

    if all_extracted:
        corrected_call = {"name": call["name"], "arguments": extracted}
        corrected_result = dict(result)
        corrected_result["function_calls"] = [corrected_call]
        return corrected_result, True
    else:
        corrected_args = dict(call.get("arguments", {}))
        for param, ext_val in extracted.items():
            corrected_args[param] = ext_val
        corrected_call = {"name": call["name"], "arguments": corrected_args}
        corrected_result = dict(result)
        corrected_result["function_calls"] = [corrected_call]
        return corrected_result, False


# ══════════════════════════════════════════════════════════════════════
# GUIDED INFERENCE + RETRY PIPELINE
# ══════════════════════════════════════════════════════════════════════

# Configs: (rag_top_k, temperature, sampling_top_k)
_ATTEMPT_CONFIGS = [
    (2, 0.0, 1),   # Deterministic with tool RAG
    (0, 0.1, 1),   # No RAG, slight temperature
    (0, 0.3, 1),   # Higher temp to unstick model
]


def _try_on_device(task_text, tools, max_attempts=3):
    """
    Optimized on-device strategy:
    Phase 1: Guided inference — keyword identifies tool, extraction provides args
             Single model call for confirmation, immediate fusion if extraction complete
    Phase 2: Full tool set with retries (only if Phase 1 fails)
    Phase 3: Pure fusion fallback — skip model entirely
    """
    task_msgs = [{"role": "user", "content": task_text}]
    total_time = 0

    # ── PHASE 1: Guided inference ──
    guided_tool = _identify_single_tool(task_text, tools)
    if guided_tool:
        single_tools = [guided_tool]

        # Pre-compute extraction
        extracted = _extract_args_from_text(guided_tool, task_text)
        required = guided_tool.get("parameters", {}).get("required", [])
        all_extracted = extracted and all(param in extracted for param in required)

        # Single model call for tool confirmation
        r = _run_local(task_msgs, single_tools, rag_top_k=0, temperature=0.0, sampling_top_k=1)
        total_time += r.get("total_time_ms", 0)

        calls = r.get("function_calls", [])
        got_right_tool = calls and calls[0].get("name") == guided_tool["name"]

        # FAST PATH: model confirms tool + extraction has all args → done immediately
        if all_extracted and (got_right_tool or r.get("confidence", 0) > 0.5):
            fusion_result = dict(r)
            fusion_result["function_calls"] = [{"name": guided_tool["name"], "arguments": extracted}]
            fusion_result["confidence"] = max(r.get("confidence", 0), 0.95)
            if _is_confident(fusion_result, single_tools, task_text):
                return fusion_result, total_time, True

        # Model output passes confidence directly
        if _is_confident(r, single_tools, task_text):
            r_fused, valid = _validate_with_fusion(r, guided_tool, task_text)
            if valid:
                return r_fused, total_time, True

        # One retry with temperature
        r = _run_local(task_msgs, single_tools, rag_top_k=0, temperature=0.1, sampling_top_k=1)
        total_time += r.get("total_time_ms", 0)

        calls = r.get("function_calls", [])
        got_right_tool = calls and calls[0].get("name") == guided_tool["name"]

        if all_extracted and (got_right_tool or r.get("confidence", 0) > 0.5):
            fusion_result = dict(r)
            fusion_result["function_calls"] = [{"name": guided_tool["name"], "arguments": extracted}]
            fusion_result["confidence"] = max(r.get("confidence", 0), 0.95)
            if _is_confident(fusion_result, single_tools, task_text):
                return fusion_result, total_time, True

        if _is_confident(r, single_tools, task_text):
            r_fused, valid = _validate_with_fusion(r, guided_tool, task_text)
            if valid:
                return r_fused, total_time, True

        # Pure fusion: extraction complete, trust it even if model is confused
        if all_extracted:
            fusion_result = {
                "function_calls": [{"name": guided_tool["name"], "arguments": extracted}],
                "total_time_ms": total_time,
                "confidence": 0.95,
            }
            if _is_confident(fusion_result, single_tools, task_text):
                return fusion_result, total_time, True

    # ── PHASE 2: Full tool set with retries (only reached if Phase 1 fails) ──
    best_result = None
    for i, (rag_k, temp, top_k) in enumerate(_ATTEMPT_CONFIGS[:max_attempts]):
        r = _run_local(task_msgs, tools, rag_top_k=rag_k, temperature=temp, sampling_top_k=top_k)
        total_time += r.get("total_time_ms", 0)

        if _is_confident(r, tools, task_text):
            calls = r.get("function_calls", [])
            if calls:
                tool_name = calls[0].get("name", "")
                tool_def = next((t for t in tools if t["name"] == tool_name), None)
                if tool_def:
                    r_fused, valid = _validate_with_fusion(r, tool_def, task_text)
                    if valid:
                        if best_result is None:
                            best_result = r_fused
                            continue
                        else:
                            c1 = best_result.get("function_calls", [])
                            c2 = r_fused.get("function_calls", [])
                            if c1 and c2 and c1[0].get("name") == c2[0].get("name"):
                                return best_result, total_time, True

    if best_result is not None:
        return best_result, total_time, True

    # ── PHASE 3: Pure fusion fallback ──
    if guided_tool:
        extracted = _extract_args_from_text(guided_tool, task_text)
        if extracted:
            required = guided_tool.get("parameters", {}).get("required", [])
            if all(param in extracted for param in required):
                fusion_result = {
                    "function_calls": [{"name": guided_tool["name"], "arguments": extracted}],
                    "total_time_ms": total_time,
                    "confidence": 0.95,
                }
                if _is_confident(fusion_result, [guided_tool], task_text):
                    return fusion_result, total_time, True

    r_last = r if 'r' in dir() else {"function_calls": [], "total_time_ms": 0, "confidence": 0}
    return r_last, total_time, False


# ══════════════════════════════════════════════════════════════════════
# DECOMPOSITION + PRONOUN RESOLUTION + DEDUPLICATION
# ══════════════════════════════════════════════════════════════════════

def _chunk_query(text):
    """Split multi-action queries into discrete single-action chunks."""
    verbs = r'(set|get|send|check|play|find|search|create|text|remind|look|wake|tell|make|open|close|start|stop|turn|call|add|remove|delete|update|show|hide|enable|disable|book|lock|unlock|read|take|log|translate|convert|track|order|navigate|drive|ride|dim|adjust|schedule)'

    pattern = re.compile(
        r'(?:,\s*and\s+|\s+and\s+(?:also\s+|then\s+)?|\s+then\s+|,\s+)(?=' + verbs + r'\b)',
        re.IGNORECASE
    )

    chunks = pattern.split(text)
    clean_chunks = [c.strip() for c in chunks
                    if c and (not re.match(verbs + r'$', c.strip(), re.IGNORECASE)) and len(c.strip()) > 3]

    if not clean_chunks:
        return [text]
    return clean_chunks


def _resolve_pronouns(chunks):
    """Resolve pronouns between chunks using names from previous chunks."""
    resolved = []
    prev_name = None

    for chunk in chunks:
        chunk_lower = chunk.lower()

        names = re.findall(r'\b([A-Z][a-z]+)\b', chunk)
        common_starts = {'Set', 'Get', 'Send', 'Check', 'Play', 'Find', 'Search',
                         'Create', 'Text', 'Remind', 'Look', 'Wake', 'Tell', 'Make',
                         'What', 'How', 'The', 'And', 'Then', 'Also'}
        real_names = [n for n in names if n not in common_starts]
        if real_names:
            prev_name = real_names[0]

        if prev_name and re.search(r'\b(him|her|them)\b', chunk_lower):
            chunk = re.sub(r'\bhim\b', prev_name, chunk, flags=re.IGNORECASE)
            chunk = re.sub(r'\bher\b', prev_name, chunk, flags=re.IGNORECASE)
            chunk = re.sub(r'\bthem\b', prev_name, chunk, flags=re.IGNORECASE)

        resolved.append(chunk)

    return resolved


def _deduplicate_calls(calls):
    """Remove duplicate function calls (same name + same args)."""
    seen = set()
    deduped = []
    for call in calls:
        key = (call["name"], json.dumps(call.get("arguments", {}), sort_keys=True))
        if key not in seen:
            seen.add(key)
            deduped.append(call)
    return deduped


# ══════════════════════════════════════════════════════════════════════
# CLOUD — gemini-2.5-flash-lite for fastest F1=1.0
# ══════════════════════════════════════════════════════════════════════

_CLOUD_SYSTEM_PROMPT = (
    "You are a precise function-calling assistant. Rules:\n"
    "1. Call ALL functions needed to fulfill the user's COMPLETE request.\n"
    "2. If the user asks for multiple actions (e.g. 'do X and Y'), make MULTIPLE parallel function calls.\n"
    "3. Extract argument values EXACTLY from the user's words with NO additions:\n"
    "   - Names: use exact spelling and case from the query\n"
    "   - Messages: extract the MINIMAL exact text. NEVER add a trailing period.\n"
    "   - Songs: extract EXACTLY what comes after 'play' (minus 'some'). 'Play jazz music'->song='jazz', 'play classical music'->song='classical music', 'play summer hits'->song='summer hits'\n"
    "   - Times: use natural format like '3:00 PM', not ISO-8601\n"
    "   - Reminders: 'remind me about X' -> title=X (no leading 'the'). 'remind me to X' -> title=X (full phrase)\n"
    "4. Never invent, paraphrase, or add information not in the query.\n"
    "5. 'text'/'message' -> send_message. 'wake me up' -> set_alarm. 'find'/'look up' + contact -> search_contacts.\n"
    "6. For set_alarm: '6 AM' -> hour=6, minute=0. '8:15 AM' -> hour=8, minute=15."
)


def _build_gemini_tools(tools):
    """Build Gemini tool declarations from our tool definitions."""
    return [
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


def _postprocess_cloud_calls(calls, user_text):
    """General-purpose postprocessing for cloud results."""
    processed = []
    for call in calls:
        name = call.get("name", "")
        args = dict(call.get("arguments", {}))

        for key, val in args.items():
            if isinstance(val, float) and val == int(val):
                args[key] = int(val)
        for key, val in args.items():
            if isinstance(val, str):
                args[key] = val.rstrip('.')
        for key, val in args.items():
            if isinstance(val, str) and 'T' in val and key == "time":
                m = re.match(r'.*T(\d{1,2}):(\d{2})', val)
                if m:
                    h, mn = int(m.group(1)), int(m.group(2))
                    ampm = "AM" if h < 12 else "PM"
                    if h > 12: h -= 12
                    if h == 0: h = 12
                    args[key] = f"{h}:{mn:02d} {ampm}"

        if name == "create_reminder" and "title" in args:
            if re.search(r'remind\s+(?:me\s+)?about\b', user_text, re.IGNORECASE):
                args["title"] = re.sub(r'^(the|a|an)\s+', '', str(args["title"]), flags=re.IGNORECASE).strip()

        if name == "play_music" and "song" in args:
            if re.search(r'play\s+some\s+', user_text, re.IGNORECASE):
                args["song"] = re.sub(r'\s+music\s*$', '', str(args["song"]), flags=re.IGNORECASE).strip()

        processed.append({"name": name, "arguments": args})
    return processed


def generate_cloud(messages, tools):
    """Cloud via fastest Gemini model with strong prompting for F1=1.0."""
    client = _get_gemini_client()
    gemini_tools = _build_gemini_tools(tools)

    user_text = " ".join(m["content"] for m in messages if m["role"] == "user")
    contents = [m["content"] for m in messages if m["role"] == "user"]
    start = time.time()

    for model_name in ["gemini-3-flash-preview", "gemini-2.5-flash-lite", "gemini-2.0-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash"]:
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=gemini_tools,
                    system_instruction=_CLOUD_SYSTEM_PROMPT,
                ),
            )
            elapsed = (time.time() - start) * 1000

            calls = []
            for candidate in resp.candidates:
                for part in candidate.content.parts:
                    if part.function_call:
                        calls.append({
                            "name": part.function_call.name,
                            "arguments": dict(part.function_call.args),
                        })

            calls = _postprocess_cloud_calls(calls, user_text)

            return {
                "function_calls": calls,
                "total_time_ms": elapsed,
                "source": "cloud",
            }

        except Exception:
            continue

    return {
        "function_calls": [],
        "total_time_ms": (time.time() - start) * 1000,
        "source": "cloud",
    }


# ══════════════════════════════════════════════════════════════════════
# MAIN HYBRID ROUTING
# ══════════════════════════════════════════════════════════════════════

def generate_hybrid(messages, tools, confidence_threshold=0.85):
    """
    Optimized hybrid routing:
    1. Decompose multi-action queries into chunks + pronoun resolution
    2. Phase 1: Guided inference (single tool + fusion)
    3. Phase 2: Full tool set with retries + dual-check
    4. Phase 3: Pure fusion fallback
    5. If ANY chunk fails → send FULL query to cloud
    6. Deduplicate results
    """
    user_text = " ".join(m["content"] for m in messages if m["role"] == "user")
    accumulated_time = 0

    sub_tasks = _chunk_query(user_text)
    sub_tasks = _resolve_pronouns(sub_tasks)

    all_calls = []
    cloud_fallback_needed = False
    on_device_count = 0
    total_tasks = len(sub_tasks)

    for task in sub_tasks:
        result, task_time, confident = _try_on_device(task, tools, max_attempts=3)
        accumulated_time += task_time

        if confident:
            all_calls.extend(result["function_calls"])
            on_device_count += 1
        else:
            cloud_fallback_needed = True
            break

    # ── CLOUD FALLBACK ────────────────────────────────────────
    if cloud_fallback_needed:
        cloud_msgs = [{"role": "user", "content": user_text}]
        cloud = generate_cloud(cloud_msgs, tools)
        accumulated_time += cloud.get("total_time_ms", 0)
        all_calls = cloud.get("function_calls", [])
        on_device_count = 0

    # Deduplicate
    all_calls = _deduplicate_calls(all_calls)

    if on_device_count == total_tasks:
        source = "on-device"
    elif on_device_count == 0:
        source = "cloud"
    else:
        source = "hybrid"

    return {
        "function_calls": all_calls,
        "total_time_ms": accumulated_time,
        "source": source,
    }


# ══════════════════════════════════════════════════════════════════════
# Convenience exports (benchmark.py compatibility)
# ══════════════════════════════════════════════════════════════════════

def generate_cactus(messages, tools):
    return _run_local(messages, tools)


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


if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string", "description": "City name"}},
            "required": ["location"],
        },
    }]
    messages = [{"role": "user", "content": "What is the weather in San Francisco?"}]

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid", hybrid)
