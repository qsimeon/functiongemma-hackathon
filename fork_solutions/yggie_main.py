# Source: https://github.com/yggie/functiongemma-hackathon

import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types

BASE_STOP_WORDS = {
    "a", "an", "the", "is", "in", "at", "to", "for", "of", "on", "my", "me",
    "it", "do", "if", "so", "up", "or", "by", "be", "as", "no", "not",
    "what", "how", "can", "will", "would", "could", "should", "some",
    "this", "that", "with", "from", "have", "has", "had",
    "i", "you", "we", "he", "she", "they", "and", "also", "then", "plus",
}

SHORT_VALUE_INDICATORS = {"name", "title", "query", "identifier", "person"}

MIN_PREFIX_LEN = 4


def _compute_stop_words(tools):
    """Extend base stop words with words shared across multiple tool names.
    E.g., if tools are set_alarm and set_timer, 'set' is stopped automatically."""
    stops = set(BASE_STOP_WORDS)
    word_counts = {}
    for tool in tools:
        for part in tool["name"].split("_"):
            if len(part) > 2:
                word_counts[part] = word_counts.get(part, 0) + 1
    for word, count in word_counts.items():
        if count > 1:
            stops.add(word)
    return stops


def _prefix_match(a, b):
    """Check if two words match exactly or share a prefix of MIN_PREFIX_LEN+ chars."""
    if len(a) < MIN_PREFIX_LEN or len(b) < MIN_PREFIX_LEN:
        return a == b
    return a[:MIN_PREFIX_LEN] == b[:MIN_PREFIX_LEN]


def _prefix_overlap(words_a, words_b):
    """Count words in words_a that have a prefix match in words_b."""
    count = 0
    for a in words_a:
        for b in words_b:
            if _prefix_match(a, b):
                count += 1
                break
    return count


def _extract_words(text, stop_words=None):
    """Extract meaningful lowercase words, splitting underscores and filtering noise."""
    if stop_words is None:
        stop_words = BASE_STOP_WORDS
    # Split on any non-alpha character to extract word tokens
    tokens = []
    current = []
    for ch in text.lower():
        if ch.isalpha():
            current.append(ch)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
    return {w for w in tokens if len(w) > 2 and w not in stop_words}


def _tool_vocab(tool, stop_words=None):
    """Build a word set from a tool's name, description, and parameter metadata."""
    parts = [
        tool["name"].replace("_", " "),
        tool.get("description", ""),
    ]
    for param_info in tool.get("parameters", {}).get("properties", {}).values():
        parts.append(param_info.get("description", ""))
    return _extract_words(" ".join(parts), stop_words=stop_words)


def _vocab_overlap_suspicious(local_result, messages, tools):
    """Check if the model's tool selection is suspicious via vocabulary analysis.

    Two checks:
    1. Wrong tool: an unchosen tool matches the prompt better than every chosen tool.
    2. Missed tool: prompt keywords not covered by chosen tools prefix-match an
       unchosen tool's vocabulary, suggesting the model missed a needed call.
    """
    called_names = {c["name"] for c in local_result.get("function_calls", [])}

    if not called_names or len(tools) <= 1:
        return False

    stop_words = _compute_stop_words(tools)

    user_text = " ".join(m["content"] for m in messages if m["role"] == "user")
    prompt_words = _extract_words(user_text, stop_words=stop_words)

    if not prompt_words:
        return False

    chosen_tools = [t for t in tools if t["name"] in called_names]
    unchosen_tools = [t for t in tools if t["name"] not in called_names]

    if not chosen_tools or not unchosen_tools:
        return False

    chosen_vocabs = [_tool_vocab(t, stop_words=stop_words) for t in chosen_tools]
    unchosen_vocabs = [_tool_vocab(t, stop_words=stop_words) for t in unchosen_tools]

    # Wrong tool: unchosen tool has better overlap than every chosen tool
    chosen_scores = [_prefix_overlap(prompt_words, v) for v in chosen_vocabs]
    unchosen_scores = [_prefix_overlap(prompt_words, v) for v in unchosen_vocabs]

    if max(unchosen_scores) > max(chosen_scores):
        return True

    # Missed tool: find prompt words not covered by any chosen tool,
    # then check if an unchosen tool matches the uncovered remainder
    covered = set()
    for pw in prompt_words:
        for vocab in chosen_vocabs:
            if any(_prefix_match(pw, tw) for tw in vocab):
                covered.add(pw)
                break
    uncovered = prompt_words - covered

    if uncovered:
        for vocab in unchosen_vocabs:
            if _prefix_overlap(uncovered, vocab) > 0:
                return True

    return False


def _is_word_boundary(text, start, end):
    """Check if text[start:end] sits on word boundaries (not embedded in a larger word)."""
    if start > 0 and text[start - 1].isalnum():
        return False
    if end < len(text) and text[end].isalnum():
        return False
    return True


def _word_boundary_match(needle, haystack):
    """Find needle in haystack and verify it's at word boundaries."""
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            return False
        if _is_word_boundary(haystack, idx, idx + len(needle)):
            return True
        start = idx + 1


def _arguments_grounded(local_result, messages):
    """Check that argument values from the model's function calls are grounded in the
    user prompt. Uses ratio-based checking: a majority of values must be present.
    Returns False if too many values appear hallucinated."""
    user_text = " ".join(m["content"] for m in messages if m["role"] == "user")
    user_text_lower = user_text.lower()

    total = 0
    grounded = 0

    for call in local_result.get("function_calls", []):
        for key, value in call.get("arguments", {}).items():
            if isinstance(value, str):
                val = value.strip().lower()
                total += 1
                if len(val) < MIN_PREFIX_LEN:
                    # Short strings: word-boundary match to avoid false substrings
                    # e.g. "hi" matching inside "this"
                    if _word_boundary_match(val, user_text_lower):
                        grounded += 1
                else:
                    if val in user_text_lower:
                        grounded += 1
            elif isinstance(value, (int, float)):
                # Skip 0 as it's often an implied default (e.g. minute=0 for "10 AM")
                if value == 0:
                    continue
                total += 1
                # Word-boundary match so "8" matches in "8:15" but not in "18"
                if _word_boundary_match(str(int(value)), user_text):
                    grounded += 1

    if total == 0:
        return True

    # Majority of arguments must be grounded in the prompt
    return grounded >= (total + 1) // 2


TYPE_VALIDATORS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, (int, float)) and (isinstance(v, int) or v == int(v)),
    "number": lambda v: isinstance(v, (int, float)),
    "boolean": lambda v: isinstance(v, bool),
}


def _structurally_valid(local_result, tools):
    """Validate function calls against tool schemas: name exists, required params
    present, types match, no duplicate calls."""
    tool_map = {t["name"]: t for t in tools}
    seen = set()

    for call in local_result.get("function_calls", []):
        name = call.get("name", "")
        args = call.get("arguments", {})

        # Tool name must exist in available tools
        if name not in tool_map:
            return False

        tool = tool_map[name]
        properties = tool.get("parameters", {}).get("properties", {})
        required = tool.get("parameters", {}).get("required", [])

        # All required parameters must be present
        for req in required:
            if req not in args:
                return False

        # Parameter types must match schema
        for key, value in args.items():
            if key in properties:
                expected_type = properties[key].get("type", "")
                validator = TYPE_VALIDATORS.get(expected_type)
                if validator and not validator(value):
                    return False

        # No duplicate calls
        call_key = (name, tuple(sorted((k, str(v)) for k, v in args.items())))
        if call_key in seen:
            return False
        seen.add(call_key)

    return True


def _arguments_plausible(local_result, tools, messages):
    """Check if argument values are plausible for their parameter descriptions.
    Uses SHORT_VALUE_INDICATORS to detect fields that should have concise values,
    and checks that no value is longer than the entire user prompt."""
    tool_map = {t["name"]: t for t in tools}
    user_text = " ".join(m["content"] for m in messages if m["role"] == "user")
    user_len = len(user_text)

    for call in local_result.get("function_calls", []):
        name = call.get("name", "")
        if name not in tool_map:
            continue

        properties = tool_map[name].get("parameters", {}).get("properties", {})

        for key, value in call.get("arguments", {}).items():
            if key not in properties or not isinstance(value, str):
                continue

            # No argument value should be longer than the user prompt
            if len(value.strip()) > user_len:
                return False

            desc = properties[key].get("description", "").lower()
            key_lower = key.lower()
            word_count = len(value.strip().split())

            # Fields whose key or description indicates a short value
            # (name, title, query, identifier, person) should be concise
            is_short_field = any(
                ind in key_lower or ind in desc
                for ind in SHORT_VALUE_INDICATORS
            )
            if is_short_field and word_count > 3:
                return False

    return True


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
            "cloud_handoff": True,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
        "cloud_handoff": raw.get("cloud_handoff", False),
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


def should_offload_to_cloud(local_result, messages, tools):
    """Determine whether to offload to cloud using combined heuristics.

    Checks (in order):
    1. Cloud handoff signal from Cactus (model's own uncertainty flag)
    2. Empty function calls (force_tools=True means this is a failure)
    3. Structural validation (tool name exists, required params, types, no dupes)
    4. Vocabulary overlap with adaptive stop words (wrong tool / missed tool)
    5. Argument grounding (values must appear in prompt, word-boundary for short strings)
    6. Argument plausibility (values sensible for their parameter descriptions)
    """
    # Cloud handoff: model itself flagged it can't handle this request
    if local_result.get("cloud_handoff", False):
        return True

    # Empty function calls: with force_tools=True the model must produce a call,
    # so an empty list means generation failed
    if not local_result.get("function_calls"):
        return True

    # Structural validation: tool names exist, required params present,
    # types match schema, no duplicate calls
    if not _structurally_valid(local_result, tools):
        return True

    # Vocabulary overlap: wrong tool selection or missed tool detection
    if _vocab_overlap_suspicious(local_result, messages, tools):
        return True

    # Argument grounding: values must be present in the user prompt
    if not _arguments_grounded(local_result, messages):
        return True

    # Argument plausibility: values should make sense for their parameter
    # descriptions (e.g. a "name" field shouldn't contain a full sentence)
    if not _arguments_plausible(local_result, tools, messages):
        return True

    return False


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """Hybrid inference strategy with multi-signal offload decision."""
    local = generate_cactus(messages, tools)

    local["should-offload"] = should_offload_to_cloud(local, messages, tools)
    local["source"] = "on-device"
    return local

    if not should_offload_to_cloud(local, messages, tools):
        local["source"] = "on-device"
        return local

    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local.get("confidence", 0)
    cloud["total_time_ms"] += local["total_time_ms"]
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
