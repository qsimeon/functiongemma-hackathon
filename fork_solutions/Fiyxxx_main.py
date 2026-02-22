# Source: https://github.com/Fiyxxx/functiongemma-hackathon
"""
Hybrid function-calling router for the Google DeepMind x Cactus Compute hackathon.

Two-phase routing: pre-inference heuristics decide obvious cloud cases,
post-inference validation catches FunctionGemma errors and escalates to Gemini.
"""

import sys
sys.path.insert(0, "cactus/python/src")

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from cactus import cactus_init, cactus_complete, cactus_embed, cactus_destroy, cactus_reset
import urllib.request
import urllib.error


# ===========================================================================
# SmartRouter (inlined — submit.py only uploads main.py)
# ===========================================================================

EmbedFn = Callable[[str], List[float]]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class SeedEntry:
    text: str
    tool_count: int
    embedding: Optional[List[float]] = field(default=None, repr=False)


class InMemoryVectorStore:
    def __init__(self):
        self._entries: List[SeedEntry] = []

    def add(self, entry: SeedEntry):
        self._entries.append(entry)

    def search(self, query_vec: List[float], top_k: int = 3):
        scored = [
            (e, _cosine_similarity(query_vec, e.embedding))
            for e in self._entries
            if e.embedding is not None
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


@dataclass
class RoutingDecision:
    route: str
    reason: str
    multi_tool_score: float
    blended_score: float


@dataclass
class PostInferenceResult:
    should_escalate: bool
    confidence: float
    reason: str


# Seed corpus: general routing exemplars for embedding similarity.
# These are NOT hardcoded answers — they guide local-vs-cloud routing only.
SEED_CORPUS = [
    # Single-tool patterns (route local when possible)
    {"text": "What's the weather like?", "tool_count": 1},
    {"text": "How's the temperature outside?", "tool_count": 1},
    {"text": "Set an alarm for the morning", "tool_count": 1},
    {"text": "Wake me up early", "tool_count": 1},
    {"text": "Play a song", "tool_count": 1},
    {"text": "Put on some music", "tool_count": 1},
    {"text": "Send a message to someone", "tool_count": 1},
    {"text": "Text my friend", "tool_count": 1},
    {"text": "Remind me about something later", "tool_count": 1},
    {"text": "Create a reminder for an appointment", "tool_count": 1},
    {"text": "Set a timer", "tool_count": 1},
    {"text": "Start a countdown", "tool_count": 1},
    {"text": "Search for a contact", "tool_count": 1},
    {"text": "Look up someone's number", "tool_count": 1},

    # Multi-tool patterns (route to cloud)
    {"text": "Set an alarm and check the weather", "tool_count": 2},
    {"text": "Send a message and set a reminder to follow up", "tool_count": 2},
    {"text": "Set a timer, play music, and remind me about something", "tool_count": 3},
    {"text": "Check the weather and play some music", "tool_count": 2},
    {"text": "Set an alarm, check the weather, and send a message", "tool_count": 3},
    {"text": "Remind me about something and text someone", "tool_count": 2},
    {"text": "Play music and set a timer", "tool_count": 2},
    {"text": "Search for a contact and send them a message", "tool_count": 2},
    {"text": "Get the weather and set a reminder", "tool_count": 2},
    {"text": "Send a message, set a timer, and play music", "tool_count": 3},
    {"text": "Check the weather and set an alarm for tomorrow", "tool_count": 2},
    {"text": "Message someone about plans and remind me to prepare", "tool_count": 2},
]

# Pre-compiled regex patterns
_CONJUNCTIONS = re.compile(r"\b(and|then|also|plus|after that|as well|additionally)\b", re.I)
_ACTION_VERBS = re.compile(
    r"\b(set|send|play|get|check|search|find|create|remind|text|message|call|"
    r"start|timer|alarm|weather|look up|tell me|wake|put on)\b", re.I
)
_COMMA_COMMANDS = re.compile(r",\s*(?:and\s+)?(?:then\s+)?", re.I)
_DATE_PATTERN = re.compile(r"\d{4}-\d{1,2}-\d{1,2}")
_TIME_PATTERN = re.compile(
    r"(\d{1,2})(?::(\d{2}))?\s*(AM|PM|am|pm|a\.m\.|p\.m\.)\b"
)
_MINUTES_PATTERN = re.compile(r"(\d+)\s*(?:minute|min)", re.I)
_PLAY_PATTERN = re.compile(r"\bplay\s+(.+?)(?:\s+(?:on|from|for me|please))?\.?\s*$", re.I)


class SmartRouter:
    def __init__(self, embed_fn: Optional[EmbedFn] = None):
        self._embed_fn = embed_fn
        self._store = InMemoryVectorStore()

        for s in SEED_CORPUS:
            entry = SeedEntry(text=s["text"], tool_count=s["tool_count"])
            if embed_fn is not None:
                try:
                    entry.embedding = embed_fn(entry.text)
                except Exception:
                    pass
            self._store.add(entry)

    def _score_multi_tool(self, query: str, tools: list) -> float:
        score = 0.0
        conjunctions = len(_CONJUNCTIONS.findall(query))
        if conjunctions >= 2:
            score += 0.45
        elif conjunctions >= 1:
            score += 0.30

        verbs = set(v.lower() for v in _ACTION_VERBS.findall(query))
        if len(verbs) >= 3:
            score += 0.35
        elif len(verbs) >= 2:
            score += 0.20

        commas = len(_COMMA_COMMANDS.findall(query))
        if commas >= 2:
            score += 0.20
        elif commas >= 1:
            score += 0.10

        if len(tools) >= 5:
            score += 0.05

        return min(score, 1.0)

    def _score_complexity(self, query: str, tools: list) -> float:
        score = 0.0
        if len(tools) >= 5:
            score += 0.20
        elif len(tools) >= 3:
            score += 0.10
        words = len(query.split())
        if words >= 15:
            score += 0.20
        elif words >= 10:
            score += 0.10
        numbers = re.findall(r"\b\d+\b", query)
        if len(numbers) >= 3:
            score += 0.10
        return min(score, 1.0)

    def _score_similarity(self, query: str):
        if self._embed_fn is None:
            return 0.0
        try:
            query_vec = self._embed_fn(query)
        except Exception:
            return 0.0
        results = self._store.search(query_vec, top_k=3)
        if not results:
            return 0.0
        best_entry, best_sim = results[0]
        sim_score = max(0.0, best_sim - 0.5) * 2.0
        if best_entry.tool_count >= 2 and best_sim >= 0.70:
            sim_score = max(sim_score, 0.8)
        multi_matches = sum(1 for e, s in results if e.tool_count >= 2 and s >= 0.60)
        if multi_matches >= 2:
            sim_score = max(sim_score, 0.6)
        return min(sim_score, 1.0)

    def should_route_to_cloud(self, query: str, tools: list) -> RoutingDecision:
        multi = self._score_multi_tool(query, tools)
        complexity = self._score_complexity(query, tools)
        sim_score = self._score_similarity(query)

        if sim_score >= 0.6:
            multi = min(1.0, multi + sim_score * 0.3)

        # Weighted blend: multi-tool detection is dominant signal
        blended = 0.55 * multi + 0.20 * sim_score + 0.25 * complexity

        threshold = 0.55
        if blended >= threshold:
            return RoutingDecision("cloud", f"blended={blended:.3f}>={threshold}", multi, blended)
        return RoutingDecision("local", f"blended={blended:.3f}<{threshold}", multi, blended)

    def post_inference_gate(
        self, confidence: float, cloud_handoff: bool,
        function_calls: list, pre_decision: RoutingDecision,
        tools: Optional[list] = None,
        query: str = "",
    ) -> PostInferenceResult:

        if not function_calls:
            return PostInferenceResult(True, confidence, "no function calls")

        if cloud_handoff:
            return PostInferenceResult(True, confidence, "cloud_handoff flag")

        # Multiple function calls = FunctionGemma unreliable here
        if len(function_calls) > 1:
            return PostInferenceResult(True, confidence, "multi-call escalate")

        call = function_calls[0]
        call_name = call.get("name", "")
        call_args = call.get("arguments", {})

        # Validate function name exists in tool set and check schema
        tool_schema = None
        if tools:
            tool_map = {t["name"]: t for t in tools}
            if call_name not in tool_map:
                return PostInferenceResult(True, confidence, f"unknown tool '{call_name}'")
            tool_schema = tool_map[call_name]

        # Validate required parameters are present
        if tool_schema:
            props = tool_schema.get("parameters", {}).get("properties", {})
            required = tool_schema.get("parameters", {}).get("required", [])
            for req_param in required:
                if req_param not in call_args:
                    return PostInferenceResult(True, confidence, f"missing required param '{req_param}'")
            # Validate types match schema
            for k, v in call_args.items():
                if k in props:
                    expected_type = props[k].get("type", "")
                    if expected_type == "integer" and not isinstance(v, int):
                        return PostInferenceResult(True, confidence, f"type mismatch: {k} should be int")
                    if expected_type == "string" and not isinstance(v, str):
                        return PostInferenceResult(True, confidence, f"type mismatch: {k} should be str")

        # Validate argument sanity
        for k, v in call_args.items():
            # Negative numbers
            if isinstance(v, (int, float)) and v < 0:
                return PostInferenceResult(True, confidence, f"negative {k}={v}")
            # Unreasonable hour values (valid: 0-23)
            if k == "hour" and isinstance(v, (int, float)) and (v > 23 or v < 0):
                return PostInferenceResult(True, confidence, f"bad hour={v}")
            # Unreasonable minute values (valid: 0-59)
            if k == "minute" and isinstance(v, (int, float)) and (v > 59 or v < 0):
                return PostInferenceResult(True, confidence, f"bad minute={v}")
            # Unreasonable timer values
            if k == "minutes" and isinstance(v, (int, float)) and (v <= 0 or v > 1440):
                return PostInferenceResult(True, confidence, f"bad minutes={v}")
            # Hallucinated ISO dates
            if isinstance(v, str) and _DATE_PATTERN.search(v):
                return PostInferenceResult(True, confidence, f"hallucinated date in {k}")
            # Empty strings
            if isinstance(v, str) and not v.strip():
                return PostInferenceResult(True, confidence, f"empty {k}")

        # If pre-inference detected multi-tool but we got single call
        if pre_decision.multi_tool_score >= 0.45:
            return PostInferenceResult(True, confidence, "multi-tool detected, single call returned")

        # --- Query-output consistency checks ---
        # Parse expected values from query and compare with model output.
        # This catches FunctionGemma errors where values are in valid range but wrong.
        if query:
            if call_name == "set_alarm":
                m = _TIME_PATTERN.search(query)
                if m:
                    q_hour = int(m.group(1))
                    q_min = int(m.group(2)) if m.group(2) else 0
                    ampm = m.group(3).upper().replace(".", "")
                    if ampm == "PM" and q_hour != 12:
                        q_hour += 12
                    elif ampm == "AM" and q_hour == 12:
                        q_hour = 0
                    out_hour = call_args.get("hour")
                    out_min = call_args.get("minute")
                    if isinstance(out_hour, int) and isinstance(out_min, int):
                        if out_hour != q_hour or out_min != q_min:
                            return PostInferenceResult(
                                True, confidence,
                                f"alarm mismatch: query={q_hour}:{q_min:02d} vs output={out_hour}:{out_min:02d}"
                            )

            elif call_name == "set_timer":
                m = _MINUTES_PATTERN.search(query)
                if m:
                    q_mins = int(m.group(1))
                    out_mins = call_args.get("minutes")
                    if isinstance(out_mins, int) and out_mins != q_mins:
                        return PostInferenceResult(
                            True, confidence,
                            f"timer mismatch: query={q_mins} vs output={out_mins}"
                        )

            elif call_name == "play_music":
                pm = _PLAY_PATTERN.search(query)
                if pm:
                    q_song = pm.group(1).strip().lower()
                    out_song = call_args.get("song", "").strip().lower()
                    # Check if query song is a substring of output or vice versa
                    if out_song and q_song and q_song not in out_song and out_song not in q_song:
                        return PostInferenceResult(
                            True, confidence,
                            f"song mismatch: query='{q_song}' vs output='{out_song}'"
                        )

        return PostInferenceResult(False, confidence, f"trust local: {call_name}")


# ===========================================================================
# Module-level init (runs once on import)
# ===========================================================================

functiongemma_path = "cactus/weights/functiongemma-270m-it"
functiongemma_model = cactus_init(functiongemma_path)

# Try to load embedding model; gracefully degrade if unavailable
_embed_fn_impl = None
try:
    embed_model_path = "cactus/weights/nomic-embed-text-v2-moe"
    _embed_model = cactus_init(embed_model_path)

    def _embed_fn_impl(text: str) -> list:
        return cactus_embed(_embed_model, text, normalize=True)
except Exception:
    _embed_fn_impl = None

router = SmartRouter(embed_fn=_embed_fn_impl)


# ===========================================================================
# Cloud inference via Gemini REST API (no SDK — avoids subprocess dependency)
# ===========================================================================
_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

_SYSTEM_INSTRUCTION = (
    "You are a tool-calling assistant. Follow these rules precisely:\n"
    "1. For each argument, extract the user's exact key words. Strip ONLY "
    "articles (a, an, the) and fillers (some, that, this) from the start of "
    "values. Keep the rest exactly as spoken. "
    "Examples: 'Remind me about the meeting' → title='meeting'. "
    "'Play Bohemian Rhapsody' → song='Bohemian Rhapsody'. "
    "'Play some jazz' → song='jazz'. "
    "'Play classical music' → song='classical music'.\n"
    "2. For time arguments, use the user's exact format (e.g. '3:00 PM').\n"
    "3. For message content, use the user's exact words after 'saying'/'that'/"
    "'says'. Do not add or change any words.\n"
    "4. If the user requests multiple actions, call ALL relevant tools in "
    "a single response. 'find X and send them a message' needs BOTH "
    "search_contacts AND send_message."
)


def _gemini_type(t: str) -> str:
    """Map JSON Schema type to Gemini API type."""
    return {"string": "STRING", "integer": "INTEGER", "number": "NUMBER",
            "boolean": "BOOLEAN", "array": "ARRAY", "object": "OBJECT"}.get(t, "STRING")


def _call_gemini_rest(model_name: str, contents: list, tools: list) -> dict:
    """Call Gemini REST API directly using urllib (no SDK needed)."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent?key={_GEMINI_API_KEY}"
    )

    # Build tool declarations
    function_declarations = []
    for t in tools:
        props = {}
        for k, v in t["parameters"]["properties"].items():
            props[k] = {"type": _gemini_type(v["type"]), "description": v.get("description", "")}
        function_declarations.append({
            "name": t["name"],
            "description": t["description"],
            "parameters": {
                "type": "OBJECT",
                "properties": props,
                "required": t["parameters"].get("required", []),
            },
        })

    body = {
        "contents": [{"parts": [{"text": c}]} for c in contents],
        "tools": [{"functionDeclarations": function_declarations}],
        "systemInstruction": {"parts": [{"text": _SYSTEM_INSTRUCTION}]},
    }

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")

    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud REST API with retry."""
    contents = [m["content"] for m in messages if m["role"] == "user"]
    start_time = time.time()

    models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash"]
    gemini_result = None
    last_error = None

    for model_name in models_to_try:
        for attempt in range(3):
            try:
                gemini_result = _call_gemini_rest(model_name, contents, tools)
                break
            except urllib.error.HTTPError as e:
                last_error = e
                if e.code == 429 and attempt < 2:
                    time.sleep(1 + attempt)
                elif e.code == 404:
                    break
                else:
                    if attempt == 2:
                        break
                    time.sleep(1)
            except Exception as e:
                last_error = e
                if attempt == 2:
                    break
                time.sleep(1)
        if gemini_result is not None:
            break

    if gemini_result is None:
        raise last_error

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    for candidate in gemini_result.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            fc = part.get("functionCall")
            if fc:
                # Gemini returns float for integer params — cast them
                args = fc.get("args", {})
                clean_args = {}
                for k, v in args.items():
                    if isinstance(v, float) and v == int(v):
                        clean_args[k] = int(v)
                    else:
                        clean_args[k] = v
                function_calls.append({"name": fc["name"], "arguments": clean_args})

    return {"function_calls": function_calls, "total_time_ms": total_time_ms}


# ===========================================================================
# Local inference via FunctionGemma + Cactus
# ===========================================================================
def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    cactus_reset(functiongemma_model)
    cactus_tools = [{"type": "function", "function": t} for t in tools]

    raw_str = cactus_complete(
        functiongemma_model,
        [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
        tools=cactus_tools,
        force_tools=True,
        tool_rag_top_k=3,
        confidence_threshold=0.3,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        # Fix FunctionGemma JSON: leading zeros on numbers (e.g. 06 -> 6)
        fixed = re.sub(r'(?<=:)\s*0(\d+)', r' \1', raw_str)
        try:
            raw = json.loads(fixed)
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


# ===========================================================================
# Hybrid routing (entry point evaluated by benchmark.py)
# ===========================================================================
def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """
    SmartRouter-powered hybrid inference.

    Phase 1: Pre-inference heuristics decide if query should go to cloud.
    Phase 2: If local, run FunctionGemma then validate output before trusting.
    """
    # Extract user query
    user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_msg = m.get("content", "")
            break

    # --- Phase 1: Pre-inference routing ---
    pre_decision = router.should_route_to_cloud(user_msg, tools)

    if pre_decision.route == "cloud":
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback)"
        return cloud

    # --- Local inference with retry ---
    total_local_ms = 0
    max_local_attempts = 2

    for attempt in range(max_local_attempts):
        local = generate_cactus(messages, tools)
        total_local_ms += local.get("total_time_ms", 0)

        # Parse function_calls arguments if they are JSON strings
        function_calls = local.get("function_calls", [])
        parsed_calls = []
        for fc in function_calls:
            call = dict(fc)
            if isinstance(call.get("arguments"), str):
                try:
                    call["arguments"] = json.loads(call["arguments"])
                except (json.JSONDecodeError, TypeError):
                    pass
            parsed_calls.append(call)

        # --- Phase 2: Post-inference validation ---
        post = router.post_inference_gate(
            confidence=local.get("confidence", 0),
            cloud_handoff=local.get("cloud_handoff", False),
            function_calls=parsed_calls,
            pre_decision=pre_decision,
            tools=tools,
            query=user_msg,
        )

        if not post.should_escalate:
            # Validated — trust local result
            return {
                "function_calls": parsed_calls,
                "total_time_ms": total_local_ms,
                "confidence": local.get("confidence", 0),
                "source": "on-device",
            }

        # If first attempt failed validation, retry before cloud fallback
        if attempt < max_local_attempts - 1:
            continue

    # All local attempts failed validation — escalate to cloud
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["total_time_ms"] += total_local_ms
    return cloud
