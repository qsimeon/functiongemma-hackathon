# Source: https://github.com/Rosh2403/functiongemma-hackathon
"""
FunctionGemma Hybrid Routing — 4-Agent Architecture
=====================================================
Agent 1 — Complexity Assessor  : Predictive routing signals (Stage 1)
Agent 2 — Local Executor        : Heuristic + FunctionGemma + validation (Stage 2)
Agent 3 — Cloud Executor        : Anthropic Claude / Gemini fallback
Agent 4 — Orchestrator          : Coordinates agents, makes final routing decision
"""

import sys
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

# Paths — set via environment variables so any machine works without editing this file
# Example: export CACTUS_SRC="/your/path/cactus/python/src"
#          export CACTUS_WEIGHTS="/your/path/cactus/weights/functiongemma-270m-it"
_cactus_src     = os.environ.get("CACTUS_SRC",     "cactus/python/src")
functiongemma_path = os.environ.get("CACTUS_WEIGHTS", "weights/functiongemma-270m-it")
sys.path.insert(0, _cactus_src)

# ---------------------------------------------------------------------------
# Lazy-loaded clients
# ---------------------------------------------------------------------------
_CACTUS_API = None
_CACTUS_MODEL = None          # ← cached model handle (avoid init/destroy per call)
_GENAI_API = None
_GENAI_TYPES = None
_ANTHROPIC_CLIENT = None

# Heuristic short-circuit: if heuristic confidence >= this, skip the local model entirely
HEURISTIC_ONLY_THRESHOLD = 0.90


def _load_cactus():
    global _CACTUS_API
    if _CACTUS_API is None:
        try:
            from cactus import cactus_init, cactus_complete, cactus_destroy
            _CACTUS_API = (cactus_init, cactus_complete, cactus_destroy)
        except Exception:
            _CACTUS_API = ()
    return _CACTUS_API


def _get_cactus_model():
    """Return a cached model instance, initialising once if needed."""
    global _CACTUS_MODEL
    cactus_api = _load_cactus()
    if not cactus_api:
        return None
    if _CACTUS_MODEL is None:
        cactus_init, _, _ = cactus_api
        _CACTUS_MODEL = cactus_init(functiongemma_path)
    return _CACTUS_MODEL


def _load_genai():
    global _GENAI_API, _GENAI_TYPES
    if _GENAI_API is None:
        try:
            from google import genai
            from google.genai import types
            _GENAI_API = genai
            _GENAI_TYPES = types
        except Exception:
            _GENAI_API = None
            _GENAI_TYPES = None
    return _GENAI_API, _GENAI_TYPES


def _load_anthropic():
    global _ANTHROPIC_CLIENT
    if _ANTHROPIC_CLIENT is None:
        try:
            import anthropic
            _ANTHROPIC_CLIENT = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        except Exception:
            _ANTHROPIC_CLIENT = None
    return _ANTHROPIC_CLIENT


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _find_time(text: str) -> Optional[Tuple[int, int, str]]:
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", text, re.IGNORECASE)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2) or "0")
    meridiem = m.group(3).upper()
    if meridiem == "PM" and hour != 12:
        hour += 12
    if meridiem == "AM" and hour == 12:
        hour = 0
    return hour, minute, f"{m.group(1)}:{minute:02d} {meridiem}" if m.group(2) else f"{m.group(1)} {meridiem}"


def _split_segments(text: str) -> List[str]:
    normalized = _normalize_space(text)
    segments = re.split(r",\s*|\s+and\s+", normalized, flags=re.IGNORECASE)
    return [s.strip(" .") for s in segments if s.strip(" .")]


def _latest_user_text(messages: List[Dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


# ---------------------------------------------------------------------------
# Intent extractors (tool-agnostic pattern matching)
# ---------------------------------------------------------------------------

def _extract_by_patterns(segment: str, tool: Dict) -> Optional[Dict]:
    """
    Generic extractor: tries to pull required params from text based on tool schema.
    Falls back to named extractors for common intents.
    """
    name = tool.get("name", "")
    props = tool.get("parameters", {}).get("properties", {})
    required = tool.get("parameters", {}).get("required", [])

    # Named extractors for common intents
    named = {
        "get_weather": _extract_weather,
        "set_alarm": _extract_alarm,
        "set_timer": _extract_timer,
        "send_message": _extract_message,
        "create_reminder": _extract_reminder,
        "search_contacts": _extract_search_contacts,
        "play_music": _extract_music,
    }

    if name in named:
        return named[name](segment)

    # Generic: try to find string values for required fields
    args = {}
    for field in required:
        prop = props.get(field, {})
        ftype = prop.get("type", "string")
        if ftype == "string":
            m = re.search(rf"\b{re.escape(field)}\s*[:\-]?\s*([^\.,]+)", segment, re.IGNORECASE)
            if m:
                args[field] = m.group(1).strip()
        elif ftype in ("integer", "number"):
            m = re.search(r"\b(\d+)\b", segment)
            if m:
                args[field] = int(m.group(1))
    return args if all(k in args for k in required) else None


def _extract_weather(segment: str) -> Optional[Dict]:
    if not re.search(r"\b(weather|forecast|temperature|temp)\b", segment, re.IGNORECASE):
        return None
    m = re.search(r"\bin\s+([A-Za-z][A-Za-z\s'\-]+)", segment)
    if not m:
        return None
    return {"location": _normalize_space(m.group(1)).strip("?.!")}


def _extract_alarm(segment: str) -> Optional[Dict]:
    """
    FIX: Broadened trigger patterns to catch 'set an alarm for X' and 'wake me up at X'.
    Previously missed cases where only set_alarm was in the tool list (no competing timer tool).
    """
    if not re.search(
        r"\b(alarm|wake\s+me\s+up|wake\s+up)\b",
        segment, re.IGNORECASE
    ):
        return None
    found = _find_time(segment)
    if not found:
        return None
    hour, minute, _ = found
    return {"hour": hour, "minute": minute}


def _extract_timer(segment: str) -> Optional[Dict]:
    """
    FIX: Stricter check — only match 'timer' or 'countdown', never match alarm keywords.
    This prevents set_timer firing on alarm segments and vice versa.
    """
    if not re.search(r"\b(timer|countdown)\b", segment, re.IGNORECASE):
        return None
    # Explicitly reject if the segment is really about an alarm
    if re.search(r"\b(alarm|wake\s+me\s+up)\b", segment, re.IGNORECASE):
        return None
    m = re.search(r"\b(\d+)\s*(?:minutes?|mins?|hours?|hrs?|seconds?|secs?)\b", segment, re.IGNORECASE)
    if not m:
        m = re.search(r"\b(\d+)\b", segment)
    if not m:
        return None
    return {"minutes": int(m.group(1))}


def _extract_message(segment: str) -> Optional[Dict]:
    if not re.search(r"\b(send|text|message)\b", segment, re.IGNORECASE):
        return None
    recipient = None
    m_to = re.search(r"\bto\s+([A-Z][a-zA-Z'\-]+)\b", segment, re.IGNORECASE)
    m_text = re.search(r"\btext\s+([A-Z][a-zA-Z'\-]+)\b", segment, re.IGNORECASE)
    if m_to:
        recipient = m_to.group(1)
    elif m_text:
        recipient = m_text.group(1)
    message = None
    m_saying = re.search(r"\bsaying\s+(.+)$", segment, re.IGNORECASE)
    if m_saying:
        message = m_saying.group(1).strip(" .")
    if recipient and message:
        return {"recipient": recipient, "message": message}
    return None


def _extract_reminder(segment: str) -> Optional[Dict]:
    if not re.search(r"\bremind me\b", segment, re.IGNORECASE):
        return None
    m = re.search(
        r"\bremind me\s+(?:to|about)?\s*(.+?)\s+at\s+((?:\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm)))",
        segment, re.IGNORECASE,
    )
    if not m:
        return None
    title = re.sub(r"^the\s+", "", m.group(1).strip(" ."), flags=re.IGNORECASE)
    return {"title": _normalize_space(title), "time": m.group(2).upper().replace("  ", " ")}


def _extract_search_contacts(segment: str) -> Optional[Dict]:
    if not re.search(r"\b(find|look up|search)\b", segment, re.IGNORECASE):
        return None
    if not re.search(r"\bcontacts?\b", segment, re.IGNORECASE):
        return None
    m = re.search(r"\b(?:find|look up|search for?)\s+([A-Z][a-zA-Z'\-]+)\b", segment, re.IGNORECASE)
    if not m:
        return None
    return {"query": m.group(1)}


def _extract_music(segment: str) -> Optional[Dict]:
    if not re.search(r"\bplay\b", segment, re.IGNORECASE):
        return None
    m = re.search(r"\bplay\s+(?:some\s+)?(.+)$", segment, re.IGNORECASE)
    if not m:
        return None
    song = m.group(1).strip(" .")
    if re.search(r"\bplay\s+some\s+", segment, re.IGNORECASE):
        song = re.sub(r"\bmusic\b$", "", song, flags=re.IGNORECASE).strip()
    return {"song": song} if song else None


def _required_ok(args: Dict, tool: Dict) -> bool:
    required = tool.get("parameters", {}).get("required", [])
    return all(k in args and args[k] not in (None, "") for k in required)


def _heuristic_route(messages: List[Dict], tools: List[Dict]) -> Tuple[List[Dict], float]:
    text = _latest_user_text(messages)
    if not text:
        return [], 0.0

    tool_map = {t.get("name"): t for t in tools}
    segments = _split_segments(text)
    calls = []
    matched_segments = 0
    last_contact = None

    for segment in segments:
        for tool in tools:
            tool_name = tool.get("name")
            args = _extract_by_patterns(segment, tool)

            if (
                tool_name == "send_message"
                and args is None
                and last_contact
                and re.search(r"\b(send|text|message)\b", segment, re.IGNORECASE)
                and re.search(r"\b(him|her|them)\b", segment, re.IGNORECASE)
            ):
                m_saying = re.search(r"\bsaying\s+(.+)$", segment, re.IGNORECASE)
                if m_saying:
                    args = {"recipient": last_contact, "message": m_saying.group(1).strip(" .")}

            if args and _required_ok(args, tool_map.get(tool_name, tool)):
                calls.append({"name": tool_name, "arguments": args})
                matched_segments += 1
                if tool_name == "search_contacts":
                    last_contact = args.get("query")
                elif tool_name == "send_message":
                    last_contact = args.get("recipient")
                break

    confidence = 0.0
    if segments:
        confidence = min(1.0, (matched_segments / len(segments)) + (0.15 if calls else 0.0))
    return calls, confidence


# ---------------------------------------------------------------------------
# AGENT 1 — Complexity Assessor (Stage 1 predictive signals)
# ---------------------------------------------------------------------------

def agent_complexity_assessor(messages: List[Dict], tools: List[Dict]) -> Dict:
    """
    Produces a Stage-1 routing score from predictive signals BEFORE running any model.
    """
    text = _latest_user_text(messages)
    segments = _split_segments(text) if text else []
    token_count = len(re.findall(r"\b\w+\b", text))

    heuristic_calls, heuristic_conf = _heuristic_route(messages, tools)
    segment_match_rate = (len(heuristic_calls) / len(segments)) if segments else 0.0

    # Semantic complexity sub-signals
    pronoun_refs   = len(re.findall(r"\b(him|her|them|it|that|there)\b", text, re.IGNORECASE))
    cross_step     = len(re.findall(r"\b(then|after|before|if|unless|when)\b", text, re.IGNORECASE))
    ambiguous      = len(re.findall(r"\b(maybe|probably|around|somewhere|something)\b", text, re.IGNORECASE))
    multi_call     = max(0, len(heuristic_calls) - 1)
    length_factor  = max(0, token_count - 12) / 24.0

    semantic_complexity = min(1.0, sum([
        min(1.0, multi_call * 0.22),
        min(1.0, max(0, len(segments) - 1) * 0.18),
        min(1.0, pronoun_refs * 0.15),
        min(1.0, cross_step * 0.20),
        min(1.0, length_factor * 0.35),
        min(1.0, ambiguous * 0.18),
    ]))

    estimated_output_tokens = max(20, len(heuristic_calls) * 30)
    estimated_latency_ms = (estimated_output_tokens / 50) * 1000
    token_speed_score = min(1.0, estimated_latency_ms / 3000)

    tool_count_score = min(1.0, len(tools) / 10.0)

    stage1_difficulty_raw = (
        token_speed_score    * 0.10 +
        tool_count_score     * 0.10 +
        (1 - heuristic_conf) * 0.20 +
        (1 - segment_match_rate) * 0.05 +
        semantic_complexity  * 0.15
    )
    # Keep score on 0..1 scale for stable thresholding across future weight changes.
    stage1_difficulty = max(0.0, min(1.0, stage1_difficulty_raw))

    return {
        "stage1_difficulty":     round(stage1_difficulty, 4),
        "heuristic_calls":       heuristic_calls,
        "heuristic_conf":        round(heuristic_conf, 4),
        "segment_match_rate":    round(segment_match_rate, 4),
        "semantic_complexity":   round(semantic_complexity, 4),
        "token_speed_score":     round(token_speed_score, 4),
        "tool_count_score":      round(tool_count_score, 4),
        "token_count":           token_count,
        "segment_count":         len(segments),
    }


# ---------------------------------------------------------------------------
# AGENT 2 — Local Executor (FunctionGemma + Stage 2 validation)
# ---------------------------------------------------------------------------

def agent_local_executor(messages: List[Dict], tools: List[Dict], assessment: Dict) -> Dict:
    """
    FIX: Model is now cached (loaded once, reused across calls) to eliminate
    repeated init/destroy overhead (~200-400ms saved per request).

    FIX: If heuristic confidence is >= HEURISTIC_ONLY_THRESHOLD, skip the model
    entirely and return the heuristic result immediately — no inference needed.
    """
    heuristic_calls = assessment["heuristic_calls"]
    heuristic_conf  = assessment["heuristic_conf"]

    # Short-circuit: heuristic is confident enough, skip model inference
    if heuristic_conf >= HEURISTIC_ONLY_THRESHOLD:
        return {
            "available":         True,
            "function_calls":    heuristic_calls,
            "model_confidence":  None,
            "stage2_confidence": heuristic_conf,
            "total_time_ms":     0,
            "validation":        {},
            "source":            "heuristic-only",
        }

    model = _get_cactus_model()
    if model is None:
        return {
            "available": False,
            "function_calls": heuristic_calls,
            "stage2_confidence": heuristic_conf,
            "total_time_ms": 0,
            "source": "heuristic-only",
            "validation": {},
        }

    _, cactus_complete, _ = _load_cactus()
    cactus_tools = [{"type": "function", "function": t} for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        raw = {}

    model_calls = raw.get("function_calls", [])
    model_conf  = raw.get("confidence", 0.0)
    total_time  = raw.get("total_time_ms", 0.0)

    # Merge: prefer model calls, fall back to heuristic
    calls = model_calls if model_calls else heuristic_calls

    # Output validation
    valid_tool_names = {t["name"] for t in tools}
    validation = {}
    valid_count = 0
    for call in calls:
        name = call.get("name", "")
        args = call.get("arguments", {})
        tool_match = next((t for t in tools if t["name"] == name), None)
        required_ok = _required_ok(args, tool_match) if tool_match else False
        validation[name] = {
            "valid_name": name in valid_tool_names,
            "required_fields_ok": required_ok,
        }
        if name in valid_tool_names and required_ok:
            valid_count += 1

    got_calls_score      = 1.0 if calls else 0.0
    output_valid_score   = (valid_count / len(calls)) if calls else 0.0

    stage2_confidence = (
        model_conf         * 0.25 +
        output_valid_score * 0.10 +
        got_calls_score    * 0.05 +
        heuristic_conf     * 0.20
    )

    return {
        "available":         True,
        "function_calls":    calls,
        "model_confidence":  round(model_conf, 4),
        "stage2_confidence": round(stage2_confidence, 4),
        "total_time_ms":     total_time,
        "validation":        validation,
        "source":            "functiongemma",
    }


# ---------------------------------------------------------------------------
# AGENT 3 — Cloud Executor
# ---------------------------------------------------------------------------

def agent_cloud_executor(messages: List[Dict], tools: List[Dict]) -> Dict:
    """
    Calls Anthropic Claude (preferred) or Gemini as cloud fallback.
    Returns normalised function_calls list.
    """
    start = time.time()

    # --- Try Anthropic Claude first ---
    anthropic_client = _load_anthropic()
    if anthropic_client and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            claude_tools = []
            for t in tools:
                claude_tools.append({
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": t["parameters"],
                })

            response = anthropic_client.messages.create(
                model=os.environ.get("CLAUDE_MODEL", "claude-3-5-haiku-20241022"),
                max_tokens=512,
                tools=claude_tools,
                messages=messages,
            )

            function_calls = []
            for block in response.content:
                if block.type == "tool_use":
                    function_calls.append({
                        "name": block.name,
                        "arguments": dict(block.input),
                    })

            return {
                "function_calls": function_calls,
                "total_time_ms":  (time.time() - start) * 1000,
                "provider":       "anthropic-claude",
            }
        except Exception:
            pass  # fall through to Gemini

    # --- Gemini fallback ---
    genai, types = _load_genai()
    if genai is None:
        return {"function_calls": [], "total_time_ms": 0, "provider": "none"}

    try:
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
        gemini_response = client.models.generate_content(
            model=os.environ.get("GEMINI_MODEL", "models/gemini-2.5-flash"),
            contents=contents,
            config=types.GenerateContentConfig(tools=gemini_tools),
        )

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
            "total_time_ms":  (time.time() - start) * 1000,
            "provider":       "gemini",
        }
    except Exception as e:
        return {"function_calls": [], "total_time_ms": (time.time() - start) * 1000, "provider": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# AGENT 4 — Orchestrator
# ---------------------------------------------------------------------------

# Routing thresholds (tunable)
STAGE1_SKIP_LOCAL_THRESHOLD  = 0.35   # If stage1 difficulty > this, skip local entirely
STAGE2_TRUST_LOCAL_THRESHOLD = 0.35   # If stage2 confidence >= this, trust local result


def agent_orchestrator(messages: List[Dict], tools: List[Dict]) -> Dict:
    """
    Coordinates Agents 1-3 and makes the final routing decision.
    """
    t_start = time.time()

    # ── Agent 1: assess complexity ──────────────────────────────────────────
    assessment = agent_complexity_assessor(messages, tools)

    if assessment["stage1_difficulty"] > STAGE1_SKIP_LOCAL_THRESHOLD:
        cloud = agent_cloud_executor(messages, tools)
        return {
            "function_calls":    cloud["function_calls"],
            "total_time_ms":     (time.time() - t_start) * 1000,
            "source":            "cloud-direct",
            "strategy":          "stage1-skipped-local",
            "provider":          cloud.get("provider"),
            "stage1_difficulty": assessment["stage1_difficulty"],
            "stage2_confidence": None,
            "assessment":        assessment,
        }

    # ── Agent 2: try local ──────────────────────────────────────────────────
    local = agent_local_executor(messages, tools, assessment)

    if local["stage2_confidence"] >= STAGE2_TRUST_LOCAL_THRESHOLD:
        return {
            "function_calls":    local["function_calls"],
            "total_time_ms":     (time.time() - t_start) * 1000,
            "source":            "on-device",
            "strategy":          local["source"],
            "stage1_difficulty": assessment["stage1_difficulty"],
            "stage2_confidence": local["stage2_confidence"],
            "model_confidence":  local.get("model_confidence"),
            "validation":        local.get("validation"),
            "assessment":        assessment,
        }

    # ── Agent 3: cloud fallback ─────────────────────────────────────────────
    cloud = agent_cloud_executor(messages, tools)
    return {
        "function_calls":       cloud["function_calls"],
        "total_time_ms":        (time.time() - t_start) * 1000,
        "source":               "cloud-fallback",
        "strategy":             "stage2-below-threshold",
        "provider":             cloud.get("provider"),
        "stage1_difficulty":    assessment["stage1_difficulty"],
        "stage2_confidence":    local["stage2_confidence"],
        "local_model_conf":     local.get("model_confidence"),
        "assessment":           assessment,
    }


# ---------------------------------------------------------------------------
# Public API (backward-compatible wrappers)
# ---------------------------------------------------------------------------

def generate_cactus(messages: List[Dict], tools: List[Dict]) -> Dict:
    """Direct on-device execution (no routing)."""
    assessment = agent_complexity_assessor(messages, tools)
    return agent_local_executor(messages, tools, assessment)


def generate_cloud(messages: List[Dict], tools: List[Dict]) -> Dict:
    """Direct cloud execution (no routing)."""
    return agent_cloud_executor(messages, tools)


def generate_hybrid(messages: List[Dict], tools: List[Dict], **kwargs) -> Dict:
    """Full hybrid routing via Orchestrator."""
    return agent_orchestrator(messages, tools)


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_result(label: str, result: Dict) -> None:
    print(f"\n=== {label} ===\n")
    for key in ("source", "strategy", "provider"):
        if key in result:
            print(f"{key.capitalize()}: {result[key]}")
    if "stage1_difficulty" in result and result["stage1_difficulty"] is not None:
        print(f"Stage1 difficulty:  {result['stage1_difficulty']:.4f}")
    if "stage2_confidence" in result and result["stage2_confidence"] is not None:
        print(f"Stage2 confidence:  {result['stage2_confidence']:.4f}")
    if "model_confidence" in result and result["model_confidence"] is not None:
        print(f"Model confidence:   {result['model_confidence']:.4f}")
    print(f"Total time: {result.get('total_time_ms', 0):.2f}ms")
    for call in result.get("function_calls", []):
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"],
        },
    }]

    messages = [{"role": "user", "content": "What is the weather in San Francisco?"}]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Cloud (Anthropic / Gemini)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (Orchestrated)", hybrid)
