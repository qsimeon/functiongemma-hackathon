# Source: https://github.com/desmondzee/functiongemma-hackathon

import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json
import os
import re
import time
import tempfile
from pathlib import Path

from cactus import (
    cactus_init,
    cactus_complete,
    cactus_destroy,
    cactus_get_last_error,
    cactus_embed,
    cactus_rag_query,
    cactus_index_init,
    cactus_index_add,
    cactus_index_query,
    cactus_index_destroy,
)
from google import genai
from google.genai import types


# --- Tool narrowing: vector + RAG + keyword merge so cactus_complete sees fewer tools; guaranteed to include required tools. ---

TOP_K_BY_DIFFICULTY = {"easy": 1, "medium": 3, "hard": 5}

# Keywords in task name or message that imply a tool (same as testdb).
TOOL_KEYWORDS = {
    "get_weather": ["weather"],
    "set_alarm": ["alarm"],
    "send_message": ["message", "send", "text"],
    "create_reminder": ["remind", "reminder"],
    "search_contacts": ["contact", "contacts", "search", "find", "look up", "lookup"],
    "play_music": ["play", "music", "song"],
    "set_timer": ["timer"],
}

_narrowing_state = None


def _tool_to_text(tool):
    """Serialize a tool (name, description, properties) into a single string for embedding."""
    parts = [f"name: {tool['name']}", f"description: {tool['description']}"]
    props = tool.get("parameters", {}).get("properties", {})
    if props:
        parts.append("properties: " + json.dumps({k: v.get("description", v.get("type", "")) for k, v in props.items()}))
    return " ".join(parts)


def _keyword_matched_tools(name, query, valid_names):
    """Return set of tool names that have a keyword match in name or query (case-insensitive)."""
    text = f"{name} {query}".lower()
    out = set()
    for tool_name, keywords in TOOL_KEYWORDS.items():
        if tool_name not in valid_names:
            continue
        for kw in keywords:
            if kw.lower() in text:
                out.add(tool_name)
                break
    return out


def _build_tool_narrowing_state(all_tools):
    """Build and cache model (with corpus_dir), vector index, and tool metadata. Idempotent for same all_tools."""
    global _narrowing_state
    valid_names = {t["name"] for t in all_tools}
    tool_texts = [_tool_to_text(t) for t in all_tools]
    corpus_dir = Path(tempfile.mkdtemp(prefix="main_corpus_"))
    for i, tool in enumerate(all_tools):
        (corpus_dir / f"{tool['name']}.txt").write_text(tool_texts[i], encoding="utf-8")
    model = cactus_init(functiongemma_path, corpus_dir=str(corpus_dir), cache_index=False)
    if model is None:
        return
    tool_embeddings = [cactus_embed(model, text, normalize=True) for text in tool_texts]
    embedding_dim = len(tool_embeddings[0])
    index_dir = Path(tempfile.mkdtemp(prefix="main_index_"))
    index = cactus_index_init(str(index_dir), embedding_dim)
    if index is None:
        cactus_destroy(model)
        return
    ids = list(range(len(all_tools)))
    rc = cactus_index_add(index, ids, tool_texts, tool_embeddings)
    if rc != 0:
        cactus_index_destroy(index)
        cactus_destroy(model)
        return
    query_buffer_k = max(max(TOP_K_BY_DIFFICULTY.values()), len(all_tools))
    _narrowing_state = {
        "model": model,
        "index": index,
        "corpus_dir": corpus_dir,
        "index_dir": index_dir,
        "tool_texts": tool_texts,
        "all_tools": all_tools,
        "query_buffer_k": query_buffer_k,
        "valid_names": valid_names,
    }


def narrow_tools(messages, tools, all_tools, task_name=None, difficulty=None):
    """
    Return a subset of `tools` whose names are in the vector+RAG+keyword merged set.
    Intended to include all required tools so the model can be called with less context.
    task_name and difficulty are optional (defaults: empty string and top_k=5).
    """
    global _narrowing_state
    if _narrowing_state is None or _narrowing_state["all_tools"] is not all_tools:
        _build_tool_narrowing_state(all_tools)
    if _narrowing_state is None:
        return tools
    state = _narrowing_state
    model, index, tool_texts = state["model"], state["index"], state["tool_texts"]
    all_tools_list = state["all_tools"]
    valid_names = state["valid_names"]
    query_buffer_k = state["query_buffer_k"]
    query_parts = [m["content"] for m in messages if m.get("role") == "user"]
    query_str = " ".join(query_parts)
    search_text = f"{task_name or ''} {query_str}".strip()
    top_k = TOP_K_BY_DIFFICULTY.get(difficulty, 5)
    query_emb = cactus_embed(model, search_text, normalize=True)
    hits = cactus_index_query(index, query_emb, top_k=query_buffer_k)
    vector_names = [all_tools_list[h["id"]]["name"] for h in hits[:top_k]]
    raw_rag = cactus_rag_query(model, search_text, top_k=top_k)
    chunks = raw_rag.get("chunks", []) if isinstance(raw_rag, dict) else (raw_rag if isinstance(raw_rag, list) else [])
    rag_names = []
    for c in chunks:
        if not isinstance(c, dict):
            continue
        name = None
        src = c.get("source", "")
        if src:
            base = Path(src).name if isinstance(src, str) else str(src)
            if base.endswith(".txt"):
                name = base[:-4]
                if name not in valid_names:
                    name = None
        if not name:
            text = c.get("content") or c.get("text") or ""
            m = re.search(r"name:\s*(\w+)", (text or "").strip())
            if m and m.group(1) in valid_names:
                name = m.group(1)
        if name:
            rag_names.append(name)
    merged_names = set(vector_names) | set(rag_names) | _keyword_matched_tools(task_name or "", query_str, valid_names)
    if not merged_names:
        return tools
    return [t for t in tools if t["name"] in merged_names]


def generate_cactus(messages, tools, all_tools=None, task_name=None, difficulty=None):
    """
    Run function calling on-device via FunctionGemma + Cactus.
    When all_tools is provided, tool narrowing is used (vector + RAG + keyword) and the same model
    is reused (no destroy). When not provided, behavior is unchanged (init/complete/destroy per call).
    """
    use_narrowing = all_tools is not None
    if use_narrowing and _narrowing_state is None:
        _build_tool_narrowing_state(all_tools)
    if use_narrowing and _narrowing_state is not None:
        tools = narrow_tools(messages, tools, all_tools, task_name=task_name, difficulty=difficulty)
        model = _narrowing_state["model"]
    else:
        model = cactus_init(functiongemma_path)

    cactus_tools = [{"type": "function", "function": t} for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    if not use_narrowing or _narrowing_state is None:
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


# TESTING: Set to True to always use local (on-device) only — every call generates locally, cloud is never used.
FORCE_LOCAL_ONLY = True

def generate_hybrid(messages, tools, confidence_threshold=0.99, all_tools=None, task_name=None, difficulty=None):
    """Baseline hybrid inference strategy; fall back to cloud if Cactus Confidence is below threshold.
    When all_tools, task_name, difficulty are provided, tool narrowing is used for on-device and cloud (smaller context).
    """
    local = generate_cactus(messages, tools, all_tools=all_tools, task_name=task_name, difficulty=difficulty)
    tools_for_cloud = tools
    if all_tools is not None and _narrowing_state is not None:
        tools_for_cloud = narrow_tools(messages, tools, all_tools, task_name=task_name, difficulty=difficulty)

    # TESTING: Always use local result; never call cloud.
    if FORCE_LOCAL_ONLY:
        local["source"] = "on-device"
        return local

    if local["confidence"] >= confidence_threshold:
        local["source"] = "on-device"
        return local

    cloud = generate_cloud(messages, tools_for_cloud)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local["confidence"]
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
    # For production, pass full tool list and task_name/difficulty when available to use tool narrowing.
    all_tools = tools
    task_name = "example"
    difficulty = "easy"

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools, all_tools=all_tools, task_name=task_name, difficulty=difficulty)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)
