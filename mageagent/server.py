#!/usr/bin/env python3
"""
MageAgent Orchestrator - Multi-Model LLM Server for MLX
Provides OpenAI-compatible API with intelligent model routing and validation patterns

Patterns:
- mageagent:auto - Intelligent task classification and routing
- mageagent:execute - ReAct loop with REAL tool execution (reads files, runs commands)
- mageagent:self_consistent - Multiple reasoning paths, +17% accuracy improvement
- mageagent:critic - Self-critique loop for edge case detection, +10% improvement
- mageagent:validated - Generate + validate with correction loop
- mageagent:compete - Competing models with judge
- mageagent:hybrid - Qwen-72B reasoning + Hermes-3 tool extraction
- mageagent:tools - Tool-calling specialist (Hermes-3 Q8)
- mageagent:primary - Direct access to 72B model
- mageagent:validator - Direct access to 7B validator
- mageagent:fast - Quick responses with 7B model

New in v2.3:
- Project-Specific RAG: Index codebase, inject matching patterns for style consistency
- Self-Consistency: Generate N responses, select most consistent answer
- CRITIC: Iterative self-critique and revision for edge case detection
"""

# Version - keep in sync with package.json
VERSION = "2.3.0"

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

# Add the mageagent directory to the path for imports
SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mlx.core as mx
from mlx_lm import load, generate

# Timeout configuration per model (seconds)
# Calculated as: (max_tokens / tokens_per_second) + buffer for model loading
TIMEOUT_CONFIG = {
    "tools": 120,      # Hermes-3 8B: ~50 tok/s, 2048 tokens = 41s + buffer
    "primary": 600,    # Qwen 72B: ~8 tok/s, 2048 tokens = 256s + buffer
    "validator": 60,   # Qwen 7B: ~105 tok/s, 2048 tokens = 20s + buffer
    "competitor": 180, # Qwen 32B: ~25 tok/s, 2048 tokens = 82s + buffer
    "glm": 90,         # GLM-4.7 Flash: ~80 tok/s (MoE), 2048 tokens = 26s + buffer
}

# Model loading locks for thread safety (initialized after MODELS dict)
model_locks: Dict[str, asyncio.Lock] = {}


class GenerationTimeoutError(Exception):
    """Raised when model generation exceeds the configured timeout"""
    pass


# Model paths - using existing downloaded models
MLX_MODELS_DIR = Path.home() / ".cache" / "mlx-models"

MODELS = {
    "tools": {
        "path": str(MLX_MODELS_DIR / "Hermes-3-Llama-3.1-8B-8bit"),
        "role": "tool calling specialist - file operations, function execution, structured output",
        "quant": "Q8_0",
        "memory_gb": 9,
        "supports_tools": True,  # Q8 reliably supports tool calling
        "tok_per_sec": 50
    },
    "primary": {
        "path": str(MLX_MODELS_DIR / "Qwen2.5-72B-Instruct-8bit"),
        "role": "primary generator - planning, analysis, complex reasoning",
        "quant": "Q8_0",
        "memory_gb": 77,
        "supports_tools": True,  # Q8 supports tool calling
        "tok_per_sec": 8
    },
    "validator": {
        "path": str(MLX_MODELS_DIR / "Qwen2.5-Coder-7B-Instruct-4bit"),
        "role": "fast validation, cross-checking, judging",
        "quant": "Q4_K_M",
        "memory_gb": 5,
        "supports_tools": False,
        "tok_per_sec": 105
    },
    "competitor": {
        "path": str(MLX_MODELS_DIR / "Qwen2.5-Coder-32B-Instruct-4bit"),
        "role": "competing solution generator, code specialist",
        "quant": "Q4_K_M",
        "memory_gb": 18,
        "supports_tools": False,
        "tok_per_sec": 25
    },
    "glm": {
        "path": str(MLX_MODELS_DIR / "GLM-4.7-Flash-4bit"),
        "role": "fast reasoning - 30B MoE with 3B active params, excellent for quick tasks",
        "quant": "4bit",
        "memory_gb": 17,
        "supports_tools": False,
        "tok_per_sec": 80  # MoE is faster due to sparse activation
    }
}

# Draft model for speculative decoding (1.5-3x speedup)
DRAFT_MODEL = {
    "path": str(MLX_MODELS_DIR / "Qwen2.5-0.5B-Instruct-4bit"),
    "role": "draft model for speculative decoding",
    "quant": "Q4",
    "memory_gb": 0.5,
    "tok_per_sec": 200  # Very fast small model
}

# Speculative decoding configuration
# NOTE: DISABLED due to Metal GPU timeout errors on all models
# Even the 32B model with draft model causes kIOGPUCommandBufferCallbackErrorTimeout
# This appears to be a known issue with mlx-lm speculative decoding on some hardware
# See: https://github.com/ml-explore/mlx-examples/issues/1281
# The M4 Max should theoretically support this (~49% speedup reported), but
# the current mlx-lm implementation has issues with command buffer timing
SPECULATIVE_CONFIG = {
    "enabled": False,  # DISABLED - causes GPU timeout crashes
    "num_draft_tokens": 4,  # Number of tokens to draft at once
    "applicable_models": [],  # No models currently support this without crashing
}

# Lazy-loaded models cache
loaded_models: Dict[str, Any] = {}
model_tokenizers: Dict[str, Any] = {}
draft_model_cache: Dict[str, Any] = {"model": None, "tokenizer": None}  # Single draft model

# Stats tracking for throughput monitoring
inference_stats: Dict[str, Any] = {
    "total_requests": 0,
    "total_tokens_generated": 0,
    "total_prompt_tokens": 0,
    "last_inference": None,  # timestamp
    "last_model": None,
    "last_pattern": None,
    "last_tokens_per_sec": 0.0,
    "last_tokens_generated": 0,
    "last_duration_sec": 0.0,
    "requests_by_model": {},
    "tokens_by_model": {},
    "requests_by_pattern": {},
    "tokens_by_pattern": {},
    "prompt_tokens_by_model": {},
    "session_start": time.time(),
}


def update_stats(
    model: str,
    pattern: str,
    prompt_tokens: int,
    completion_tokens: int,
    duration_sec: float
):
    """Update inference statistics after each request"""
    inference_stats["total_requests"] += 1
    inference_stats["total_tokens_generated"] += completion_tokens
    inference_stats["total_prompt_tokens"] += prompt_tokens
    inference_stats["last_inference"] = time.time()
    inference_stats["last_model"] = model
    inference_stats["last_pattern"] = pattern
    inference_stats["last_tokens_generated"] = completion_tokens
    inference_stats["last_duration_sec"] = duration_sec

    if duration_sec > 0:
        inference_stats["last_tokens_per_sec"] = completion_tokens / duration_sec
    else:
        inference_stats["last_tokens_per_sec"] = 0

    # Track by model
    if model not in inference_stats["requests_by_model"]:
        inference_stats["requests_by_model"][model] = 0
        inference_stats["tokens_by_model"][model] = 0
        inference_stats["prompt_tokens_by_model"][model] = 0
    inference_stats["requests_by_model"][model] += 1
    inference_stats["tokens_by_model"][model] += completion_tokens
    inference_stats["prompt_tokens_by_model"][model] += prompt_tokens

    # Track by pattern
    if pattern not in inference_stats["requests_by_pattern"]:
        inference_stats["requests_by_pattern"][pattern] = 0
        inference_stats["tokens_by_pattern"][pattern] = 0
    inference_stats["requests_by_pattern"][pattern] += 1
    inference_stats["tokens_by_pattern"][pattern] += completion_tokens

# Request/Response models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Usage

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "mageagent"

class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


def get_model(model_type: str) -> tuple:
    """Lazy-load and cache models (synchronous - use load_model_async when possible)"""
    if model_type not in MODELS:
        raise ValueError(f"Unknown model type: {model_type}")

    if model_type not in loaded_models:
        model_config = MODELS[model_type]
        model_path = model_config["path"]

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        print(f"Loading {model_type} model from {model_path}...")
        start = time.time()
        model, tokenizer = load(model_path)
        print(f"Loaded {model_type} in {time.time() - start:.1f}s")

        loaded_models[model_type] = model
        model_tokenizers[model_type] = tokenizer

    return loaded_models[model_type], model_tokenizers[model_type]


async def load_draft_model_async() -> tuple:
    """
    Load the small draft model for speculative decoding.
    Only needs to be loaded once since all models share the same draft.
    """
    global draft_model_cache

    if draft_model_cache["model"] is not None:
        return draft_model_cache["model"], draft_model_cache["tokenizer"]

    draft_path = DRAFT_MODEL["path"]
    if not Path(draft_path).exists():
        print(f"Draft model not found at {draft_path}, speculative decoding disabled")
        return None, None

    print(f"Loading draft model for speculative decoding from {draft_path}...")
    start = time.time()

    loop = asyncio.get_event_loop()
    draft_model, draft_tokenizer = await loop.run_in_executor(
        None,
        lambda: load(draft_path)
    )

    draft_model_cache["model"] = draft_model
    draft_model_cache["tokenizer"] = draft_tokenizer
    print(f"Draft model loaded in {time.time() - start:.1f}s (speculative decoding enabled)")

    return draft_model, draft_tokenizer


async def load_model_async(model_type: str) -> tuple:
    """
    Thread-safe async model loading with lock to prevent concurrent loads.
    This prevents Metal crashes from simultaneous model loading.
    """
    global model_locks

    if model_type not in MODELS:
        raise ValueError(f"Unknown model type: {model_type}")

    # Fast path: model already loaded
    if model_type in loaded_models:
        return loaded_models[model_type], model_tokenizers[model_type]

    # Initialize lock for this model type if not exists
    if model_type not in model_locks:
        model_locks[model_type] = asyncio.Lock()

    # Acquire lock to prevent concurrent loads
    async with model_locks[model_type]:
        # Double-check after acquiring lock (another request may have loaded it)
        if model_type in loaded_models:
            return loaded_models[model_type], model_tokenizers[model_type]

        model_config = MODELS[model_type]
        model_path = model_config["path"]

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        print(f"Loading {model_type} model from {model_path}...")
        start = time.time()

        # Load in executor to not block event loop
        loop = asyncio.get_event_loop()
        model, tokenizer = await loop.run_in_executor(
            None,
            lambda: load(model_path)
        )

        loaded_models[model_type] = model
        model_tokenizers[model_type] = tokenizer
        print(f"✓ Loaded {model_type} in {time.time() - start:.1f}s")

        return model, tokenizer


def format_chat_prompt(messages: List[ChatMessage], tokenizer) -> str:
    """Format messages into a chat prompt using the tokenizer's chat template"""
    formatted_messages = [{"role": m.role, "content": m.content} for m in messages]

    # Use the tokenizer's chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        return tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )

    # Fallback to simple formatting
    prompt = ""
    for msg in formatted_messages:
        if msg["role"] == "system":
            prompt += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
        elif msg["role"] == "user":
            prompt += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
        elif msg["role"] == "assistant":
            prompt += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


async def _generate_internal(model_type: str, messages: List[ChatMessage], max_tokens: int, temperature: float) -> str:
    """Internal generation function that does the actual work, with optional speculative decoding."""
    model, tokenizer = await load_model_async(model_type)
    prompt = format_chat_prompt(messages, tokenizer)

    # Check if we should use speculative decoding for this model
    use_speculative = (
        SPECULATIVE_CONFIG["enabled"] and
        model_type in SPECULATIVE_CONFIG["applicable_models"]
    )

    draft_model = None
    if use_speculative:
        draft_model, _ = await load_draft_model_async()
        if draft_model is not None:
            print(f"Using speculative decoding for {model_type} (draft: Qwen-0.5B)")

    # Track start time for throughput calculation
    gen_start = time.time()

    # Run generation in a thread pool to not block event loop
    loop = asyncio.get_event_loop()

    if draft_model is not None:
        # Speculative decoding path - 1.5x-3x speedup for large models
        response = await loop.run_in_executor(
            None,
            lambda: generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False,
                draft_model=draft_model
            )
        )
    else:
        # Standard generation path
        response = await loop.run_in_executor(
            None,
            lambda: generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False
            )
        )

    # Calculate and update stats
    gen_duration = time.time() - gen_start
    # Estimate tokens generated (rough estimate based on response length)
    # More accurate would be to use tokenizer.encode, but this is faster
    tokens_generated = len(response.split()) + len(response) // 4  # rough estimate
    tokens_per_sec = tokens_generated / gen_duration if gen_duration > 0 else 0

    # Update global stats
    inference_stats["total_requests"] += 1
    inference_stats["total_tokens_generated"] += tokens_generated
    inference_stats["last_inference"] = time.time()
    inference_stats["last_model"] = model_type + (" (speculative)" if draft_model else "")
    inference_stats["last_tokens_per_sec"] = round(tokens_per_sec, 1)
    inference_stats["last_tokens_generated"] = tokens_generated
    inference_stats["last_duration_sec"] = round(gen_duration, 2)

    # Track per-model stats
    if model_type not in inference_stats["requests_by_model"]:
        inference_stats["requests_by_model"][model_type] = 0
        inference_stats["tokens_by_model"][model_type] = 0
    inference_stats["requests_by_model"][model_type] += 1
    inference_stats["tokens_by_model"][model_type] += tokens_generated

    return response


async def generate_with_model(
    model_type: str,
    messages: List[ChatMessage],
    max_tokens: int = 2048,
    temperature: float = 0.7
) -> str:
    """
    Generate response using specified model with proper timeout handling.

    Timeouts are configured per-model based on their generation speed:
    - validator (7B): 60s  (~105 tok/s)
    - tools (8B): 120s     (~50 tok/s)
    - competitor (32B): 180s (~25 tok/s)
    - primary (72B): 600s  (~8 tok/s)

    Uses asyncio.wait_for for Python 3.9+ compatibility.
    """
    timeout = TIMEOUT_CONFIG.get(model_type, 300)  # Default 5 min if unknown

    try:
        # Use asyncio.wait_for for Python 3.9+ compatibility (asyncio.timeout is 3.11+)
        response = await asyncio.wait_for(
            _generate_internal(model_type, messages, max_tokens, temperature),
            timeout=timeout
        )
        return response

    except asyncio.TimeoutError:
        raise GenerationTimeoutError(
            f"Generation timeout after {timeout}s for model '{model_type}'. "
            f"The {MODELS[model_type]['quant']} model at ~{MODELS[model_type]['tok_per_sec']} tok/s "
            f"couldn't complete {max_tokens} tokens in time. "
            f"Try reducing max_tokens or using a faster model like 'validator'."
        )


def needs_tool_extraction(prompt: str) -> bool:
    """Check if the prompt requires tool extraction - be very liberal"""
    tool_patterns = [
        r'\bread\b.*\bfile\b', r'\bwrite\b.*\bfile\b', r'\blist\b',
        r'\bexecute\b', r'\brun\b', r'\bcreate\b.*\bfile\b', r'\bdelete\b',
        r'\bsearch\b', r'\bfind\b', r'\bedit\b', r'\bmodify\b',
        r'\btool\b', r'\bfunction\b.*\bcall\b', r'\bapi\b.*\bcall\b',
        r'\bglob\b', r'\bgrep\b', r'\bbash\b', r'\bshell\b',
        # Filesystem-related patterns
        r'\bfile[s]?\b', r'\bdirectory\b', r'\bfolder\b', r'\bpath\b',
        r'/users/', r'~/', r'\.\w+$',  # Path patterns
        # Count/listing patterns
        r'\bhow many\b', r'\bcount\b', r'\blist\b.*\bfiles?\b',
        # Web patterns
        r'\bweb\b.*\bsearch\b', r'\bsearch\b.*\bweb\b', r'\bonline\b',
        r'\binternet\b', r'\burl\b', r'\bhttp', r'\bfetch\b',
        # Command patterns
        r'\bcommand\b', r'\bterminal\b', r'\bcli\b',
        # Additional patterns for common filesystem tasks (added for better tool triggering)
        r'\bshow\b.*\bme\b', r'\bwhat\b.*\bin\b', r'\bwhat\b.*\bcontains\b',
        r'\bcontents?\b', r'\bexport\b', r'\bimport\b', r'\bpackage\.json\b',
        r'\.py\b', r'\.ts\b', r'\.js\b', r'\.md\b', r'\.yaml\b', r'\.json\b',
        r'\bcode\b.*\bin\b', r'\bproject\b', r'\bcodebase\b', r'\brepo\b',
        r'\bgit\b', r'\bls\b', r'\bcat\b', r'\bhead\b', r'\btail\b',
        r'\bsummarize\b.*\bfile\b', r'\banalyze\b.*\bfile\b',
        r'\bk8s\b', r'\bkubectl\b', r'\bdocker\b', r'\bnpm\b', r'\bpip\b'
    ]
    return any(re.search(p, prompt.lower()) for p in tool_patterns)


async def extract_tool_calls(user_content: str, response: str) -> list:
    """
    Use Hermes-3 Q8 to extract tool calls from any response.
    This is the ONLY model that should handle tool extraction.
    """
    print("Hermes-3 Q8 extracting tool calls...")
    tool_messages = [
        ChatMessage(role="system", content="""You are an AGGRESSIVE tool-calling assistant. Your job is to identify what tools are needed to complete a task.

ALWAYS prefer using tools over generating text explanations. If the task involves:
- Reading/viewing files → Use Read tool
- Running commands → Use Bash tool
- Finding files → Use Glob or Bash with find/ls
- Searching content → Use Grep
- Counting files → Use Bash with find | wc -l
- Web search → Use WebSearch
- Fetching URLs → Use WebFetch

Output tool calls as JSON array:
[{"tool": "tool_name", "arguments": {"arg1": "value1"}}]

Available tools:
- Read: {"file_path": "path"} - Read file contents (use absolute paths)
- Write: {"file_path": "path", "content": "content"} - Write to file
- Edit: {"file_path": "path", "old_string": "text", "new_string": "text"} - Edit file
- Bash: {"command": "shell_command"} - Execute ANY shell command (ls, find, cat, etc.)
- Glob: {"pattern": "**/*.py", "path": "dir"} - Find files by pattern
- Grep: {"pattern": "regex", "path": "dir"} - Search file contents
- WebSearch: {"query": "search terms"} - Search the web
- WebFetch: {"url": "https://...", "prompt": "what to extract"} - Fetch and process URL

IMPORTANT: If the task requires getting ACTUAL data from the filesystem or web, you MUST output tools.
Only output [] if the task is purely conversational with no data needs.

Example: "How many Python files in /foo?" → [{"tool": "Bash", "arguments": {"command": "find /foo -name '*.py' | wc -l"}}]
Example: "List files in /bar" → [{"tool": "Bash", "arguments": {"command": "ls -la /bar"}}]"""),
        ChatMessage(role="user", content=f"""Task: {user_content}

The model's initial response was:
{response[:1000]}

CRITICAL: The response above is just EXPLAINING how to do the task. We need to ACTUALLY DO IT by calling tools.

What tools should be executed to complete this task? Output JSON array. If listing files, use Bash with ls. If reading a file, use Read. If searching, use Grep.

JSON output only (no explanation):""")
    ]

    tool_response = await generate_with_model("tools", tool_messages, 512, 0.1)

    # Parse tool calls - try multiple approaches
    print(f"  Tool extraction raw response: {tool_response[:300]}", flush=True)

    # Approach 1: Find JSON array with proper bracket matching
    try:
        # Look for the first [ and find matching ]
        start = tool_response.find('[')
        if start != -1:
            depth = 0
            end = start
            for i, c in enumerate(tool_response[start:], start):
                if c == '[':
                    depth += 1
                elif c == ']':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            json_str = tool_response[start:end]
            result = json.loads(json_str)
            if result and isinstance(result, list):
                print(f"  ✓ Extracted {len(result)} tool(s): {[t.get('tool') for t in result]}", flush=True)
                return result
    except Exception as e:
        print(f"  Tool extraction error: {e}", flush=True)

    # Approach 2: Try to extract tool info even if JSON is malformed
    if '"tool"' in tool_response and '"Bash"' in tool_response:
        print("  Attempting to extract Bash tool from malformed response...", flush=True)
        try:
            # Extract command from response
            cmd_match = re.search(r'"command"\s*:\s*"([^"]+)"', tool_response)
            if cmd_match:
                return [{"tool": "Bash", "arguments": {"command": cmd_match.group(1)}}]
        except:
            pass

    print("  No tools extracted", flush=True)
    return []


async def execute_extracted_tools(
    tool_calls: list,
    user_content: str,
    initial_response: str,
    max_iterations: int = 3
) -> Dict[str, Any]:
    """
    Execute extracted tool calls and feed results back for a final response.
    This is the shared tool execution logic used by ALL patterns.
    """
    if not tool_calls:
        return {
            "final_response": initial_response,
            "observations": [],
            "tools_executed": 0
        }

    from tool_executor import ToolExecutor
    executor = ToolExecutor()

    all_observations = []

    # Execute all tool calls
    print(f"Executing {len(tool_calls)} extracted tool(s)...")
    for i, tc in enumerate(tool_calls):
        tool_name = tc.get("tool", "unknown")
        print(f"  [{i+1}/{len(tool_calls)}] {tool_name}")

        result = executor.execute(tc)
        all_observations.append({
            "tool": tool_name,
            "arguments": tc.get("arguments", {}),
            "result": result
        })

        if "error" in result:
            print(f"    ❌ {result['error']}")
        else:
            print(f"    ✓ Success")

    # Generate final response with tool results
    print("Generating final response with tool results...")
    obs_text = "\n\n".join([
        f"**{o['tool']}** ({json.dumps(o['arguments'])}): ```{json.dumps(o['result'], indent=2)[:500]}```"
        for o in all_observations
    ])

    final_messages = [
        ChatMessage(role="system", content="You are a helpful assistant. Use the tool execution results provided to give an accurate, factual answer."),
        ChatMessage(role="user", content=f"""Original task: {user_content}

Tool execution results:
{obs_text}

Based on these ACTUAL results, provide your final answer:""")
    ]

    final_response = await generate_with_model("tools", final_messages, 2048, 0.3)

    return {
        "final_response": final_response,
        "observations": all_observations,
        "tools_executed": len(all_observations)
    }


def classify_task(prompt: str) -> str:
    """Classify task type for model routing"""
    coding_patterns = [
        r'\bwrite\b.*\bcode\b', r'\bimplement\b', r'\bfunction\b',
        r'\bclass\b', r'\brefactor\b', r'\bfix\b.*\bbug\b',
        r'```', r'\btypescript\b', r'\bpython\b', r'\brust\b',
        r'\bjavascript\b', r'\bjava\b', r'\bgo\b', r'\bc\+\+\b'
    ]
    reasoning_patterns = [
        r'\bexplain\b', r'\banalyze\b', r'\bplan\b', r'\bdesign\b',
        r'\barchitecture\b', r'\bwhy\b', r'\bhow does\b', r'\bcompare\b',
        r'\bwhat is\b', r'\bdefine\b', r'\bdescribe\b'
    ]

    prompt_lower = prompt.lower()

    coding_score = sum(1 for p in coding_patterns if re.search(p, prompt_lower))
    reasoning_score = sum(1 for p in reasoning_patterns if re.search(p, prompt_lower))

    if coding_score > reasoning_score:
        return "coding"
    elif reasoning_score > 0:
        return "reasoning"
    else:
        return "simple"


async def generate_with_validation(
    messages: List[ChatMessage],
    max_tokens: int = 2048,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """Generate with primary model, then validate with validator model"""

    # Step 1: Generate with primary model
    print("Step 1: Generating with primary model (72B)...")
    primary_response = await generate_with_model(
        "primary", messages, max_tokens, temperature
    )

    # Step 2: Validate with fast model
    print("Step 2: Validating with validator model (7B)...")
    user_content = messages[-1].content if messages else ""

    validation_messages = [
        ChatMessage(role="system", content="""You are a code reviewer. Review the response for issues:
1. Syntax errors
2. Logic bugs
3. Missing error handling
4. Security vulnerabilities
5. Performance problems

Output ONLY "PASS" if no issues found, or "FAIL: <brief list of issues>" if problems exist."""),
        ChatMessage(role="user", content=f"""Original question:
{user_content}

Response to review:
{primary_response}

Your review (PASS or FAIL with issues):""")
    ]

    validation = await generate_with_model(
        "validator", validation_messages, 512, 0.3
    )

    # Step 3: If issues found, regenerate with feedback
    needs_revision = "FAIL" in validation.upper() or "PASS" not in validation.upper()

    if needs_revision:
        print("Step 3: Issues found, regenerating with feedback...")
        revision_messages = messages.copy()
        revision_messages.append(ChatMessage(
            role="assistant",
            content=primary_response
        ))
        revision_messages.append(ChatMessage(
            role="user",
            content=f"""The previous response had these issues:
{validation}

Please provide a corrected response addressing these issues."""
        ))

        primary_response = await generate_with_model(
            "primary", revision_messages, max_tokens, temperature
        )

    # Step 4: Extract AND EXECUTE tool calls if needed
    tool_calls = None
    tool_result = {"observations": [], "tools_executed": 0}
    final_response = primary_response

    if needs_tool_extraction(user_content):
        print("Step 4: Hermes-3 Q8 extracting tool calls...")
        tool_calls = await extract_tool_calls(user_content, primary_response)

        if tool_calls:
            print("Step 5: EXECUTING extracted tools...")
            tool_result = await execute_extracted_tools(
                tool_calls, user_content, primary_response
            )
            final_response = tool_result["final_response"]

    return {
        "response": final_response,
        "validation": validation,
        "revised": needs_revision,
        "tool_calls": tool_calls,
        "observations": tool_result["observations"],
        "tools_executed": tool_result["tools_executed"],
        "model_flow": f"72B-Q8 -> 7B-validator -> hermes-3-Q8 -> exec ({tool_result['tools_executed']} tools)" if tool_calls else "72B-Q8 -> 7B-validator"
    }


async def generate_competing(
    messages: List[ChatMessage],
    max_tokens: int = 2048,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """Generate with two models sequentially, judge picks best"""

    # Step 1: Generate with both models SEQUENTIALLY (parallel crashes Metal on large models)
    print("Step 1a: Generating with primary (72B)...")
    primary_response = await generate_with_model("primary", messages, max_tokens, temperature)

    print("Step 1b: Generating with competitor (32B)...")
    competitor_response = await generate_with_model("competitor", messages, max_tokens, temperature)

    # Step 2: Judge picks best
    print("Step 2: Judging with validator (7B)...")
    user_content = messages[-1].content if messages else ""

    judge_messages = [
        ChatMessage(role="system", content="""You are a code quality judge. Compare two solutions and pick the better one.
Consider: correctness, efficiency, readability, error handling.
Output ONLY "A" or "B" followed by a brief one-sentence explanation."""),
        ChatMessage(role="user", content=f"""Original question:
{user_content}

Solution A (72B reasoning model):
{primary_response}

Solution B (32B coding model):
{competitor_response}

Which is better? (A or B with brief reason):""")
    ]

    judgment = await generate_with_model("validator", judge_messages, 256, 0.3)

    # Parse judgment
    winner = "A" if judgment.strip().startswith("A") else "B"
    best_response = primary_response if winner == "A" else competitor_response

    # Step 3: Extract AND EXECUTE tool calls if needed
    tool_calls = None
    tool_result = {"observations": [], "tools_executed": 0}
    final_response = best_response

    if needs_tool_extraction(user_content):
        print("Step 3: Hermes-3 Q8 extracting tool calls...")
        tool_calls = await extract_tool_calls(user_content, best_response)

        if tool_calls:
            print("Step 4: EXECUTING extracted tools...")
            tool_result = await execute_extracted_tools(
                tool_calls, user_content, best_response
            )
            final_response = tool_result["final_response"]

    return {
        "response": final_response,
        "winner": winner,
        "judgment": judgment,
        "solution_a": primary_response,
        "solution_b": competitor_response,
        "tool_calls": tool_calls,
        "observations": tool_result["observations"],
        "tools_executed": tool_result["tools_executed"],
        "model_flow": f"72B + 32B -> 7B-judge -> exec ({tool_result['tools_executed']} tools, winner: {winner})" if tool_calls else f"72B + 32B -> 7B-judge (winner: {winner})"
    }


async def generate_hybrid(
    messages: List[ChatMessage],
    max_tokens: int = 2048,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Hybrid pattern: Qwen-72B Q8 for reasoning + Hermes-3 Q8 for tool execution
    ALWAYS extracts tools via Hermes-3 for best capability, then EXECUTES them.
    """

    user_content = messages[-1].content if messages else ""

    # Step 1: Qwen-72B generates the main response with reasoning
    print("Step 1: Qwen-72B Q8 analyzing and generating response...")
    primary_response = await generate_with_model(
        "primary", messages, max_tokens, temperature
    )

    # Step 2: Extract AND EXECUTE tool calls via Hermes-3
    tool_calls = None
    tool_result = {"observations": [], "tools_executed": 0}
    final_response = primary_response

    if needs_tool_extraction(user_content):
        print("Step 2: Hermes-3 Q8 extracting tool calls...")
        tool_calls = await extract_tool_calls(user_content, primary_response)

        if tool_calls:
            print("Step 3: EXECUTING extracted tools...")
            tool_result = await execute_extracted_tools(
                tool_calls, user_content, primary_response
            )
            final_response = tool_result["final_response"]

    return {
        "response": final_response,
        "tool_calls": tool_calls,
        "observations": tool_result["observations"],
        "tools_executed": tool_result["tools_executed"],
        "model_flow": f"qwen-72b-q8 -> hermes-3-q8 -> exec ({tool_result['tools_executed']} tools)" if tool_calls else "qwen-72b-q8"
    }


async def generate_with_tool_execution(
    messages: List[ChatMessage],
    max_tokens: int = 2048,
    temperature: float = 0.7,
    max_iterations: int = 5
) -> Dict[str, Any]:
    """
    ReAct loop: Generate → Extract Tools → ACTUALLY EXECUTE → Observe → Repeat

    This is the key innovation: instead of just generating tool call JSON,
    we actually execute the tools and feed real results back to the model.
    """
    from tool_executor import ToolExecutor
    executor = ToolExecutor()

    current_messages = list(messages)
    all_observations = []
    iterations = 0
    user_content = messages[-1].content if messages else ""

    print(f"Starting ReAct loop for: {user_content[:100]}...")

    while iterations < max_iterations:
        iterations += 1
        print(f"\n=== ReAct Iteration {iterations}/{max_iterations} ===")

        # Step 1: Generate response with primary model (Qwen-72B or tools model)
        # Use tools model for faster iteration if primary is slow
        model_to_use = "tools" if iterations > 1 else "primary"
        print(f"Step 1: Generating with {model_to_use} model...")

        response = await generate_with_model(
            model_to_use, current_messages, max_tokens, temperature
        )

        # Step 2: ALWAYS extract tool calls with Hermes-3 Q8 (be aggressive)
        print("Step 2: Extracting tool calls with Hermes-3 Q8...")
        tool_calls = await extract_tool_calls(user_content, response)

        # On first iteration, be very aggressive - if no tools extracted but task seems to need them, force it
        print(f"  Initial tool_calls: {tool_calls}")
        if iterations == 1 and (not tool_calls or tool_calls == []) and needs_tool_extraction(user_content):
            print("  Forcing tool extraction for data-requiring task...")
            # Use a more direct prompt that focuses on the task, not the response
            force_prompt = f"""You MUST output tools for this task. Do NOT say 'no tools needed'.

Task: {user_content}

Output the JSON tool call array NOW. Example: [{{"tool": "Bash", "arguments": {{"command": "ls -la /path"}}}}]

JSON:"""
            tool_calls = await extract_tool_calls(force_prompt, "")
            print(f"  Forced tool_calls: {tool_calls}")

        if not tool_calls:
            # No more tools needed - return final response
            print(f"No more tools needed. Returning final response after {iterations} iterations.")
            return {
                "response": response,
                "observations": all_observations,
                "iterations": iterations,
                "tools_executed": len(all_observations),
                "model_flow": f"react-loop ({iterations} iterations, {len(all_observations)} tools executed)"
            }

        # Step 3: ACTUALLY EXECUTE tools and collect observations
        print(f"Step 3: Executing {len(tool_calls)} tool(s)...")
        observations = []
        for i, tc in enumerate(tool_calls):
            tool_name = tc.get("tool", "unknown")
            print(f"  Executing [{i+1}/{len(tool_calls)}]: {tool_name}")

            result = executor.execute(tc)
            observations.append({
                "tool": tool_name,
                "arguments": tc.get("arguments", {}),
                "result": result
            })

            # Log result summary
            if "error" in result:
                print(f"    ❌ Error: {result['error']}")
            else:
                result_str = str(result)[:100]
                print(f"    ✓ Success: {result_str}...")

        all_observations.extend(observations)

        # Step 4: Feed REAL observations back to model
        print("Step 4: Feeding tool results back to model...")
        obs_text = "\n\n".join([
            f"### Tool: {o['tool']}\n**Arguments:** {json.dumps(o['arguments'])}\n**Result:**\n```json\n{json.dumps(o['result'], indent=2)}\n```"
            for o in observations
        ])

        current_messages.append(ChatMessage(role="assistant", content=response))
        current_messages.append(ChatMessage(
            role="user",
            content=f"""Tool execution completed. Here are the REAL results:

{obs_text}

Based on these actual results, please continue with the task. If you have all the information you need, provide your final answer. If you need more information, specify what additional tools to call."""
        ))

    # Max iterations reached
    print(f"Max iterations ({max_iterations}) reached.")
    return {
        "response": response,
        "observations": all_observations,
        "iterations": iterations,
        "max_iterations_reached": True,
        "tools_executed": len(all_observations),
        "model_flow": f"react-loop (max {max_iterations} iterations, {len(all_observations)} tools executed)"
    }


async def generate_self_consistent(
    messages: List[ChatMessage],
    max_tokens: int = 2048,
    temperature: float = 0.7,
    num_samples: int = 3
) -> Dict[str, Any]:
    """
    Self-Consistency: Generate multiple reasoning paths, select most consistent.
    Research shows +17% accuracy improvement on complex reasoning tasks.

    How it works:
    1. Generate N diverse responses with higher temperature
    2. Use validator to extract and compare key answers
    3. Select the response with the most consistent/common conclusion
    """
    user_content = messages[-1].content if messages else ""

    # Step 1: Generate N diverse responses with higher temperature for diversity
    print(f"Step 1: Generating {num_samples} diverse reasoning paths...")
    responses = []
    for i in range(num_samples):
        print(f"  Path {i+1}/{num_samples}...")
        response = await generate_with_model(
            "primary", messages, max_tokens,
            temperature=max(0.7, temperature)  # Ensure diversity
        )
        responses.append(response)

    # Step 2: Use validator to extract and compare answers
    print("Step 2: Extracting and comparing answers...")

    # Build comparison prompt
    responses_text = "\n\n---\n\n".join([
        f"**Response {i+1}:**\n{resp}" for i, resp in enumerate(responses)
    ])

    comparison_messages = [
        ChatMessage(role="system", content="""You are an answer extractor and consistency checker.
Given multiple responses to the same question, extract the KEY ANSWER/CONCLUSION from each,
then identify the MOST CONSISTENT answer (appears most often or is most supported).

Output format:
EXTRACTED_ANSWERS:
1. [key answer/conclusion from response 1]
2. [key answer/conclusion from response 2]
3. [key answer/conclusion from response 3]

MOST_CONSISTENT: [the answer that appears most often or is most logically supported]
BEST_RESPONSE_INDEX: [1, 2, or 3 - which response best represents the consistent answer]
CONFIDENCE: [HIGH/MEDIUM/LOW based on agreement level between responses]
REASONING: [brief explanation of why this answer is most consistent]"""),
        ChatMessage(role="user", content=f"""Question: {user_content}

{responses_text}

Extract key answers and find the most consistent one:""")
    ]

    analysis = await generate_with_model("validator", comparison_messages, 1024, 0.3)

    # Step 3: Select best response based on analysis
    final_response = responses[0]  # Default to first
    best_index = 0
    confidence = "MEDIUM"

    try:
        # Parse the analysis to find best response
        for line in analysis.split("\n"):
            line_upper = line.upper().strip()
            if "BEST_RESPONSE_INDEX:" in line_upper:
                idx_str = line.split(":")[-1].strip()
                idx = int(''.join(filter(str.isdigit, idx_str))) - 1
                if 0 <= idx < len(responses):
                    best_index = idx
                    final_response = responses[idx]
            elif "CONFIDENCE:" in line_upper:
                conf = line.split(":")[-1].strip().upper()
                if conf in ["HIGH", "MEDIUM", "LOW"]:
                    confidence = conf
    except Exception as e:
        print(f"  Warning: Could not parse analysis: {e}")

    return {
        "response": final_response,
        "num_samples": num_samples,
        "all_responses": responses,
        "analysis": analysis,
        "best_index": best_index + 1,
        "confidence": confidence,
        "model_flow": f"primary x{num_samples} -> validator (self-consistency, confidence: {confidence})"
    }


async def generate_with_critic(
    messages: List[ChatMessage],
    max_tokens: int = 2048,
    temperature: float = 0.7,
    max_iterations: int = 2
) -> Dict[str, Any]:
    """
    CRITIC Framework: Self-critique and revision for edge case detection.
    Research shows +10% improvement on edge case handling.

    How it works:
    1. Generate initial response
    2. Self-critique: Ask model to find issues, edge cases, security problems
    3. Revise: Generate improved response addressing the critique
    4. Repeat until no issues found or max iterations
    """
    user_content = messages[-1].content if messages else ""

    # Step 1: Generate initial response
    print("Step 1: Generating initial response...")
    current_response = await generate_with_model(
        "primary", messages, max_tokens, temperature
    )

    critiques = []
    revision_count = 0

    for iteration in range(max_iterations):
        # Step 2: Self-critique
        print(f"Step 2.{iteration+1}: Self-critiquing (iteration {iteration+1}/{max_iterations})...")

        critique_messages = [
            ChatMessage(role="system", content="""You are a critical code reviewer focused on finding edge cases and potential issues.

Analyze the response for:
1. EDGE CASES not handled (null, empty, negative, overflow, boundary conditions, etc.)
2. ERROR CONDITIONS not addressed (exceptions, invalid input, network failures)
3. ASSUMPTIONS that might not hold in production
4. SECURITY vulnerabilities (injection, XSS, authentication bypass, etc.)
5. PERFORMANCE issues (O(n²) when O(n) possible, memory leaks, etc.)
6. LOGIC ERRORS (off-by-one, incorrect conditions, race conditions)

Output format:
ISSUES_FOUND: [YES/NO]

If YES:
CRITICAL_ISSUES:
- [issue 1 with specific location/code]
- [issue 2 with specific location/code]

SUGGESTIONS:
- [specific fix for issue 1]
- [specific fix for issue 2]

If NO issues found, output:
ISSUES_FOUND: NO
ASSESSMENT: Response handles edge cases appropriately."""),
            ChatMessage(role="user", content=f"""Original question:
{user_content}

Response to critique:
{current_response}

Provide your critical analysis:""")
        ]

        critique = await generate_with_model("validator", critique_messages, 1024, 0.3)
        critiques.append(critique)

        # Step 3: Check if issues found
        critique_upper = critique.upper()
        no_issues = (
            "ISSUES_FOUND: NO" in critique_upper or
            "NO SIGNIFICANT ISSUES" in critique_upper or
            "NO CRITICAL ISSUES" in critique_upper or
            ("ISSUES_FOUND:" in critique_upper and "NO" in critique_upper.split("ISSUES_FOUND:")[1][:20])
        )

        if no_issues:
            print(f"  ✓ No issues found, stopping after {iteration+1} iteration(s)")
            break

        # Step 4: Revise based on critique
        print(f"Step 3.{iteration+1}: Revising based on critique...")
        revision_count += 1

        revision_messages = messages.copy()
        revision_messages.append(ChatMessage(role="assistant", content=current_response))
        revision_messages.append(ChatMessage(
            role="user",
            content=f"""A code review found these issues:

{critique}

Please provide an IMPROVED response that:
1. Addresses ALL the identified issues and edge cases
2. Adds proper error handling where needed
3. Fixes any security vulnerabilities
4. Maintains the original functionality

Provide the complete corrected response:"""
        ))

        current_response = await generate_with_model(
            "primary", revision_messages, max_tokens, temperature
        )

    final_status = "passed" if len(critiques) > 0 and "ISSUES_FOUND: NO" in critiques[-1].upper() else "max_iterations"

    return {
        "response": current_response,
        "iterations": len(critiques),
        "revisions": revision_count,
        "critiques": critiques,
        "final_status": final_status,
        "model_flow": f"primary -> (validator-critique -> primary-revise) x{len(critiques)} [{final_status}]"
    }


# FastAPI app with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown lifecycle for MageAgent server.
    Pre-loads critical models and initializes RAG integration.
    """
    print("=" * 60)
    print(f"MageAgent Server v{VERSION} Starting...")
    print("=" * 60)
    print(f"Available models: {list(MODELS.keys())}")
    print(f"Timeout config: {TIMEOUT_CONFIG}")

    # Pre-load critical models to avoid cold start timeouts
    # Load smallest models first (validator, tools) - these are always needed
    preload_models = ["validator", "tools"]
    for model_type in preload_models:
        try:
            print(f"Pre-loading {model_type} model...")
            await load_model_async(model_type)
            print(f"✓ {model_type} model ready!")
        except Exception as e:
            print(f"⚠ Warning: Could not pre-load {model_type}: {e}")

    # Initialize Project-Specific RAG if project root is set
    app.state.rag_integration = None
    project_root = os.environ.get('MAGEAGENT_PROJECT_ROOT')

    if project_root:
        try:
            from rag_integration import MageAgentRAGIntegration
            print(f"\n[RAG] Initializing for project: {project_root}")
            app.state.rag_integration = MageAgentRAGIntegration(project_root)
            # Force initialization
            app.state.rag_integration._ensure_initialized()
            summary = app.state.rag_integration.get_project_summary()
            print(f"[RAG] ✓ Indexed {summary['total_patterns']} patterns")
            print(f"[RAG]   Language: {summary['primary_language']}")
            print(f"[RAG]   Frameworks: {', '.join(summary['frameworks']) if summary['frameworks'] else 'None'}")
        except Exception as e:
            print(f"[RAG] ⚠ RAG initialization failed: {e}")
            app.state.rag_integration = None
    else:
        print("\n[RAG] Disabled (set MAGEAGENT_PROJECT_ROOT to enable)")

    print("=" * 60)
    print(f"Server ready! Pre-loaded models: {list(loaded_models.keys())}")
    print(f"Endpoint: http://localhost:3457")
    print(f"RAG: {'Enabled' if app.state.rag_integration else 'Disabled'}")
    print("=" * 60)

    yield

    # Shutdown
    print("MageAgent server shutting down...")
    loaded_models.clear()
    model_tokenizers.clear()
    print("Cleanup complete.")


app = FastAPI(
    title="MageAgent Orchestrator",
    description="Multi-Model LLM Server with Validation Patterns",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "name": "MageAgent Orchestrator",
        "version": "2.2.0",  # Version bump for self-consistency and CRITIC patterns
        "models": list(MODELS.keys()),
        "endpoints": [
            "mageagent:auto - Intelligent routing",
            "mageagent:execute - ⭐ REAL tool execution (reads files, runs commands, web search)",
            "mageagent:self_consistent - 🎯 Multiple reasoning paths, +17% accuracy",
            "mageagent:critic - 🔍 Self-critique loop, +10% edge case detection",
            "mageagent:hybrid - Qwen-72B + Hermes-3 (best capability)",
            "mageagent:validated - Generate + validate",
            "mageagent:compete - Competing models",
            "mageagent:tools - Tool calling (Hermes-3 Q8)",
            "mageagent:primary - Direct 72B access (Q8)",
            "mageagent:validator - Direct 7B access",
            "mageagent:competitor - Direct 32B access"
        ],
        "new_in_v2.2": [
            "mageagent:self_consistent - Self-consistency pattern (+17% reasoning accuracy)",
            "mageagent:critic - CRITIC framework (+10% edge case handling)"
        ]
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatible)"""
    models = [
        ModelInfo(id="mageagent:auto", created=int(time.time())),
        ModelInfo(id="mageagent:execute", created=int(time.time())),  # ReAct with real tool execution
        ModelInfo(id="mageagent:self_consistent", created=int(time.time())),  # NEW: +17% reasoning
        ModelInfo(id="mageagent:critic", created=int(time.time())),  # NEW: +10% edge cases
        ModelInfo(id="mageagent:hybrid", created=int(time.time())),
        ModelInfo(id="mageagent:validated", created=int(time.time())),
        ModelInfo(id="mageagent:compete", created=int(time.time())),
        ModelInfo(id="mageagent:tools", created=int(time.time())),
        ModelInfo(id="mageagent:primary", created=int(time.time())),
        ModelInfo(id="mageagent:validator", created=int(time.time())),
        ModelInfo(id="mageagent:competitor", created=int(time.time())),
    ]
    return ModelsResponse(data=models)


@app.get("/health")
async def health():
    # Get RAG status if available
    rag_status = None
    if app.state.rag_integration:
        try:
            rag_status = app.state.rag_integration.get_project_summary()
        except Exception:
            rag_status = {"status": "error"}

    return {
        "status": "healthy",
        "version": VERSION,
        "loaded_models": list(loaded_models.keys()),
        "available_models": list(MODELS.keys()),
        "speculative_decoding": {
            "enabled": SPECULATIVE_CONFIG["enabled"],
            "draft_model_loaded": draft_model_cache["model"] is not None,
            "applicable_models": SPECULATIVE_CONFIG["applicable_models"]
        },
        "rag": rag_status
    }


# ================== Model Management API ==================

class ModelLoadRequest(BaseModel):
    """Request to load a specific model"""
    model: str  # Model key (tools, primary, validator, competitor, glm)


@app.post("/models/load")
async def load_model_endpoint(request: ModelLoadRequest):
    """
    Load a specific model into GPU memory.

    This endpoint allows explicit control over which models are loaded.
    Models are loaded lazily by default, but this allows pre-loading.
    """
    model_key = request.model.lower().replace("mageagent:", "")

    if model_key not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {request.model}. Available: {list(MODELS.keys())}"
        )

    if model_key in loaded_models:
        return {
            "status": "already_loaded",
            "model": model_key,
            "memory_gb": MODELS[model_key]["memory_gb"]
        }

    try:
        start = time.time()
        await load_model_async(model_key)
        duration = time.time() - start

        # Update shared state for cross-app discovery
        update_shared_state()

        return {
            "status": "loaded",
            "model": model_key,
            "memory_gb": MODELS[model_key]["memory_gb"],
            "load_time_sec": round(duration, 1)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load {model_key}: {str(e)}")


class ModelUnloadRequest(BaseModel):
    """Request to unload a specific model"""
    model: str  # Model key (tools, primary, validator, competitor, glm)


@app.post("/models/unload")
async def unload_model_endpoint(request: ModelUnloadRequest):
    """
    Unload a specific model from GPU memory.

    This frees up GPU memory by removing the model from the cache.
    The model can be reloaded later on demand.
    """
    model_key = request.model.lower().replace("mageagent:", "")

    if model_key not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {request.model}. Available: {list(MODELS.keys())}"
        )

    if model_key not in loaded_models:
        return {
            "status": "not_loaded",
            "model": model_key,
            "message": "Model was not loaded"
        }

    try:
        memory_gb = MODELS[model_key]["memory_gb"]

        # Remove from caches
        del loaded_models[model_key]
        if model_key in model_tokenizers:
            del model_tokenizers[model_key]

        # Force garbage collection to free GPU memory
        import gc
        gc.collect()

        # Synchronize MLX to ensure memory is freed
        mx.eval([])  # Empty eval to sync

        # Update shared state for cross-app discovery
        update_shared_state()

        return {
            "status": "unloaded",
            "model": model_key,
            "memory_freed_gb": memory_gb,
            "loaded_models": list(loaded_models.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unload {model_key}: {str(e)}")


@app.post("/models/unload-all")
async def unload_all_models():
    """
    Unload ALL models from GPU memory.

    This is useful for freeing up all GPU memory quickly.
    """
    try:
        unloaded = list(loaded_models.keys())
        total_memory = sum(MODELS[m]["memory_gb"] for m in unloaded)

        # Clear all caches
        loaded_models.clear()
        model_tokenizers.clear()

        # Clear draft model too
        draft_model_cache["model"] = None
        draft_model_cache["tokenizer"] = None

        # Force garbage collection
        import gc
        gc.collect()

        # Synchronize MLX
        mx.eval([])

        # Update shared state for cross-app discovery
        update_shared_state()

        return {
            "status": "all_unloaded",
            "models_unloaded": unloaded,
            "total_memory_freed_gb": total_memory
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unload models: {str(e)}")


@app.get("/models/status")
async def models_status():
    """
    Get detailed status of all models (loaded and available).
    """
    model_status = {}

    for model_key, model_info in MODELS.items():
        model_status[model_key] = {
            "loaded": model_key in loaded_models,
            "memory_gb": model_info["memory_gb"],
            "role": model_info["role"],
            "quant": model_info["quant"],
            "supports_tools": model_info.get("supports_tools", False),
            "tok_per_sec": model_info.get("tok_per_sec", 0)
        }

    total_loaded_memory = sum(
        MODELS[m]["memory_gb"] for m in loaded_models.keys()
    )

    return {
        "models": model_status,
        "loaded_models": list(loaded_models.keys()),
        "total_loaded_memory_gb": total_loaded_memory,
        "draft_model_loaded": draft_model_cache["model"] is not None
    }


# ================== Multi-App Discovery API ==================
# These endpoints enable other apps (VSCode, Claude Code CLI, Nexus CLI)
# to easily discover and use local models

# Shared state file path for cross-app communication
SHARED_STATE_FILE = Path.home() / ".claude" / "nexus-local-compute-state.json"


def update_shared_state():
    """Update the shared state file for cross-app discovery"""
    try:
        state = {
            "server_url": "http://localhost:3457",
            "version": VERSION,
            "status": "running",
            "loaded_models": list(loaded_models.keys()),
            "available_models": list(MODELS.keys()),
            "available_patterns": [
                "mageagent:auto",
                "mageagent:execute",
                "mageagent:self_consistent",
                "mageagent:critic",
                "mageagent:validated",
                "mageagent:compete",
                "mageagent:hybrid",
                "mageagent:tools",
                "mageagent:primary",
                "mageagent:validator",
                "mageagent:fast"
            ],
            "total_loaded_memory_gb": sum(MODELS[m]["memory_gb"] for m in loaded_models.keys()),
            "updated_at": time.time(),
            "api_endpoints": {
                "chat": "/v1/chat/completions",
                "models": "/v1/models",
                "health": "/health",
                "status": "/models/status",
                "load": "/models/load",
                "unload": "/models/unload",
                "discover": "/discover"
            }
        }

        SHARED_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SHARED_STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)

    except Exception as e:
        print(f"Warning: Could not update shared state file: {e}")


@app.get("/discover")
async def discover():
    """
    Discovery endpoint for multi-app access.

    Returns comprehensive information about this server's capabilities,
    enabling other apps to dynamically discover and use local models.

    Usage from other apps:
        curl http://localhost:3457/discover

    Response includes:
        - server_url: Base URL for API calls
        - loaded_models: Currently loaded models (ready for inference)
        - available_models: All models that can be loaded
        - available_patterns: Orchestration patterns (hybrid, execute, etc.)
        - api_endpoints: Available API endpoints with descriptions
        - model_details: Detailed info per model (memory, speed, capabilities)
    """
    # Update shared state file
    update_shared_state()

    model_details = {}
    for model_key, model_info in MODELS.items():
        model_details[model_key] = {
            "api_id": f"mageagent:{model_key}",
            "loaded": model_key in loaded_models,
            "memory_gb": model_info["memory_gb"],
            "role": model_info["role"],
            "quantization": model_info["quant"],
            "supports_tools": model_info.get("supports_tools", False),
            "tokens_per_second": model_info.get("tok_per_sec", 0),
            "timeout_seconds": TIMEOUT_CONFIG.get(model_key, 120)
        }

    return {
        "server_name": "Nexus Local Compute",
        "server_url": "http://localhost:3457",
        "version": VERSION,
        "status": "running",
        "openai_compatible": True,
        "loaded_models": list(loaded_models.keys()),
        "loaded_model_ids": [f"mageagent:{m}" for m in loaded_models.keys()],
        "available_models": list(MODELS.keys()),
        "available_model_ids": [f"mageagent:{m}" for m in MODELS.keys()],
        "total_loaded_memory_gb": sum(MODELS[m]["memory_gb"] for m in loaded_models.keys()),
        "available_patterns": [
            {
                "id": "mageagent:auto",
                "name": "Auto Router",
                "description": "Intelligent task classification and routing"
            },
            {
                "id": "mageagent:execute",
                "name": "Execute",
                "description": "ReAct loop with REAL tool execution"
            },
            {
                "id": "mageagent:self_consistent",
                "name": "Self-Consistent",
                "description": "Multiple reasoning paths, +17% accuracy"
            },
            {
                "id": "mageagent:critic",
                "name": "CRITIC",
                "description": "Self-critique loop for edge cases, +10% improvement"
            },
            {
                "id": "mageagent:hybrid",
                "name": "Hybrid",
                "description": "72B reasoning + 8B tool extraction"
            },
            {
                "id": "mageagent:validated",
                "name": "Validated",
                "description": "Generate + validate with correction loop"
            },
            {
                "id": "mageagent:compete",
                "name": "Compete",
                "description": "Multiple competing models with judge"
            }
        ],
        "model_details": model_details,
        "api_endpoints": {
            "chat_completions": {
                "path": "/v1/chat/completions",
                "method": "POST",
                "description": "OpenAI-compatible chat completions"
            },
            "list_models": {
                "path": "/v1/models",
                "method": "GET",
                "description": "List available models"
            },
            "health": {
                "path": "/health",
                "method": "GET",
                "description": "Server health check"
            },
            "model_status": {
                "path": "/models/status",
                "method": "GET",
                "description": "Detailed model status"
            },
            "load_model": {
                "path": "/models/load",
                "method": "POST",
                "description": "Load a model into memory"
            },
            "unload_model": {
                "path": "/models/unload",
                "method": "POST",
                "description": "Unload a model from memory"
            },
            "statistics": {
                "path": "/stats",
                "method": "GET",
                "description": "Inference statistics"
            }
        },
        "shared_state_file": str(SHARED_STATE_FILE),
        "usage_example": {
            "curl": 'curl -X POST http://localhost:3457/v1/chat/completions -H "Content-Type: application/json" -d \'{"model": "mageagent:hybrid", "messages": [{"role": "user", "content": "Hello"}]}\'',
            "python": 'import openai; client = openai.OpenAI(base_url="http://localhost:3457/v1", api_key="local"); response = client.chat.completions.create(model="mageagent:hybrid", messages=[{"role": "user", "content": "Hello"}])'
        }
    }


@app.get("/rag/status")
async def rag_status():
    """Get detailed RAG integration status"""
    if not app.state.rag_integration:
        return {
            "enabled": False,
            "message": "Set MAGEAGENT_PROJECT_ROOT environment variable to enable"
        }

    try:
        summary = app.state.rag_integration.get_project_summary()
        return {
            "enabled": True,
            **summary
        }
    except Exception as e:
        return {
            "enabled": True,
            "status": "error",
            "error": str(e)
        }


@app.post("/rag/reindex")
async def rag_reindex():
    """Force re-index of project codebase"""
    if not app.state.rag_integration:
        raise HTTPException(status_code=400, detail="RAG not enabled")

    try:
        # Force reinitialization
        app.state.rag_integration._initialized = False
        app.state.rag_integration._ensure_initialized()
        summary = app.state.rag_integration.get_project_summary()
        return {
            "status": "reindexed",
            **summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def stats():
    """Return inference statistics for monitoring throughput"""
    session_duration = time.time() - inference_stats["session_start"]

    return {
        "session": {
            "start": inference_stats["session_start"],
            "duration_sec": round(session_duration, 1),
            "duration_human": f"{int(session_duration // 3600)}h {int((session_duration % 3600) // 60)}m"
        },
        "totals": {
            "requests": inference_stats["total_requests"],
            "completion_tokens": inference_stats["total_tokens_generated"],
            "prompt_tokens": inference_stats["total_prompt_tokens"],
            "total_tokens": inference_stats["total_tokens_generated"] + inference_stats["total_prompt_tokens"]
        },
        "last_request": {
            "timestamp": inference_stats["last_inference"],
            "model": inference_stats["last_model"],
            "pattern": inference_stats["last_pattern"],
            "tokens_per_sec": round(inference_stats["last_tokens_per_sec"], 1),
            "tokens_generated": inference_stats["last_tokens_generated"],
            "duration_sec": round(inference_stats["last_duration_sec"], 2)
        },
        "by_model": {
            model: {
                "requests": inference_stats["requests_by_model"].get(model, 0),
                "completion_tokens": inference_stats["tokens_by_model"].get(model, 0),
                "prompt_tokens": inference_stats["prompt_tokens_by_model"].get(model, 0)
            }
            for model in set(list(inference_stats["requests_by_model"].keys()) +
                           list(inference_stats["tokens_by_model"].keys()))
        },
        "by_pattern": {
            pattern: {
                "requests": inference_stats["requests_by_pattern"].get(pattern, 0),
                "tokens": inference_stats["tokens_by_pattern"].get(pattern, 0)
            }
            for pattern in inference_stats["requests_by_pattern"].keys()
        }
    }


@app.post("/stats/reset")
async def reset_stats():
    """Reset all inference statistics"""
    inference_stats["total_requests"] = 0
    inference_stats["total_tokens_generated"] = 0
    inference_stats["total_prompt_tokens"] = 0
    inference_stats["last_inference"] = None
    inference_stats["last_model"] = None
    inference_stats["last_pattern"] = None
    inference_stats["last_tokens_per_sec"] = 0.0
    inference_stats["last_tokens_generated"] = 0
    inference_stats["last_duration_sec"] = 0.0
    inference_stats["requests_by_model"] = {}
    inference_stats["tokens_by_model"] = {}
    inference_stats["requests_by_pattern"] = {}
    inference_stats["tokens_by_pattern"] = {}
    inference_stats["prompt_tokens_by_model"] = {}
    inference_stats["session_start"] = time.time()

    return {"status": "stats_reset", "session_start": inference_stats["session_start"]}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """OpenAI-compatible chat completions endpoint with optional RAG enrichment"""

    start_time = time.time()
    model_name = request.model

    # Extract user prompt for classification
    user_prompt = request.messages[-1].content if request.messages else ""

    # RAG enrichment for applicable patterns
    messages_to_use = list(request.messages)
    rag_enriched = False

    if app.state.rag_integration and model_name in [
        "mageagent:auto", "mageagent:hybrid", "mageagent:validated",
        "mageagent:compete", "mageagent:primary", "mageagent:self_consistent",
        "mageagent:critic"
    ]:
        try:
            messages_to_use = await app.state.rag_integration.enrich_generation_request(
                request.messages,
                user_prompt,
                model_name,
                max_patterns=3
            )
            if len(messages_to_use) != len(request.messages):
                rag_enriched = True
                print(f"[RAG] Enriched prompt with project patterns")
        except Exception as e:
            print(f"[RAG] Enrichment failed: {e}, using original prompt")

    try:
        if model_name == "mageagent:execute":
            # ReAct loop with REAL tool execution - the key innovation!
            # This actually reads files, runs commands, and searches the web
            result = await generate_with_tool_execution(
                request.messages,
                request.max_tokens or 2048,
                request.temperature or 0.7
            )
            response_text = result["response"]

            # Add execution summary
            if result.get("observations"):
                tools_summary = ", ".join([o["tool"] for o in result["observations"]])
                response_text += f"\n\n---\n*Executed {len(result['observations'])} tools: {tools_summary}*"

            used_model = f"mageagent:execute ({result['model_flow']})"

        elif model_name == "mageagent:self_consistent":
            # Self-Consistency pattern: Multiple reasoning paths, select most consistent
            # Research shows +17% accuracy improvement on complex reasoning tasks
            result = await generate_self_consistent(
                messages_to_use,  # RAG-enriched messages
                request.max_tokens or 2048,
                request.temperature or 0.7,
                num_samples=3  # Generate 3 diverse responses
            )
            response_text = result["response"]
            # Add consistency summary
            rag_tag = " +RAG" if rag_enriched else ""
            response_text += f"\n\n---\n*Self-consistency: {result['num_samples']} paths, best #{result['best_index']}, confidence: {result['confidence']}{rag_tag}*"
            used_model = f"mageagent:self_consistent ({result['model_flow']})"

        elif model_name == "mageagent:critic":
            # CRITIC Framework: Self-critique and revision for edge case detection
            # Research shows +10% improvement on edge case handling
            result = await generate_with_critic(
                messages_to_use,  # RAG-enriched messages
                request.max_tokens or 2048,
                request.temperature or 0.7,
                max_iterations=2  # Up to 2 critique-revise cycles
            )
            response_text = result["response"]
            # Add critique summary
            rag_tag = " +RAG" if rag_enriched else ""
            response_text += f"\n\n---\n*CRITIC: {result['iterations']} critiques, {result['revisions']} revisions, status: {result['final_status']}{rag_tag}*"
            used_model = f"mageagent:critic ({result['model_flow']})"

        elif model_name == "mageagent:validated":
            # Generate + validate pattern (with real tool execution)
            result = await generate_with_validation(
                messages_to_use,  # RAG-enriched messages
                request.max_tokens or 2048,
                request.temperature or 0.7
            )
            response_text = result["response"]
            # Add execution summary if tools were run
            rag_tag = " +RAG" if rag_enriched else ""
            if result.get("tools_executed", 0) > 0:
                tools_summary = ", ".join([o["tool"] for o in result.get("observations", [])])
                response_text += f"\n\n---\n*Executed {result['tools_executed']} tools: {tools_summary}{rag_tag}*"
            used_model = f"mageagent:validated ({result.get('model_flow', '72B-Q8 -> 7B-validator')})"

        elif model_name == "mageagent:compete":
            # Competing models pattern (with real tool execution)
            result = await generate_competing(
                messages_to_use,  # RAG-enriched messages
                request.max_tokens or 2048,
                request.temperature or 0.7
            )
            response_text = result["response"]
            # Add execution summary if tools were run
            rag_tag = " +RAG" if rag_enriched else ""
            if result.get("tools_executed", 0) > 0:
                tools_summary = ", ".join([o["tool"] for o in result.get("observations", [])])
                response_text += f"\n\n---\n*Executed {result['tools_executed']} tools: {tools_summary}{rag_tag}*"
            winner = result.get('winner', '?')
            used_model = f"mageagent:compete ({result.get('model_flow', f'winner: {winner}')})"

        elif model_name == "mageagent:hybrid":
            # Hybrid pattern: Qwen-72B Q8 reasoning + Hermes-3 Q8 tools (with real execution)
            result = await generate_hybrid(
                messages_to_use,  # RAG-enriched messages
                request.max_tokens or 2048,
                request.temperature or 0.7
            )
            response_text = result["response"]
            # Add execution summary if tools were run
            rag_tag = " +RAG" if rag_enriched else ""
            if result.get("tools_executed", 0) > 0:
                tools_summary = ", ".join([o["tool"] for o in result.get("observations", [])])
                response_text += f"\n\n---\n*Executed {result['tools_executed']} tools: {tools_summary}{rag_tag}*"
            used_model = f"mageagent:hybrid ({result['model_flow']})"

        elif model_name == "mageagent:auto":
            # Intelligent routing based on task classification
            task_type = classify_task(user_prompt)
            print(f"Task classified as: {task_type}")

            if task_type == "coding":
                # Use validation pattern for coding tasks (with real tool execution)
                result = await generate_with_validation(
                    messages_to_use,  # RAG-enriched messages
                    request.max_tokens or 2048,
                    request.temperature or 0.7
                )
                response_text = result["response"]
                rag_tag = " +RAG" if rag_enriched else ""
                if result.get("tools_executed", 0) > 0:
                    tools_summary = ", ".join([o["tool"] for o in result.get("observations", [])])
                    response_text += f"\n\n---\n*Executed {result['tools_executed']} tools: {tools_summary}{rag_tag}*"
                used_model = f"mageagent:auto->validated ({result.get('model_flow', '')})"
            elif task_type == "reasoning":
                # Use hybrid for reasoning (with real tool execution)
                result = await generate_hybrid(
                    messages_to_use,  # RAG-enriched messages
                    request.max_tokens or 2048,
                    request.temperature or 0.7
                )
                response_text = result["response"]
                rag_tag = " +RAG" if rag_enriched else ""
                if result.get("tools_executed", 0) > 0:
                    tools_summary = ", ".join([o["tool"] for o in result.get("observations", [])])
                    response_text += f"\n\n---\n*Executed {result['tools_executed']} tools: {tools_summary}{rag_tag}*"
                used_model = f"mageagent:auto->hybrid ({result.get('model_flow', '')})"
            else:
                # Use fast validator for simple tasks (no tools needed)
                response_text = await generate_with_model(
                    "validator",
                    messages_to_use,  # RAG-enriched messages
                    request.max_tokens or 2048,
                    request.temperature or 0.7
                )
                used_model = "mageagent:auto->validator"

        elif model_name in ["mageagent:primary", "mageagent:reasoning"]:
            # Direct primary model access
            response_text = await generate_with_model(
                "primary",
                messages_to_use,  # RAG-enriched messages
                request.max_tokens or 2048,
                request.temperature or 0.7
            )
            used_model = "mageagent:primary"

        elif model_name in ["mageagent:validator", "mageagent:fast"]:
            # Direct validator model access
            response_text = await generate_with_model(
                "validator",
                messages_to_use,  # RAG-enriched messages
                request.max_tokens or 2048,
                request.temperature or 0.7
            )
            used_model = "mageagent:validator"

        elif model_name in ["mageagent:competitor", "mageagent:coding"]:
            # Direct competitor model access
            response_text = await generate_with_model(
                "competitor",
                request.messages,
                request.max_tokens or 2048,
                request.temperature or 0.7
            )
            used_model = "mageagent:competitor"

        elif model_name in ["mageagent:tools", "mageagent:hermes"]:
            # Direct tools model access (Hermes-3 Q8 for tool calling)
            response_text = await generate_with_model(
                "tools",
                request.messages,
                request.max_tokens or 2048,
                request.temperature or 0.7
            )
            used_model = "mageagent:tools"

        else:
            # Default to auto
            response_text = await generate_with_model(
                "validator",
                request.messages,
                request.max_tokens or 2048,
                request.temperature or 0.7
            )
            used_model = "mageagent:default->validator"

        elapsed = time.time() - start_time
        print(f"Request completed in {elapsed:.1f}s using {used_model}")

        # Estimate token counts
        prompt_tokens = sum(len(m.content.split()) for m in request.messages)
        completion_tokens = len(response_text.split())

        return ChatResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=used_model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except GenerationTimeoutError as e:
        print(f"Timeout: {e}")
        raise HTTPException(
            status_code=504,  # Gateway Timeout
            detail=str(e)
        )
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=3457,
        timeout_keep_alive=700,  # Must exceed longest generation time (600s for 72B)
        limit_concurrency=4,     # Limit concurrent requests - Metal can't handle many
    )
