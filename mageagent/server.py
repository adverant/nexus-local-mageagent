#!/usr/bin/env python3
"""
MageAgent Orchestrator - Multi-Model LLM Server for MLX
Provides OpenAI-compatible API with intelligent model routing and validation patterns

Patterns:
- mageagent:auto - Intelligent task classification and routing
- mageagent:execute - ReAct loop with REAL tool execution (reads files, runs commands)
- mageagent:validated - Generate + validate with correction loop
- mageagent:compete - Competing models with judge
- mageagent:hybrid - Qwen-72B reasoning + Hermes-3 tool extraction
- mageagent:tools - Tool-calling specialist (Hermes-3 Q8)
- mageagent:primary - Direct access to 72B model
- mageagent:validator - Direct access to 7B validator
- mageagent:fast - Quick responses with 7B model
"""

# Version - keep in sync with package.json
VERSION = "2.1.0"

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
    }
}

# Lazy-loaded models cache
loaded_models: Dict[str, Any] = {}
model_tokenizers: Dict[str, Any] = {}

# Stats tracking for throughput monitoring
inference_stats: Dict[str, Any] = {
    "total_requests": 0,
    "total_tokens_generated": 0,
    "last_inference": None,  # timestamp
    "last_model": None,
    "last_tokens_per_sec": 0.0,
    "last_tokens_generated": 0,
    "last_duration_sec": 0.0,
    "requests_by_model": {},
    "tokens_by_model": {},
}

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
    """Internal generation function that does the actual work."""
    model, tokenizer = await load_model_async(model_type)
    prompt = format_chat_prompt(messages, tokenizer)

    # Track start time for throughput calculation
    gen_start = time.time()

    # Run generation in a thread pool to not block event loop
    loop = asyncio.get_event_loop()
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
    inference_stats["last_model"] = model_type
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
        r'\bcommand\b', r'\bterminal\b', r'\bcli\b'
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

What tools should be executed to complete this task? Output JSON array only:""")
    ]

    tool_response = await generate_with_model("tools", tool_messages, 512, 0.1)

    # Parse tool calls
    try:
        match = re.search(r'\[.*\]', tool_response, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass
    return None


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
        if iterations == 1 and not tool_calls and needs_tool_extraction(user_content):
            print("  Forcing tool extraction for data-requiring task...")
            tool_calls = await extract_tool_calls(
                user_content + "\n\nIMPORTANT: This task REQUIRES using tools to get real data. Do NOT just explain - execute tools!",
                response
            )

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


# FastAPI app with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown lifecycle for MageAgent server.
    Pre-loads critical models to avoid cold start timeouts.
    """
    print("=" * 60)
    print("MageAgent Server v2.0 Starting...")
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

    print("=" * 60)
    print(f"Server ready! Pre-loaded models: {list(loaded_models.keys())}")
    print(f"Endpoint: http://localhost:3457")
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
        "version": "2.0.0",  # Major version bump for tool execution
        "models": list(MODELS.keys()),
        "endpoints": [
            "mageagent:auto - Intelligent routing",
            "mageagent:execute - ⭐ REAL tool execution (reads files, runs commands, web search)",
            "mageagent:hybrid - Qwen-72B + Hermes-3 (best capability)",
            "mageagent:validated - Generate + validate",
            "mageagent:compete - Competing models",
            "mageagent:tools - Tool calling (Hermes-3 Q8)",
            "mageagent:primary - Direct 72B access (Q8)",
            "mageagent:validator - Direct 7B access",
            "mageagent:competitor - Direct 32B access"
        ],
        "new_in_v2": "mageagent:execute - ReAct loop that ACTUALLY executes tools instead of hallucinating"
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatible)"""
    models = [
        ModelInfo(id="mageagent:auto", created=int(time.time())),
        ModelInfo(id="mageagent:execute", created=int(time.time())),  # NEW: Real tool execution!
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
    return {
        "status": "healthy",
        "version": VERSION,
        "loaded_models": list(loaded_models.keys()),
        "available_models": list(MODELS.keys())
    }


@app.get("/stats")
async def stats():
    """Return inference statistics for monitoring throughput"""
    return {
        "total_requests": inference_stats["total_requests"],
        "total_tokens_generated": inference_stats["total_tokens_generated"],
        "last_inference": inference_stats["last_inference"],
        "last_model": inference_stats["last_model"],
        "last_tokens_per_sec": inference_stats["last_tokens_per_sec"],
        "last_tokens_generated": inference_stats["last_tokens_generated"],
        "last_duration_sec": inference_stats["last_duration_sec"],
        "requests_by_model": inference_stats["requests_by_model"],
        "tokens_by_model": inference_stats["tokens_by_model"],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """OpenAI-compatible chat completions endpoint"""

    start_time = time.time()
    model_name = request.model

    # Extract user prompt for classification
    user_prompt = request.messages[-1].content if request.messages else ""

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

        elif model_name == "mageagent:validated":
            # Generate + validate pattern (with real tool execution)
            result = await generate_with_validation(
                request.messages,
                request.max_tokens or 2048,
                request.temperature or 0.7
            )
            response_text = result["response"]
            # Add execution summary if tools were run
            if result.get("tools_executed", 0) > 0:
                tools_summary = ", ".join([o["tool"] for o in result.get("observations", [])])
                response_text += f"\n\n---\n*Executed {result['tools_executed']} tools: {tools_summary}*"
            used_model = f"mageagent:validated ({result.get('model_flow', '72B-Q8 -> 7B-validator')})"

        elif model_name == "mageagent:compete":
            # Competing models pattern (with real tool execution)
            result = await generate_competing(
                request.messages,
                request.max_tokens or 2048,
                request.temperature or 0.7
            )
            response_text = result["response"]
            # Add execution summary if tools were run
            if result.get("tools_executed", 0) > 0:
                tools_summary = ", ".join([o["tool"] for o in result.get("observations", [])])
                response_text += f"\n\n---\n*Executed {result['tools_executed']} tools: {tools_summary}*"
            winner = result.get('winner', '?')
            used_model = f"mageagent:compete ({result.get('model_flow', f'winner: {winner}')})"

        elif model_name == "mageagent:hybrid":
            # Hybrid pattern: Qwen-72B Q8 reasoning + Hermes-3 Q8 tools (with real execution)
            result = await generate_hybrid(
                request.messages,
                request.max_tokens or 2048,
                request.temperature or 0.7
            )
            response_text = result["response"]
            # Add execution summary if tools were run
            if result.get("tools_executed", 0) > 0:
                tools_summary = ", ".join([o["tool"] for o in result.get("observations", [])])
                response_text += f"\n\n---\n*Executed {result['tools_executed']} tools: {tools_summary}*"
            used_model = f"mageagent:hybrid ({result['model_flow']})"

        elif model_name == "mageagent:auto":
            # Intelligent routing based on task classification
            task_type = classify_task(user_prompt)
            print(f"Task classified as: {task_type}")

            if task_type == "coding":
                # Use validation pattern for coding tasks (with real tool execution)
                result = await generate_with_validation(
                    request.messages,
                    request.max_tokens or 2048,
                    request.temperature or 0.7
                )
                response_text = result["response"]
                if result.get("tools_executed", 0) > 0:
                    tools_summary = ", ".join([o["tool"] for o in result.get("observations", [])])
                    response_text += f"\n\n---\n*Executed {result['tools_executed']} tools: {tools_summary}*"
                used_model = f"mageagent:auto->validated ({result.get('model_flow', '')})"
            elif task_type == "reasoning":
                # Use hybrid for reasoning (with real tool execution)
                result = await generate_hybrid(
                    request.messages,
                    request.max_tokens or 2048,
                    request.temperature or 0.7
                )
                response_text = result["response"]
                if result.get("tools_executed", 0) > 0:
                    tools_summary = ", ".join([o["tool"] for o in result.get("observations", [])])
                    response_text += f"\n\n---\n*Executed {result['tools_executed']} tools: {tools_summary}*"
                used_model = f"mageagent:auto->hybrid ({result.get('model_flow', '')})"
            else:
                # Use fast validator for simple tasks (no tools needed)
                response_text = await generate_with_model(
                    "validator",
                    request.messages,
                    request.max_tokens or 2048,
                    request.temperature or 0.7
                )
                used_model = "mageagent:auto->validator"

        elif model_name in ["mageagent:primary", "mageagent:reasoning"]:
            # Direct primary model access
            response_text = await generate_with_model(
                "primary",
                request.messages,
                request.max_tokens or 2048,
                request.temperature or 0.7
            )
            used_model = "mageagent:primary"

        elif model_name in ["mageagent:validator", "mageagent:fast"]:
            # Direct validator model access
            response_text = await generate_with_model(
                "validator",
                request.messages,
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
