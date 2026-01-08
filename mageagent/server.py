#!/usr/bin/env python3
"""
MageAgent Orchestrator - Multi-Model LLM Server for MLX
Provides OpenAI-compatible API with intelligent model routing and validation patterns

Patterns:
- mageagent:auto - Intelligent task classification and routing
- mageagent:validated - Generate + validate with correction loop
- mageagent:compete - Competing models with judge
- mageagent:tools - Tool-calling specialist (Hermes-3 Q8)
- mageagent:primary - Direct access to 72B model
- mageagent:validator - Direct access to 7B validator
- mageagent:fast - Quick responses with 7B model
"""

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mlx.core as mx
from mlx_lm import load, generate

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
    """Lazy-load and cache models"""
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


async def generate_with_model(
    model_type: str,
    messages: List[ChatMessage],
    max_tokens: int = 2048,
    temperature: float = 0.7
) -> str:
    """Generate response using specified model"""
    model, tokenizer = get_model(model_type)
    prompt = format_chat_prompt(messages, tokenizer)

    # Run generation in a thread pool to not block
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

    return response


def needs_tool_extraction(prompt: str) -> bool:
    """Check if the prompt requires tool extraction"""
    tool_patterns = [
        r'\bread\b.*\bfile\b', r'\bwrite\b.*\bfile\b', r'\blist\b.*\bdir',
        r'\bexecute\b', r'\brun\b', r'\bcreate\b.*\bfile\b', r'\bdelete\b',
        r'\bsearch\b', r'\bfind\b', r'\bedit\b', r'\bmodify\b',
        r'\btool\b', r'\bfunction\b.*\bcall\b', r'\bapi\b.*\bcall\b',
        r'\bglob\b', r'\bgrep\b', r'\bbash\b', r'\bshell\b'
    ]
    return any(re.search(p, prompt.lower()) for p in tool_patterns)


async def extract_tool_calls(user_content: str, response: str) -> list:
    """
    Use Hermes-3 Q8 to extract tool calls from any response.
    This is the ONLY model that should handle tool extraction.
    """
    print("Hermes-3 Q8 extracting tool calls...")
    tool_messages = [
        ChatMessage(role="system", content="""You are a tool-calling assistant. Based on the task and response, extract required tool calls.

Output tool calls as JSON array:
[{"tool": "tool_name", "arguments": {"arg1": "value1"}}]

Available tools:
- Read: {"file_path": "path"} - Read file contents
- Write: {"file_path": "path", "content": "content"} - Write to file
- Edit: {"file_path": "path", "old_string": "text", "new_string": "text"} - Edit file
- Bash: {"command": "shell_command"} - Execute shell command
- Glob: {"pattern": "**/*.py", "path": "dir"} - Find files by pattern
- Grep: {"pattern": "regex", "path": "dir"} - Search file contents

If no tools are needed, output: []"""),
        ChatMessage(role="user", content=f"""Task: {user_content}

Response:
{response}

Extract tool calls (JSON array only):""")
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

    # Step 4: Extract tool calls via Hermes-3 if needed
    tool_calls = None
    if needs_tool_extraction(user_content):
        print("Step 4: Hermes-3 Q8 extracting tool calls...")
        tool_calls = await extract_tool_calls(user_content, primary_response)

    return {
        "response": primary_response,
        "validation": validation,
        "revised": needs_revision,
        "tool_calls": tool_calls,
        "model_flow": "72B-Q8 -> 7B-validator -> hermes-3-Q8" if tool_calls else "72B-Q8 -> 7B-validator"
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

    # Step 3: Extract tool calls via Hermes-3 if needed
    tool_calls = None
    if needs_tool_extraction(user_content):
        print("Step 3: Hermes-3 Q8 extracting tool calls...")
        tool_calls = await extract_tool_calls(user_content, best_response)

    return {
        "response": best_response,
        "winner": winner,
        "judgment": judgment,
        "solution_a": primary_response,
        "solution_b": competitor_response,
        "tool_calls": tool_calls,
        "model_flow": f"72B + 32B -> 7B-judge -> hermes-3-Q8 (winner: {winner})" if tool_calls else f"72B + 32B -> 7B-judge (winner: {winner})"
    }


async def generate_hybrid(
    messages: List[ChatMessage],
    max_tokens: int = 2048,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Hybrid pattern: Qwen-72B Q8 for reasoning + Hermes-3 Q8 for tool execution
    ALWAYS extracts tools via Hermes-3 for best capability.
    """

    user_content = messages[-1].content if messages else ""

    # Step 1: Qwen-72B generates the main response with reasoning
    print("Step 1: Qwen-72B Q8 analyzing and generating response...")
    primary_response = await generate_with_model(
        "primary", messages, max_tokens, temperature
    )

    # Step 2: ALWAYS extract tool calls via Hermes-3 (hybrid = best capability)
    tool_calls = None
    if needs_tool_extraction(user_content):
        print("Step 2: Hermes-3 Q8 extracting tool calls...")
        tool_calls = await extract_tool_calls(user_content, primary_response)

    return {
        "response": primary_response,
        "tool_calls": tool_calls,
        "model_flow": "qwen-72b-q8 -> hermes-3-q8" if tool_calls else "qwen-72b-q8"
    }


# FastAPI app with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Pre-load validator model (smallest, always needed)
    print("MageAgent server starting...")
    print(f"Available models: {list(MODELS.keys())}")

    # Only pre-load validator since it's small and always used
    try:
        print("Pre-loading validator model...")
        get_model("validator")
        print("Validator model ready!")
    except Exception as e:
        print(f"Warning: Could not pre-load validator: {e}")

    yield

    # Shutdown
    print("MageAgent server shutting down...")
    loaded_models.clear()
    model_tokenizers.clear()


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
        "version": "1.1.0",
        "models": list(MODELS.keys()),
        "endpoints": [
            "mageagent:auto - Intelligent routing",
            "mageagent:hybrid - Qwen-72B + Hermes-3 (best capability)",
            "mageagent:validated - Generate + validate",
            "mageagent:compete - Competing models",
            "mageagent:tools - Tool calling (Hermes-3 Q8)",
            "mageagent:primary - Direct 72B access (Q8)",
            "mageagent:validator - Direct 7B access",
            "mageagent:competitor - Direct 32B access"
        ]
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatible)"""
    models = [
        ModelInfo(id="mageagent:auto", created=int(time.time())),
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
        "loaded_models": list(loaded_models.keys()),
        "available_models": list(MODELS.keys())
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """OpenAI-compatible chat completions endpoint"""

    start_time = time.time()
    model_name = request.model

    # Extract user prompt for classification
    user_prompt = request.messages[-1].content if request.messages else ""

    try:
        if model_name == "mageagent:validated":
            # Generate + validate pattern
            result = await generate_with_validation(
                request.messages,
                request.max_tokens or 2048,
                request.temperature or 0.7
            )
            response_text = result["response"]
            # Include tool calls in response if present
            if result.get("tool_calls"):
                response_text += f"\n\n<tool_calls>\n{json.dumps(result['tool_calls'], indent=2)}\n</tool_calls>"
            used_model = f"mageagent:validated ({result.get('model_flow', '72B-Q8 -> 7B-validator')})"

        elif model_name == "mageagent:compete":
            # Competing models pattern
            result = await generate_competing(
                request.messages,
                request.max_tokens or 2048,
                request.temperature or 0.7
            )
            response_text = result["response"]
            # Include tool calls in response if present
            if result.get("tool_calls"):
                response_text += f"\n\n<tool_calls>\n{json.dumps(result['tool_calls'], indent=2)}\n</tool_calls>"
            winner = result.get('winner', '?')
            used_model = f"mageagent:compete ({result.get('model_flow', f'winner: {winner}')})"

        elif model_name == "mageagent:hybrid":
            # Hybrid pattern: Qwen-72B Q8 reasoning + Hermes-3 Q8 tools
            result = await generate_hybrid(
                request.messages,
                request.max_tokens or 2048,
                request.temperature or 0.7
            )
            response_text = result["response"]
            # Include tool calls in response if present
            if result.get("tool_calls"):
                response_text += f"\n\n<tool_calls>\n{json.dumps(result['tool_calls'], indent=2)}\n</tool_calls>"
            used_model = f"mageagent:hybrid ({result['model_flow']})"

        elif model_name == "mageagent:auto":
            # Intelligent routing based on task classification
            task_type = classify_task(user_prompt)
            print(f"Task classified as: {task_type}")

            if task_type == "coding":
                # Use validation pattern for coding tasks (includes Hermes-3 tools)
                result = await generate_with_validation(
                    request.messages,
                    request.max_tokens or 2048,
                    request.temperature or 0.7
                )
                response_text = result["response"]
                if result.get("tool_calls"):
                    response_text += f"\n\n<tool_calls>\n{json.dumps(result['tool_calls'], indent=2)}\n</tool_calls>"
                used_model = f"mageagent:auto->validated ({result.get('model_flow', '')})"
            elif task_type == "reasoning":
                # Use hybrid for reasoning (includes Hermes-3 tools)
                result = await generate_hybrid(
                    request.messages,
                    request.max_tokens or 2048,
                    request.temperature or 0.7
                )
                response_text = result["response"]
                if result.get("tool_calls"):
                    response_text += f"\n\n<tool_calls>\n{json.dumps(result['tool_calls'], indent=2)}\n</tool_calls>"
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
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=3457)
