# MageAgent Orchestration Patterns

> Deep dive into multi-model orchestration patterns for optimal AI performance

## Overview

MageAgent provides intelligent multi-model orchestration using Apple's MLX framework. Instead of relying on a single model, MageAgent routes tasks through specialized model combinations to achieve quality approaching Claude Opus while running entirely locally.

**Key Innovation**: All patterns route tool extraction through **Hermes-3 Q8**, ensuring reliable function calling regardless of which reasoning model generated the response.

## Pattern Summary

| Pattern | Models Used | Quality Gain | Latency | Use Case |
|---------|-------------|--------------|---------|----------|
| `hybrid` | 72B + Hermes | Baseline+ | 30-60s | Best capability |
| `validated` | 72B + 7B + Hermes | +5-10% | 40-80s | Code with error checking |
| `compete` | 72B + 32B + 7B + Hermes | +10-15% | 60-120s | Critical decisions |
| `auto` | Dynamic | Variable | Variable | Automatic optimization |
| `tools` | Hermes only | Fast | 3-5s | Quick tool calls |
| `primary` | 72B only | Direct | 20-40s | Complex reasoning |

---

## Pattern Details

### 1. Hybrid Pattern (Recommended)

**Command**: `mageagent:hybrid`

The hybrid pattern combines Qwen-72B Q8's superior reasoning with Hermes-3 Q8's reliable tool calling.

```
User Request
     │
     ▼
┌─────────────────────┐
│   Qwen-72B Q8       │  ← Primary reasoning
│   (77GB, 8 tok/s)   │
└─────────────────────┘
     │
     │ Response with reasoning
     ▼
┌─────────────────────┐
│   Hermes-3 Q8       │  ← Tool extraction (if needed)
│   (9GB, 50 tok/s)   │
└─────────────────────┘
     │
     ▼
Final Response + Tool Calls
```

**Flow**:
1. Qwen-72B Q8 analyzes the request and generates comprehensive response
2. If tools are needed, Hermes-3 Q8 extracts structured tool calls
3. Response returned with both reasoning and executable tool calls

**Best For**:
- Tasks requiring both analysis AND file operations
- Architecture decisions that need implementation
- Complex code generation with file modifications
- General-purpose "best effort" requests

**Example**:
```bash
curl -X POST http://localhost:3457/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mageagent:hybrid",
    "messages": [{
      "role": "user",
      "content": "Analyze the architecture of src/ and suggest refactoring improvements, then create a plan file"
    }]
  }'
```

---

### 2. Validated Pattern

**Command**: `mageagent:validated`

The validated pattern adds a quality check step that catches errors before output.

```
User Request
     │
     ▼
┌─────────────────────┐
│   Qwen-72B Q8       │  ← Generate initial response
│   (Primary)         │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│   Qwen-7B Q4        │  ← Validate for errors
│   (Validator)       │
│   PASS / FAIL       │
└─────────────────────┘
     │
     │ If FAIL
     ▼
┌─────────────────────┐
│   Qwen-72B Q8       │  ← Regenerate with feedback
│   (Revision)        │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│   Hermes-3 Q8       │  ← Extract tool calls
│   (Tools)           │
└─────────────────────┘
     │
     ▼
Final Response + Tool Calls
```

**Flow**:
1. Qwen-72B Q8 generates initial response
2. Qwen-7B Q4 validates for:
   - Syntax errors
   - Logic bugs
   - Missing error handling
   - Security vulnerabilities
   - Performance problems
3. If issues found, 72B regenerates with feedback
4. Hermes-3 Q8 extracts tool calls if needed

**Quality Improvement**: +5-10% over single model

**Best For**:
- Code generation where correctness is critical
- Production code that needs to work first time
- Complex algorithms requiring validation
- Code reviews and improvements

**Validation Prompt Used**:
```
Review the response for issues:
1. Syntax errors
2. Logic bugs
3. Missing error handling
4. Security vulnerabilities
5. Performance problems

Output ONLY "PASS" if no issues found, or "FAIL: <brief list of issues>"
```

---

### 3. Compete Pattern

**Command**: `mageagent:compete`

The compete pattern generates two independent solutions and uses a judge to select the best one.

```
User Request
     │
     ├────────────────────────────────────┐
     ▼                                    ▼
┌─────────────────┐              ┌─────────────────┐
│  Qwen-72B Q8    │              │  Qwen-32B Q4    │
│  Solution A     │              │  Solution B     │
│  (Reasoning)    │              │  (Coding)       │
└─────────────────┘              └─────────────────┘
     │                                    │
     └───────────────┬────────────────────┘
                     ▼
            ┌─────────────────┐
            │   Qwen-7B Q4    │
            │   (Judge)       │
            │   Pick A or B   │
            └─────────────────┘
                     │
                     ▼
            ┌─────────────────┐
            │  Hermes-3 Q8    │
            │  (Tools)        │
            └─────────────────┘
                     │
                     ▼
            Best Solution + Tool Calls
```

**Flow**:
1. Qwen-72B Q8 generates Solution A (reasoning-focused)
2. Qwen-32B Q4 generates Solution B (coding-focused)
3. Qwen-7B Q4 judges both solutions on:
   - Correctness
   - Efficiency
   - Readability
   - Error handling
4. Best solution selected
5. Hermes-3 Q8 extracts tool calls if needed

**Quality Improvement**: +10-15% over single model

**Best For**:
- Critical production code
- Complex architectural decisions
- When you want multiple perspectives
- Important features where quality > speed

**Judge Prompt Used**:
```
Compare two solutions and pick the better one.
Consider: correctness, efficiency, readability, error handling.
Output ONLY "A" or "B" followed by a brief one-sentence explanation.
```

**Note**: Models run sequentially (not parallel) due to Metal GPU limitations with large models.

---

### 4. Auto Pattern

**Command**: `mageagent:auto`

The auto pattern intelligently routes tasks based on classification.

```
User Request
     │
     ▼
┌─────────────────────┐
│  Task Classifier    │
│  (Pattern matching) │
└─────────────────────┘
     │
     ├─── Coding task ──────► Validated Pattern
     │
     ├─── Reasoning task ───► Hybrid Pattern
     │
     └─── Simple task ──────► Validator (fast)
```

**Classification Criteria**:

| Task Type | Patterns Detected | Route |
|-----------|-------------------|-------|
| Coding | `implement`, `function`, `class`, `refactor`, `fix bug`, code blocks | Validated |
| Reasoning | `explain`, `analyze`, `plan`, `design`, `architecture`, `compare` | Hybrid |
| Simple | Everything else | Validator (fast) |

**Best For**:
- General-purpose usage
- When you don't want to think about which pattern
- Mixed workloads
- Default configuration

---

### 5. Tools Pattern

**Command**: `mageagent:tools`

Direct access to Hermes-3 Q8 for fast tool calling.

```
User Request
     │
     ▼
┌─────────────────────┐
│   Hermes-3 Q8       │
│   (9GB, 50 tok/s)   │
│   Tool specialist   │
└─────────────────────┘
     │
     ▼
Response + Tool Calls
```

**Best For**:
- Quick file operations
- Simple tool calls
- Fast iterations
- When you don't need heavy reasoning

**Why Q8 for Tools?**

Research shows Q4 quantization breaks tool calling in most models. Q8 preserves the precision needed for reliable function calling.

| Quantization | Tool Calling | Notes |
|--------------|--------------|-------|
| Q8_0 | Reliable | Recommended for tool calling |
| Q6_K | Partial | May work, less reliable |
| Q4_K_M | Broken | Do not use for tools |

---

### 6. Primary Pattern

**Command**: `mageagent:primary`

Direct access to Qwen-72B Q8 without tool extraction.

```
User Request
     │
     ▼
┌─────────────────────┐
│   Qwen-72B Q8       │
│   (77GB, 8 tok/s)   │
│   Full reasoning    │
└─────────────────────┘
     │
     ▼
Response (no tools)
```

**Best For**:
- Pure reasoning tasks
- Analysis and explanation
- Architecture planning
- When you don't need file operations

---

## Model Specifications

| Model | Role | Quantization | Memory | Speed | Tool Calling |
|-------|------|--------------|--------|-------|--------------|
| Qwen2.5-72B | Primary | Q8_0 | 77GB | 8 tok/s | Yes |
| Qwen2.5-Coder-32B | Competitor | Q4_K_M | 18GB | 25 tok/s | No |
| Qwen2.5-Coder-7B | Validator | Q4_K_M | 5GB | 105 tok/s | No |
| Hermes-3-Llama-8B | Tools | Q8_0 | 9GB | 50 tok/s | Yes |

**Total Memory**: ~109GB (fits in 128GB with 19GB headroom)

---

## Tool Extraction Architecture

The key innovation in MageAgent is centralized tool extraction through Hermes-3 Q8.

### Why Centralized Tool Extraction?

1. **Q4 breaks tools**: Lower quantization loses the precision needed for structured JSON output
2. **Consistent behavior**: Same tool format regardless of which model reasoned
3. **Separation of concerns**: Reasoning models focus on reasoning, tool model focuses on tools
4. **Reliability**: Hermes-3 is specifically trained for function calling

### Tool Extraction Prompt

```python
"""You are a tool-calling assistant. Based on the task and response, extract required tool calls.

Output tool calls as JSON array:
[{"tool": "tool_name", "arguments": {"arg1": "value1"}}]

Available tools:
- Read: {"file_path": "path"} - Read file contents
- Write: {"file_path": "path", "content": "content"} - Write to file
- Edit: {"file_path": "path", "old_string": "text", "new_string": "text"} - Edit file
- Bash: {"command": "shell_command"} - Execute shell command
- Glob: {"pattern": "**/*.py", "path": "dir"} - Find files by pattern
- Grep: {"pattern": "regex", "path": "dir"} - Search file contents

If no tools are needed, output: []"""
```

### Tool Detection Heuristics

MageAgent detects when tool extraction is needed using pattern matching:

```python
tool_patterns = [
    r'\bread\b.*\bfile\b', r'\bwrite\b.*\bfile\b', r'\blist\b.*\bdir',
    r'\bexecute\b', r'\brun\b', r'\bcreate\b.*\bfile\b', r'\bdelete\b',
    r'\bsearch\b', r'\bfind\b', r'\bedit\b', r'\bmodify\b',
    r'\btool\b', r'\bfunction\b.*\bcall\b', r'\bapi\b.*\bcall\b',
    r'\bglob\b', r'\bgrep\b', r'\bbash\b', r'\bshell\b'
]
```

---

## Performance Benchmarks

On M4 Max (128GB unified memory):

### Pattern Latency

| Pattern | Min | Typical | Max |
|---------|-----|---------|-----|
| tools | 2s | 3-5s | 10s |
| primary | 15s | 20-40s | 60s |
| hybrid | 20s | 30-60s | 90s |
| validated | 30s | 40-80s | 120s |
| compete | 50s | 60-120s | 180s |

### Quality Comparison

Based on internal testing and research (Together AI MoA):

| Approach | vs Single Model | vs Claude Opus |
|----------|-----------------|----------------|
| Single Qwen-72B | Baseline | ~70% |
| MageAgent Validated | +5-10% | ~80% |
| MageAgent Compete | +10-15% | ~85% |
| Claude Opus | N/A | 100% |

---

## Choosing the Right Pattern

### Decision Matrix

| Your Priority | Recommended Pattern |
|---------------|---------------------|
| Maximum quality | `compete` |
| Quality + speed | `validated` |
| Balanced | `hybrid` |
| Don't know | `auto` |
| Speed | `tools` or `primary` |
| Just reasoning | `primary` |
| Just tools | `tools` |

### Task-Based Recommendations

| Task | Pattern | Why |
|------|---------|-----|
| Write production code | `validated` | Error checking catches bugs |
| Architectural design | `compete` | Multiple perspectives |
| Code review | `hybrid` | Reasoning + suggestions |
| Quick file operations | `tools` | Fast, reliable tools |
| Explain complex code | `primary` | Pure reasoning |
| General coding | `auto` | Intelligent routing |
| Critical feature | `compete` | Maximum quality |
| Rapid prototyping | `tools` | Speed over quality |

---

## Custom Patterns (Future)

MageAgent is designed to support custom orchestration patterns. Future versions will allow:

```python
# Example custom pattern (not yet implemented)
CUSTOM_PATTERNS = {
    "review_and_fix": {
        "steps": [
            {"model": "primary", "prompt": "Review this code..."},
            {"model": "validator", "prompt": "List issues..."},
            {"model": "competitor", "prompt": "Fix these issues..."},
            {"model": "tools", "prompt": "Extract tool calls..."}
        ]
    }
}
```

---

## Troubleshooting Patterns

### Pattern Times Out

- Check available memory: `top -l 1 | grep PhysMem`
- Reduce context length in request
- Use faster pattern (`tools` or `auto`)

### Tool Calls Not Extracted

- Verify Hermes-3 Q8 model is downloaded
- Check prompt contains tool-triggering keywords
- Use `mageagent:hybrid` which always attempts tool extraction

### Quality Lower Than Expected

- Use `validated` or `compete` patterns
- Increase temperature slightly (0.7-0.8)
- Provide more context in prompt

### Model Not Loading

- Check model exists: `ls ~/.cache/mlx-models/`
- Verify memory available: need ~77GB for 72B model
- Check logs: `tail -f ~/.claude/debug/mageagent.log`

---

## API Reference

All patterns are accessible via the OpenAI-compatible API:

```bash
curl -X POST http://localhost:3457/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mageagent:PATTERN",
    "messages": [{"role": "user", "content": "Your prompt"}],
    "temperature": 0.7,
    "max_tokens": 2048
  }'
```

Replace `PATTERN` with: `auto`, `hybrid`, `validated`, `compete`, `tools`, `primary`, `validator`, `competitor`

---

*Made with care by [Adverant](https://github.com/adverant)*
