# Nexus Local Compute API Documentation

**Version:** 2.3.0
**Server URL:** `http://localhost:3457`
**OpenAI Compatible:** Yes

## Overview

Nexus Local Compute provides a local MLX-powered AI inference server with intelligent model orchestration, supporting multiple LLMs and advanced reasoning patterns.

## Quick Start

### Discovery Endpoint

Get comprehensive information about server capabilities:

```bash
curl http://localhost:3457/discover
```

### Chat Completion (OpenAI Compatible)

```bash
curl -X POST http://localhost:3457/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mageagent:hybrid",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 2048
  }'
```

## Available Models

### Direct Model Access

| Model ID | Name | Memory | Role | Speed |
|----------|------|--------|------|-------|
| `mageagent:primary` | Qwen-72B Q8 | 77GB | Primary reasoning | 8 tok/s |
| `mageagent:tools` | Hermes-3 8B Q8 | 9GB | Tool calling | 50 tok/s |
| `mageagent:validator` | Qwen-Coder 7B | 5GB | Fast validation | 105 tok/s |
| `mageagent:competitor` | Qwen-Coder 32B | 18GB | Coding specialist | 25 tok/s |
| `mageagent:glm` | GLM-4.7 Flash | 17GB | Fast MoE reasoning | 80 tok/s |

### Orchestration Patterns

| Pattern ID | Name | Description | Accuracy Boost |
|-----------|------|-------------|----------------|
| `mageagent:auto` | Auto Router | Intelligent task classification | - |
| `mageagent:execute` | Execute | ReAct loop with real tool execution | - |
| `mageagent:self_consistent` | Self-Consistent | Multiple reasoning paths | +17% |
| `mageagent:critic` | CRITIC | Self-critique for edge cases | +10% |
| `mageagent:hybrid` | Hybrid | 72B reasoning + 8B tools | - |
| `mageagent:validated` | Validated | Generate + validate loop | - |
| `mageagent:compete` | Compete | Multiple models with judge | - |
| `mageagent:fast` | Fast | Quick 7B responses | - |

## API Endpoints

### Chat & Completion

#### POST `/v1/chat/completions`
OpenAI-compatible chat completions endpoint.

**Request:**
```json
{
  "model": "mageagent:hybrid",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 2048
}
```

**Response:**
```json
{
  "id": "chatcmpl-xyz",
  "object": "chat.completion",
  "created": 1706198400,
  "model": "mageagent:hybrid",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help you today?"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 10,
    "total_tokens": 25
  }
}
```

#### GET `/v1/models`
List all available models.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "mageagent:primary",
      "object": "model",
      "created": 1706198400,
      "owned_by": "local"
    }
  ]
}
```

### Model Management

#### GET `/models/status`
Get detailed status of all models.

**Response:**
```json
{
  "models": {
    "primary": {
      "loaded": true,
      "memory_gb": 77,
      "role": "primary generator",
      "quant": "Q8_0",
      "supports_tools": true,
      "tok_per_sec": 8
    }
  },
  "loaded_models": ["primary", "tools"],
  "total_loaded_memory_gb": 86
}
```

#### POST `/models/load`
Load a specific model into GPU memory.

**Request:**
```json
{
  "model": "primary"
}
```

**Response:**
```json
{
  "status": "loaded",
  "model": "primary",
  "memory_gb": 77,
  "load_time_sec": 12.3
}
```

#### POST `/models/unload`
Unload a specific model from GPU memory.

**Request:**
```json
{
  "model": "primary"
}
```

**Response:**
```json
{
  "status": "unloaded",
  "model": "primary",
  "memory_freed_gb": 77,
  "loaded_models": ["tools"]
}
```

#### POST `/models/unload-all`
Unload ALL models from GPU memory.

**Response:**
```json
{
  "status": "all_unloaded",
  "models_unloaded": ["primary", "tools"],
  "total_memory_freed_gb": 86
}
```

### Statistics & Monitoring

#### GET `/stats`
Get inference statistics for token tracking.

**Response:**
```json
{
  "session": {
    "start": 1706198400.0,
    "duration_sec": 3600.5,
    "duration_human": "1h 0m"
  },
  "totals": {
    "requests": 42,
    "completion_tokens": 15234,
    "prompt_tokens": 8901,
    "total_tokens": 24135
  },
  "last_request": {
    "timestamp": 1706202000.0,
    "model": "primary",
    "pattern": "mageagent:hybrid",
    "tokens_per_sec": 8.5,
    "tokens_generated": 128,
    "duration_sec": 15.06
  },
  "by_model": {
    "primary": {
      "requests": 25,
      "completion_tokens": 10123,
      "prompt_tokens": 5234
    },
    "tools": {
      "requests": 17,
      "completion_tokens": 5111,
      "prompt_tokens": 3667
    }
  },
  "by_pattern": {
    "mageagent:hybrid": {
      "requests": 30,
      "tokens": 12000
    },
    "mageagent:execute": {
      "requests": 12,
      "tokens": 3234
    }
  }
}
```

#### POST `/stats/reset`
Reset all inference statistics.

**Response:**
```json
{
  "status": "stats_reset",
  "session_start": 1706202000.0
}
```

#### GET `/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "2.3.0",
  "loaded_models": ["primary", "tools"],
  "available_models": ["primary", "tools", "validator", "competitor", "glm"]
}
```

### Discovery & Integration

#### GET `/discover`
Comprehensive discovery endpoint for multi-app integration.

Returns:
- Server capabilities
- Available models and patterns
- API endpoints with descriptions
- Usage examples (curl, Python)
- Shared state file location

### RAG Integration

#### GET `/rag/status`
Get RAG integration status.

**Response:**
```json
{
  "enabled": true,
  "status": "active",
  "project_root": "/path/to/project",
  "indexed_files": 123,
  "last_indexed": 1706198400.0
}
```

#### POST `/rag/reindex`
Force re-index of project codebase.

**Response:**
```json
{
  "status": "reindexed",
  "indexed_files": 123,
  "patterns_matched": 45
}
```

## Shared State File

**Location:** `~/.claude/nexus-local-compute-state.json`

Auto-updated when models load/unload. Used for cross-app discovery.

**Contents:**
```json
{
  "server_url": "http://localhost:3457",
  "version": "2.3.0",
  "status": "running",
  "loaded_models": ["primary", "tools"],
  "available_models": ["primary", "tools", "validator", "competitor", "glm"],
  "total_loaded_memory_gb": 86,
  "updated_at": 1706198400.0,
  "api_endpoints": {
    "chat": "/v1/chat/completions",
    "models": "/v1/models",
    "discover": "/discover"
  }
}
```

## Python Integration

### Using OpenAI SDK

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:3457/v1",
    api_key="local"  # Required but unused for local server
)

response = client.chat.completions.create(
    model="mageagent:hybrid",
    messages=[
        {"role": "user", "content": "Explain quantum computing"}
    ]
)

print(response.choices[0].message.content)
```

### Direct HTTP

```python
import httpx
import json

client = httpx.Client(base_url="http://localhost:3457")

response = client.post(
    "/v1/chat/completions",
    json={
        "model": "mageagent:execute",
        "messages": [{"role": "user", "content": "List files in current directory"}],
        "max_tokens": 2048
    }
)

data = response.json()
print(data["choices"][0]["message"]["content"])
```

## Model Presets

The Mac app includes preset configurations:

- **Full Power**: All models (109GB)
- **Coding Mode**: Competitor + Tools (27GB)
- **Fast Mode**: Validator + GLM (22GB)
- **Hybrid Only**: Primary + Tools (86GB)
- **Tool Calling**: Tools only (9GB)

## Features

### Model Disable/Enable
- Persistent model disable via UserDefaults
- Disabled models cannot be loaded by any pattern
- Survives app restarts

### Token Tracking
- Per-model token counts (prompt + completion)
- Per-pattern usage statistics
- Session duration tracking
- Real-time throughput monitoring

### Usage Analytics
- Visual analytics dashboard in Mac app
- Refresh stats in real-time
- Reset statistics with confirmation

### Multi-App Discovery
- `/discover` endpoint for capability discovery
- Shared state file for cross-app communication
- OpenAI SDK compatible

## Keyboard Shortcuts

- **⌘⇧C**: Open Chat Window
- **⌘⇧A**: Open Analytics Dashboard
- **⌘W**: Load All Models
- **⌘⇧U**: Unload All Models
- **⌘Q**: Quit Application

## System Requirements

- macOS 13.0 or later
- Apple Silicon (M1/M2/M3/M4)
- 32GB+ RAM recommended (64GB+ for Full Power preset)
- MLX installed (`pip install mlx mlx-lm`)

## Notes

- Models are loaded lazily on first use
- Unloading models frees GPU memory immediately
- All endpoints are CORS-enabled
- Server runs on port 3457 by default
- RAG integration requires `MAGEAGENT_PROJECT_ROOT` environment variable

## Support

For issues and feature requests, see the project repository.

---

**Generated:** 2026-01-25
**App:** Nexus Local Compute v2.3.0
