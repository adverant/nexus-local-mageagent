# Nexus Local MageAgent

> **Multi-Model AI Orchestration for Apple Silicon - Intelligent task routing with MLX models**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-M4_Max-success.svg)](https://www.apple.com/mac/)
[![MLX](https://img.shields.io/badge/MLX-Native-blue.svg)](https://github.com/ml-explore/mlx)

## Overview

Nexus Local MageAgent provides **intelligent multi-model orchestration** using Apple's MLX framework for native Apple Silicon performance. Instead of manual model switching, MageAgent automatically routes tasks to the optimal model combination:

- **Qwen-72B Q8** - Superior reasoning, planning, complex analysis (77GB)
- **Qwen-32B Q4** - Code generation specialist (18GB)
- **Qwen-7B Q4** - Fast validation and judging (5GB)
- **Hermes-3 Q8** - Tool calling specialist (9GB)

### Key Innovation: Hermes-3 Tool Routing

All MageAgent patterns route tool extraction through **Hermes-3 Q8**, ensuring reliable function calling regardless of which reasoning model generates the response. Q8 quantization preserves tool-calling capability that Q4 breaks.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MageAgent Orchestrator                       │
│                        (Port 3457)                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ mageagent:   │    │ mageagent:   │    │ mageagent:   │       │
│  │   hybrid     │    │  validated   │    │   compete    │       │
│  │              │    │              │    │              │       │
│  │ 72B + Hermes │    │ 72B + 7B +   │    │ 72B + 32B +  │       │
│  │              │    │   Hermes     │    │ 7B + Hermes  │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                   │                   │               │
│         └───────────────────┴───────────────────┘               │
│                             │                                    │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   MLX Model Pool                          │   │
│  │                                                           │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────┐ │   │
│  │  │ Qwen-72B   │ │ Qwen-32B   │ │ Qwen-7B    │ │Hermes-3│ │   │
│  │  │ Q8 (77GB)  │ │ Q4 (18GB)  │ │ Q4 (5GB)   │ │Q8 (9GB)│ │   │
│  │  │ Reasoning  │ │ Coding     │ │ Validation │ │ Tools  │ │   │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Orchestration Patterns

| Pattern | Models | Flow | Use Case |
|---------|--------|------|----------|
| `mageagent:hybrid` | 72B Q8 + Hermes Q8 | Reasoning → Tool Extraction | Best capability |
| `mageagent:validated` | 72B Q8 + 7B + Hermes Q8 | Generate → Validate → Fix → Tools | Code with error checking |
| `mageagent:compete` | 72B + 32B + 7B + Hermes Q8 | Two solutions → Judge → Best + Tools | Critical decisions |
| `mageagent:auto` | Dynamic | Routes based on task type | Automatic optimization |
| `mageagent:tools` | Hermes-3 Q8 | Direct tool calling | Fast function execution |
| `mageagent:primary` | Qwen-72B Q8 | Direct reasoning | Complex analysis |

### Pattern Details

#### Hybrid (Recommended)
```
User Request → Qwen-72B Q8 (reasoning) → Hermes-3 Q8 (tool extraction) → Response + Tool Calls
```
Best for: Tasks requiring both reasoning AND tool execution

#### Validated
```
User Request → Qwen-72B Q8 → Qwen-7B (validate) → [Fix if needed] → Hermes-3 Q8 (tools) → Response
```
Best for: Code generation where correctness matters (+5-10% quality)

#### Compete
```
User Request → Qwen-72B Q8 ──┐
                             ├→ Qwen-7B (judge) → Winner → Hermes-3 Q8 (tools) → Response
User Request → Qwen-32B Q4 ──┘
```
Best for: Critical code where multiple perspectives improve quality (+10-15% quality)

## Quick Start

### Prerequisites
- Apple Silicon Mac (M1/M2/M3/M4)
- 128GB+ Unified Memory recommended
- Python 3.9+
- MLX framework

### Installation

```bash
# Clone the repository
git clone https://github.com/adverant/nexus-local-mageagent.git
cd nexus-local-mageagent

# Run setup
./scripts/install.sh

# Start MageAgent server
~/.claude/scripts/mageagent-server.sh start
```

### Model Downloads

The installer will download MLX models (~110GB total):

| Model | Size | Purpose |
|-------|------|---------|
| Qwen2.5-72B-Instruct-8bit | 77GB | Primary reasoning (Q8 for tools) |
| Qwen2.5-Coder-32B-4bit | 18GB | Code generation |
| Qwen2.5-Coder-7B-4bit | 5GB | Fast validation |
| Hermes-3-Llama-3.1-8B-8bit | 9GB | Tool calling specialist |

### Usage

#### Via Claude Code
```bash
# Switch to MageAgent
/model mageagent,mageagent:hybrid

# Natural language switching
"use mage hybrid"
"use best local"
```

#### API Endpoint
```bash
curl -X POST http://localhost:3457/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mageagent:hybrid",
    "messages": [{"role": "user", "content": "Read and refactor /path/to/file.py"}]
  }'
```

## Model Quantization & Tool Calling

**Critical Insight**: Q8 quantization preserves tool-calling capability, Q4 breaks it.

| Model | Quantization | Tool Calling | Purpose |
|-------|--------------|--------------|---------|
| Hermes-3-8B | Q8_0 | Yes | Tool calling specialist |
| Qwen-72B | Q8_0 | Yes | Reasoning with tools |
| Qwen-32B | Q4_K_M | No | Code generation only |
| Qwen-7B | Q4_K_M | No | Validation only |

This is why MageAgent routes ALL tool extraction through Hermes-3 Q8 - it guarantees reliable function calling regardless of which model generated the reasoning.

## Performance

On M4 Max (128GB):

| Model | Tokens/sec | Load Time |
|-------|------------|-----------|
| Hermes-3 Q8 | ~50 tok/s | 1.5s |
| Qwen-7B Q4 | ~105 tok/s | 0.8s |
| Qwen-32B Q4 | ~25 tok/s | 3s |
| Qwen-72B Q8 | ~8 tok/s | 8s |

Pattern completion times (typical):
- `mageagent:tools`: 3-5s
- `mageagent:hybrid`: 30-60s
- `mageagent:validated`: 40-80s
- `mageagent:compete`: 60-120s

## Configuration

### Router Config (`~/.claude-code-router/config.json`)
```json
{
  "providers": [
    {
      "name": "mageagent",
      "api_base_url": "http://localhost:3457/v1/chat/completions",
      "api_key": "local",
      "models": [
        "mageagent:auto",
        "mageagent:hybrid",
        "mageagent:validated",
        "mageagent:compete",
        "mageagent:tools",
        "mageagent:primary",
        "mageagent:validator",
        "mageagent:competitor"
      ]
    }
  ]
}
```

### Server Management
```bash
# Start server
~/.claude/scripts/mageagent-server.sh start

# Stop server
~/.claude/scripts/mageagent-server.sh stop

# Check status
~/.claude/scripts/mageagent-server.sh status

# View logs
~/.claude/scripts/mageagent-server.sh logs
```

## Integration with Nexus

MageAgent is part of the Nexus AI platform:

- **Nexus GraphRAG** - Long-term memory across sessions
- **Nexus Skills Engine** - Custom workflow automation
- **Nexus Dashboard** - Monitoring and analytics
- **Nexus Local MageAgent** - Multi-model orchestration

## Comparison: MageAgent vs Single Model

| Aspect | Single Model | MageAgent |
|--------|--------------|-----------|
| Quality | Baseline | +5-15% (validated/compete) |
| Tool Calling | Unreliable at Q4 | Guaranteed via Hermes Q8 |
| Cost | Cloud: $$$/Local: Free | Local: Free |
| Privacy | Cloud: No/Local: Yes | Full local privacy |
| Latency | Fast | Slower (orchestration) |
| Flexibility | One model | Dynamic routing |

## Roadmap

- [ ] Streaming responses
- [ ] Parallel model execution (when Metal supports it)
- [ ] Custom orchestration patterns
- [ ] MCP tool server integration
- [ ] Web UI for monitoring
- [ ] Ollama backend option

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE)

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
- [Qwen](https://github.com/QwenLM/Qwen2.5) - Base models
- [NousResearch](https://nousresearch.com/) - Hermes-3 model
- Together AI research on Mixture of Agents

---

**Made with care by [Adverant](https://github.com/adverant)**

*Local AI orchestration for serious developers*
