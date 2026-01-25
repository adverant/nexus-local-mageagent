<div align="center">

# Nexus Local Compute

**Multi-Model AI Orchestration for Apple Silicon**

[![Platform](https://img.shields.io/badge/platform-macOS%2013%2B-blue)](https://www.apple.com/macos/)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-M1/M2/M3/M4-black.svg?logo=apple)](https://www.apple.com/mac/)
[![MLX](https://img.shields.io/badge/MLX-Native-blue.svg)](https://github.com/ml-explore/mlx)
[![Version](https://img.shields.io/badge/Version-2.3.0-green.svg)](https://github.com/adverant/Adverant-Nexus-Local-Mageagent)

*Run 5 specialized models together. Get GPT-4 class results. Pay nothing. Own your data.*

---

[Quick Start](#quick-start) • [Features](#features) • [Patterns](#orchestration-patterns) • [API Docs](API-DOCUMENTATION.md) • [Install](#installation)

</div>

---

## Why Nexus Local Compute?

You bought an M1/M2/M3/M4 Mac with 64GB+ unified memory. You want true AI capabilities locally. But:

- **Single models plateau** - Even 72B models benefit from orchestration
- **Cloud APIs cost money** - $200+/month sending your code to someone's servers
- **Tool calling is unreliable** - Models hallucinate instead of executing
- **Privacy matters** - Your code shouldn't leave your machine

**Nexus Local Compute solves all of this.**

---

## The Solution

Nexus orchestrates **5 specialized models** with intelligent routing:

```
┌────────────────────────────────────────────────────────────────┐
│                         Your Request                            │
└──────────────────────────┬─────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  Auto Router │ ← Classifies task type
                    └──────┬──────┘
             ┌─────────────┼─────────────┐
             │             │             │
      ┌──────▼─────┐ ┌────▼────┐ ┌─────▼──────┐
      │  Qwen-72B  │ │Hermes-3 │ │  GLM-4.7   │
      │  Reasoning │ │  Tools  │ │  Fast MoE  │
      │   8 tok/s  │ │ 50 tok/s│ │  80 tok/s  │
      └──────┬─────┘ └────┬────┘ └─────┬──────┘
             │            │            │
             │   ┌────────▼────────┐   │
             └───►   Validator     ◄───┘
                 │   7B Fast       │
                 │   105 tok/s     │
                 └────────┬────────┘
                          │
                    ┌─────▼─────┐
                    │  Response  │
                    └────────────┘
```

**Result:** +17% accuracy, real tool execution, zero hallucinations on file operations.

---

## 🚀 Features

### Multi-Model Orchestration
✅ **5 Local Models** - Qwen-72B, Hermes-3 8B, Qwen-Coder 7B/32B, GLM-4.7 Flash
✅ **8 Reasoning Patterns** - Auto, Execute, Self-Consistent (+17%), CRITIC (+10%), Hybrid, Validated, Compete, Fast
✅ **Real Tool Execution** - Actually reads files, runs bash commands, searches code
✅ **OpenAI Compatible** - Drop-in replacement for OpenAI SDK

### Intelligent Management
✅ **Load/Unload Models** - Free GPU memory on demand (⌘⇧U)
✅ **Model Presets** - Quick-switch configurations (Full Power, Coding Mode, Fast Mode, etc.)
✅ **Persistent Disable** - Block models from loading permanently
✅ **Visual Status** - ✓ loaded, ○ available, ⊘ disabled

### Usage Analytics
✅ **Token Tracking** - Per-model and per-pattern statistics
✅ **Session Monitoring** - Duration and throughput
✅ **Visual Dashboard** - Real-time analytics (⌘⇧A)
✅ **Cost Analysis** - Track prompt and completion tokens

### Developer Tools
✅ **Chat Interface** - Built-in chat window (⌘⇧C)
✅ **File Attachments** - Drag & drop PDFs, code, text
✅ **Multi-App Integration** - VSCode, Claude Code CLI, Nexus CLI
✅ **Comprehensive API** - 15+ REST endpoints
✅ **Full Documentation** - [API docs included](API-DOCUMENTATION.md)

---

## Quick Start

### 1. Install Dependencies
```bash
# Install MLX
pip install mlx mlx-lm

# Download models (first use only, ~120GB total)
# Models download automatically on first use
```

### 2. Build & Install
```bash
cd menubar-app
chmod +x build.sh
./build.sh
```

App installs to `/Applications/NexusLocalCompute.app`

### 3. Launch
```bash
open /Applications/NexusLocalCompute.app
```

Menu bar icon appears automatically.

### 4. Load Models
Click icon → **Models → Presets → Hybrid Only** (recommended first use: 86GB)

### 5. Start Chatting
Press **⌘⇧C** or select **💬 Chat with Local AI**

---

## Installation

### System Requirements
- **macOS** 13.0 or later
- **Apple Silicon** M1/M2/M3/M4
- **RAM** 32GB minimum (64GB+ recommended)
- **Storage** 120GB free for all models
- **Python** 3.10+ with MLX

### Download Models

Models are downloaded automatically on first use. To download manually:

```bash
# Qwen-72B (77GB) - Primary reasoning
mlx_lm.convert --hf-path Qwen/Qwen2.5-72B-Instruct \
  --mlx-path ~/.cache/mlx-models/Qwen2.5-72B-Instruct-8bit --quantize

# Hermes-3 8B (9GB) - Tool calling
mlx_lm.convert --hf-path NousResearch/Hermes-3-Llama-3.1-8B \
  --mlx-path ~/.cache/mlx-models/Hermes-3-Llama-3.1-8B-8bit --quantize

# Qwen-Coder 7B (5GB) - Fast validation
mlx_lm.convert --hf-path Qwen/Qwen2.5-Coder-7B-Instruct \
  --mlx-path ~/.cache/mlx-models/Qwen2.5-Coder-7B-Instruct-4bit --q-bits 4

# Qwen-Coder 32B (18GB) - Coding
mlx_lm.convert --hf-path Qwen/Qwen2.5-Coder-32B-Instruct \
  --mlx-path ~/.cache/mlx-models/Qwen2.5-Coder-32B-Instruct-4bit --q-bits 4

# GLM-4.7 Flash (17GB) - Fast MoE
mlx_lm.convert --hf-path THUDM/glm-4-9b-chat \
  --mlx-path ~/.cache/mlx-models/GLM-4.7-Flash-4bit --q-bits 4
```

---

## Available Models

| Model | Size | Speed | Role | Best For |
|-------|------|-------|------|----------|
| **Qwen-72B** | 77GB | 8 tok/s | Primary reasoning | Complex analysis, planning, difficult questions |
| **Hermes-3 8B** | 9GB | 50 tok/s | Tool calling | Function execution, structured output, JSON |
| **Qwen-Coder 7B** | 5GB | 105 tok/s | Fast validation | Quick checks, judging, validation |
| **Qwen-Coder 32B** | 18GB | 25 tok/s | Coding specialist | Code generation, refactoring, debugging |
| **GLM-4.7 Flash** | 17GB | 80 tok/s | Fast MoE | Quick reasoning, simple tasks, speed |

---

## Orchestration Patterns

| Pattern | Models Used | Benefit | Use Case |
|---------|-------------|---------|----------|
| **auto** | Auto-selected | Intelligent routing | General purpose, unknown tasks |
| **execute** | Tools + Primary | Real tool execution | File operations, bash commands, reads/writes |
| **self_consistent** | Primary + Validator | **+17% accuracy** | Important decisions, critical tasks |
| **critic** | Primary + Validator | **+10% edge cases** | Quality assurance, edge case detection |
| **hybrid** | Primary + Tools | Best of both worlds | Complex tasks requiring reasoning + execution |
| **validated** | Primary + Validator | Correction loop | High accuracy requirements |
| **compete** | Multiple models | Judge selects best | Critical outputs, multiple perspectives |
| **fast** | Validator only | Speed optimized | Quick responses, simple questions |

---

## Usage

### Keyboard Shortcuts
- **⌘⇧C** - Open Chat Window
- **⌘⇧A** - Open Analytics Dashboard
- **⌘W** - Load All Models
- **⌘⇧U** - Unload All Models
- **⌘Q** - Quit Application

### Python API (OpenAI Compatible)
```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:3457/v1",
    api_key="local"
)

response = client.chat.completions.create(
    model="mageagent:hybrid",
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)

print(response.choices[0].message.content)
```

### cURL
```bash
curl -X POST http://localhost:3457/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mageagent:execute",
    "messages": [{"role": "user", "content": "List files in current directory"}],
    "max_tokens": 2048
  }'
```

### Discovery API
```bash
# Get all capabilities
curl http://localhost:3457/discover | jq

# Check model status
curl http://localhost:3457/models/status | jq

# View usage statistics
curl http://localhost:3457/stats | jq
```

---

## Model Presets

Built-in configurations for common scenarios:

### Full Power (109GB)
All 5 models loaded. Maximum capability.
- **Use for:** Research, complex analysis, critical tasks
- **Memory:** 109GB
- **Speed:** Variable (8-105 tok/s depending on task)

### Coding Mode (27GB)
Qwen-Coder 32B + Hermes-3 8B
- **Use for:** Software development, code review, refactoring
- **Memory:** 27GB
- **Speed:** 25-50 tok/s

### Fast Mode (22GB)
Qwen-Coder 7B + GLM-4.7 Flash
- **Use for:** Quick questions, simple tasks, speed priority
- **Memory:** 22GB
- **Speed:** 80-105 tok/s

### Hybrid Only (86GB)
Qwen-72B + Hermes-3 8B (recommended)
- **Use for:** General purpose, best balance
- **Memory:** 86GB
- **Speed:** 8-50 tok/s

### Tool Calling (9GB)
Hermes-3 8B only
- **Use for:** Pure tool execution, minimal memory
- **Memory:** 9GB
- **Speed:** 50 tok/s

**Custom Presets:** Save your own configurations via the menu.

---

## Real Tool Execution

Unlike other local AI solutions, Nexus **actually executes tools**:

```python
# ❌ Other solutions: Hallucinate file contents
"Here's what I think might be in that file..."

# ✅ Nexus: Actually reads the file
Read tool → Returns actual contents → Accurate response
```

**Available Tools:**
- **Read** - Read files (code, PDFs, text, Jupyter notebooks)
- **Write** - Create/modify files
- **Edit** - Precise string replacements
- **Bash** - Execute shell commands
- **Glob** - Find files by pattern
- **Grep** - Search file contents
- **WebFetch** - Fetch and analyze web pages
- **WebSearch** - Search the web

**Execute Pattern Example:**
```
User: "Read server.py and explain the authentication flow"

Nexus:
1. Read tool → Actually reads server.py
2. Analyzes actual code (not hallucinated)
3. Explains authentication flow with line numbers
4. Can make changes if requested
```

---

## Multi-App Integration

Nexus provides a discovery API for seamless integration:

### Shared State File
`~/.claude/nexus-local-compute-state.json`

Auto-updated with:
- Server URL and version
- Loaded models
- Available patterns
- API endpoints
- Timestamp

### Discovery Endpoint
```bash
curl http://localhost:3457/discover
```

Returns comprehensive capabilities for dynamic integration.

### Supported Apps
- **VSCode Extensions** - Use local models in editor
- **Claude Code CLI** - Terminal-based coding assistant
- **Nexus CLI** - Command-line interface
- **Custom Applications** - OpenAI SDK compatible

---

## Analytics Dashboard

Press **⌘⇧A** to view real-time usage statistics:

- **Session Overview** - Duration and uptime
- **Total Usage** - Requests and tokens
- **Last Request** - Model, pattern, speed, duration
- **Usage by Model** - Per-model breakdown
- **Usage by Pattern** - Per-pattern statistics

**Reset Stats:** Clear all statistics with confirmation dialog.

---

## API Reference

See **[API-DOCUMENTATION.md](API-DOCUMENTATION.md)** for complete documentation including:

- All 15+ REST endpoints
- Request/response examples
- Python integration guide
- Model specifications
- Pattern descriptions
- Troubleshooting

---

## Troubleshooting

### Server Not Starting
```bash
# Check if port 3457 is in use
lsof -i :3457

# Start server manually
cd ~/.claude/mageagent
python3 server.py
```

### GPU Out of Memory
- Unload models: **⌘⇧U**
- Use smaller preset: **Fast Mode** or **Tool Calling**
- Close other GPU apps
- Restart application

### Models Won't Load
- Verify MLX installed: `pip show mlx mlx-lm`
- Check model paths: `ls ~/.cache/mlx-models/`
- Ensure sufficient RAM
- Check console logs

---

## Project Structure

```
Nexus-Local-Compute/
├── mageagent/                 # Python server
│   ├── server.py              # FastAPI server (77KB)
│   ├── codebase_indexer.py    # RAG indexing
│   ├── pattern_matcher.py     # Pattern matching
│   ├── rag_integration.py     # RAG integration
│   └── tool_executor.py       # Real tool execution
├── menubar-app/               # macOS app
│   ├── MageAgentMenuBar/
│   │   ├── AppDelegate.swift  # Main app (5400+ lines)
│   │   ├── Info.plist
│   │   └── main.swift
│   ├── Package.swift
│   └── build.sh
├── API-DOCUMENTATION.md       # Complete API reference
└── README.md                  # This file
```

---

## Credits

**Built with:**
- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- Cocoa - macOS native UI

**Powered by open-source models:**
- Qwen by Alibaba
- Hermes by Nous Research
- GLM by Tsinghua University

---

## License

Private - Adverant AI

---

<div align="center">

**Nexus Local Compute v2.3.0**

*True AI. Your Mac. Your Data. Zero Cloud.*

[Documentation](API-DOCUMENTATION.md) • [Issues](https://github.com/adverant/Adverant-Nexus-Local-Mageagent/issues) • [Adverant AI](https://adverant.ai)

© 2026 Adverant AI

</div>
