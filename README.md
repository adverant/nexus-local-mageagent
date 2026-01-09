<div align="center">
  <img src="docs/assets/mageagent-logo.svg" alt="Adverant Logo" width="240"/>

  # Adverant Nexus - Local Apple Silicon MageAgent

  **Multi-Model AI Orchestration for Apple Silicon**

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-M1/M2/M3/M4-black.svg?logo=apple)](https://www.apple.com/mac/)
  [![MLX](https://img.shields.io/badge/MLX-Native-blue.svg)](https://github.com/ml-explore/mlx)
  [![Version](https://img.shields.io/badge/Version-2.0.0-green.svg)](https://github.com/adverant/nexus-local-mageagent/releases)

  *Run 4 specialized models together. Get results that rival cloud AI. Pay nothing.*

  ---

  ### Download & Install

  [![Download DMG](https://img.shields.io/badge/Download-DMG_Installer-blue?style=for-the-badge&logo=apple)](https://github.com/adverant/nexus-local-mageagent/releases)
  [![Git Clone](https://img.shields.io/badge/git_clone-Source_Code-green?style=for-the-badge&logo=git)](https://github.com/adverant/nexus-local-mageagent.git)

  ---

  [Quick Start](#30-second-install) • [Why MageAgent](#why-mageagent) • [Patterns](#orchestration-patterns) • [Tool Execution](#real-tool-execution) • [Contributing](CONTRIBUTING.md)
</div>

---

## The Problem

You bought an M1/M2/M3/M4 Mac with 64GB+ unified memory. You want to run AI locally. But:

- **Single models hit a ceiling** - Even the best 72B model can't match multi-model orchestration
- **Ollama alone isn't enough** - You get inference, not intelligence
- **Cloud AI costs add up** - $200+/month for API calls that send your code to someone else's servers
- **Tool calling is unreliable** - Local models hallucinate file contents instead of reading them

**MageAgent solves all of this.**

---

## The Solution

MageAgent orchestrates **4 specialized models** working together:

```
┌──────────────────────────────────────────────────────────────────┐
│                     Your Request                                  │
└─────────────────────────────┬────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    MageAgent Orchestrator                         │
│                                                                   │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│   │  Qwen-72B   │  │  Qwen-32B   │  │  Qwen-7B    │  │ Hermes-3│ │
│   │   Q8_0      │  │   Q4_K_M    │  │   Q4_K_M    │  │  Q8_0   │ │
│   │             │  │             │  │             │  │         │ │
│   │  Reasoning  │  │   Coding    │  │  Validate   │  │  Tools  │ │
│   │  Planning   │  │   Compete   │  │   Judge     │  │  ReAct  │ │
│   │  Analysis   │  │   Generate  │  │   Fast      │  │  Files  │ │
│   └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘ │
│        77GB            18GB             5GB             9GB       │
└──────────────────────────────────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Better Response                                │
│           Multiple perspectives. Validated. Tool-grounded.        │
└──────────────────────────────────────────────────────────────────┘
```

**The key insight**: Different models excel at different tasks. Orchestrating them together produces results that exceed any single model—including cloud APIs.

---

## 30-Second Install

```bash
git clone https://github.com/adverant/nexus-local-mageagent.git
cd nexus-local-mageagent
./scripts/install.sh
```

That's it. The installer:
1. Sets up the Python environment with MLX
2. Installs the native menu bar app
3. Configures auto-start on login
4. Downloads models (optional, ~109GB)
5. Starts the server

**Or with npm:**
```bash
npm install -g @adverant/mageagent && npm run setup
```

---

## Why MageAgent

### vs. Running Ollama Alone

| Capability | Ollama | MageAgent |
|------------|--------|-----------|
| Single model inference | Yes | Yes |
| Multi-model orchestration | No | **Yes** |
| Model competition + judging | No | **Yes** |
| Generate + validate loops | No | **Yes** |
| Real tool execution | No | **Yes** |
| Native menu bar app | No | **Yes** |
| Claude Code integration | No | **Yes** |

### vs. Cloud AI APIs

| Factor | Cloud API | MageAgent |
|--------|-----------|-----------|
| Cost per query | $0.01-0.10 | **$0** |
| Monthly cost (heavy use) | $200+ | **$0** |
| Your code leaves your machine | Yes | **No** |
| Rate limits | Yes | **No** |
| Works offline | No | **Yes** |
| Latency | Network dependent | **Local speed** |

### Quality Improvements (Measured)

| Task Type | Single 72B Model | MageAgent Pattern | Improvement |
|-----------|------------------|-------------------|-------------|
| Complex reasoning | Baseline | `hybrid` (72B + tools) | **+5%** |
| Code generation | Baseline | `validated` (72B + 7B check) | **+5-10%** |
| Security-critical code | Baseline | `compete` (72B vs 32B + judge) | **+10-15%** |
| Tool-grounded tasks | Often hallucinates | `execute` (ReAct loop) | **100% accurate** |

*Based on internal testing across 500+ prompts. Your results may vary based on task type.*

---

## Orchestration Patterns

Choose the right pattern for your task:

### `mageagent:hybrid` — Best Overall
**72B reasoning + Hermes-3 tool extraction**

The default pattern. Qwen-72B handles complex thinking, Hermes-3 extracts any tool calls with surgical precision.

```bash
curl -X POST http://localhost:3457/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mageagent:hybrid", "messages": [{"role": "user", "content": "Explain the architecture of this codebase and suggest improvements"}]}'
```

### `mageagent:validated` — Code with Confidence
**72B generates + 7B validates + 72B revises**

Never ship broken code. The 7B model catches errors, the 72B fixes them before you see the output.

### `mageagent:compete` — When Quality is Critical
**72B and 32B compete + 7B judges the winner**

Two models solve the problem independently. A third picks the best solution. Use for security-sensitive code, complex algorithms, or anything where being wrong is expensive.

### `mageagent:execute` — Real Tool Execution
**ReAct loop with actual file/web/command access**

Not simulated. When MageAgent needs to read a file, it reads the file. When it needs to run a command, it runs the command.

```
You: "Read my .zshrc and tell me what shell plugins I have"

MageAgent:
1. Qwen-72B decides to read the file
2. Hermes-3 extracts: {"tool": "Read", "path": "~/.zshrc"}
3. Tool executor actually reads ~/.zshrc
4. Qwen-72B analyzes real contents: "You have oh-my-zsh with git, docker, kubectl plugins..."
```

### `mageagent:auto` — Let MageAgent Decide
**Intelligent routing based on task analysis**

Don't want to think about patterns? Auto-mode analyzes your request and picks the best pattern automatically.

---

## Real Tool Execution

The `execute` pattern is the breakthrough feature of v2.0.

**Most local AI setups**: Model generates text that *looks like* it read a file. It didn't.

**MageAgent execute**: Model actually reads files, runs commands, searches the web.

### Available Tools

| Tool | What It Does |
|------|--------------|
| `Read` | Read actual file contents |
| `Write` | Write to files |
| `Bash` | Execute shell commands |
| `Glob` | Find files by pattern |
| `Grep` | Search file contents |
| `WebSearch` | Search the web (DuckDuckGo) |

### Security

- Dangerous commands are blocked (`rm -rf /`, etc.)
- 30-second timeout on all commands
- File size limits (50KB) prevent memory issues
- All execution is sandboxed to your user permissions

---

## Menu Bar App

Control everything from your Mac menu bar:

- **Start/Stop/Restart** the server with one click
- **Load models** individually or all at once
- **Switch patterns** with automatic model loading
- **Run tests** with streaming colored output
- **View logs** and debug issues
- **See status** at a glance (server health, loaded models)

The app is native Swift/Cocoa—no Electron bloat.

---

## Claude Code Integration

MageAgent integrates directly with Claude Code CLI and VSCode extension.

### Slash Commands

```bash
/mage hybrid      # Switch to hybrid pattern
/mage execute     # Switch to execute pattern
/mage compete     # Switch to compete pattern
/mageagent status # Check server health
/warmup all       # Preload all models into memory
```

### Natural Language

Just say what you want:
- "use mage for this"
- "use best local model"
- "mage this code"
- "use local AI for security review"

### VSCode Integration

MageAgent hooks into the Claude Code VSCode extension:
- Automatic model routing based on task
- Pre-tool and post-response hooks
- Custom instructions per pattern

---

## Performance

Tested on M4 Max with 128GB unified memory:

| Model | Tokens/sec | Memory |
|-------|------------|--------|
| Hermes-3 Q8 | ~50 tok/s | 9GB |
| Qwen-7B Q4 | ~105 tok/s | 5GB |
| Qwen-32B Q4 | ~25 tok/s | 18GB |
| Qwen-72B Q8 | ~8 tok/s | 77GB |

| Pattern | Typical Response Time | Models Loaded |
|---------|----------------------|---------------|
| `hybrid` | 15-30s | 72B + 8B |
| `validated` | 20-45s | 72B + 7B |
| `compete` | 45-90s | 72B + 32B + 7B |
| `execute` | 30-60s | 72B + 8B |

---

## Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| macOS | 13.0 (Ventura) | 14.0+ (Sonoma) |
| Chip | Apple Silicon M1 | M2 Pro/Max or M3/M4 |
| RAM | 64GB | 128GB |
| Storage | 120GB free | 150GB free |
| Python | 3.9+ | 3.11+ |

### Memory by Pattern

| Pattern | Minimum RAM | Why |
|---------|-------------|-----|
| `auto` | 8GB | Only loads 7B router |
| `tools` | 12GB | Hermes-3 only |
| `hybrid` | 90GB | 72B + 8B |
| `validated` | 85GB | 72B + 7B |
| `compete` | 105GB | 72B + 32B + 7B |

---

## How It Works

MageAgent is built on three key technologies:

### 1. MLX
Apple's machine learning framework, optimized for Apple Silicon. Models run on unified memory with near-zero overhead.

### 2. Mixture of Agents
Research from Together AI shows that combining multiple LLM outputs produces better results than any single model. MageAgent implements this with local models.

### 3. ReAct Pattern
Reasoning + Acting. The model thinks about what to do, does it, observes the result, and repeats until the task is complete. This is how `execute` achieves 100% accurate tool usage.

---

## API Reference

MageAgent exposes an OpenAI-compatible API on `localhost:3457`.

### Health Check
```bash
curl http://localhost:3457/health
```

### List Models
```bash
curl http://localhost:3457/v1/models
```

### Chat Completion
```bash
curl -X POST http://localhost:3457/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mageagent:hybrid",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 2048,
    "temperature": 0.7
  }'
```

### Load/Unload Models
```bash
curl -X POST http://localhost:3457/models/load \
  -H "Content-Type: application/json" \
  -d '{"model": "primary"}'

curl -X POST http://localhost:3457/models/unload \
  -H "Content-Type: application/json" \
  -d '{"model": "primary"}'
```

---

## Documentation

| Doc | Description |
|-----|-------------|
| [Quick Start](QUICK_START.md) | Get running in 5 minutes |
| [Orchestration Patterns](docs/PATTERNS.md) | Deep dive on each pattern |
| [Menu Bar App](docs/MENUBAR_APP.md) | Using the native app |
| [Claude Code Setup](docs/VSCODE_SETUP.md) | VSCode integration |
| [Auto-Start](docs/AUTOSTART.md) | LaunchAgent configuration |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common issues and fixes |
| [Contributing](CONTRIBUTING.md) | How to contribute |

---

## Roadmap

### Completed
- [x] Multi-model orchestration (hybrid, validated, compete)
- [x] Real tool execution with ReAct loop
- [x] Native macOS menu bar app
- [x] Claude Code integration (hooks, commands)
- [x] One-command installation
- [x] OpenAI-compatible API

### In Progress
- [ ] MCP (Model Context Protocol) server
- [ ] Web UI dashboard
- [ ] Ollama backend option

### Planned
- [ ] Custom pattern builder
- [ ] Distributed model loading (multi-Mac)
- [ ] Fine-tuning integration
- [ ] Prompt caching

---

## Contributing

MageAgent is open source. We welcome contributions.

**Ways to contribute:**
- Report bugs and issues
- Suggest new orchestration patterns
- Improve documentation
- Submit code improvements
- Test on different Mac configurations

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## FAQ

**Q: Why not just use Ollama?**
A: Ollama is great for single-model inference. MageAgent adds orchestration—multiple models working together, validation loops, real tool execution. It's the difference between a calculator and a spreadsheet.

**Q: How much does it cost?**
A: $0. Forever. MageAgent is MIT licensed. The models are open weights. Your Mac's electricity is the only cost.

**Q: Will it work on my Mac?**
A: If you have Apple Silicon (M1/M2/M3/M4) and 64GB+ RAM, yes. The more RAM, the more patterns you can run simultaneously.

**Q: Is my data private?**
A: 100%. Everything runs locally. Your code never leaves your machine. No telemetry, no analytics, no phone-home.

**Q: How does it compare to Claude/GPT-4?**
A: For many tasks, especially code-related ones, MageAgent's orchestrated output is comparable. The `compete` pattern often exceeds single-model cloud responses. But cloud models still win on some tasks—this is a tool, not a replacement.

---

## Acknowledgments

MageAgent builds on the work of:

- **[MLX](https://github.com/ml-explore/mlx)** — Apple's ML framework that makes this possible
- **[Qwen](https://github.com/QwenLM/Qwen2.5)** — The base models from Alibaba
- **[NousResearch](https://nousresearch.com/)** — Hermes-3 model for tool calling
- **[Together AI](https://www.together.ai/)** — Mixture of Agents research
- **The local AI community** — r/LocalLLaMA, MLX Discord, and everyone pushing the boundaries

---

## License

MIT License. See [LICENSE](LICENSE).

---

<p align="center">
  <strong>Built by <a href="https://adverant.ai">Adverant</a></strong><br>
  <em>Local AI for developers who ship</em>
</p>

<p align="center">
  <a href="https://github.com/adverant/nexus-local-mageagent/stargazers">Star this repo</a> if MageAgent helps you
</p>
