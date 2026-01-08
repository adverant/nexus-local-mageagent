# Nexus Local MageAgent - Quick Start

Get MageAgent running in 5 minutes on your Apple Silicon Mac.

## Prerequisites

- Apple Silicon Mac (M1/M2/M3/M4) with 128GB+ unified memory
- Python 3.9+
- pip3

## Step 1: Install Dependencies

```bash
# Install MLX
pip3 install mlx mlx-lm

# Install FastAPI for the server
pip3 install fastapi uvicorn pydantic
```

## Step 2: Download Models

```bash
# Create models directory
mkdir -p ~/.cache/mlx-models
cd ~/.cache/mlx-models

# Download models from Hugging Face (this takes time)
python3 -c "
from huggingface_hub import snapshot_download

# Hermes-3 Q8 - Tool calling specialist (9GB)
snapshot_download('mlx-community/Hermes-3-Llama-3.1-8B-8bit', local_dir='Hermes-3-Llama-3.1-8B-8bit')

# Qwen-72B Q8 - Primary reasoning (77GB)
snapshot_download('mlx-community/Qwen2.5-72B-Instruct-8bit', local_dir='Qwen2.5-72B-Instruct-8bit')

# Qwen-32B Q4 - Coding specialist (18GB)
snapshot_download('mlx-community/Qwen2.5-Coder-32B-Instruct-4bit', local_dir='Qwen2.5-Coder-32B-Instruct-4bit')

# Qwen-7B Q4 - Fast validator (5GB)
snapshot_download('mlx-community/Qwen2.5-Coder-7B-Instruct-4bit', local_dir='Qwen2.5-Coder-7B-Instruct-4bit')
"
```

## Step 3: Install MageAgent Server

```bash
# Clone the repo
git clone https://github.com/adverant/nexus-local-mageagent.git
cd nexus-local-mageagent

# Copy server files
mkdir -p ~/.claude/mageagent
cp mageagent/server.py ~/.claude/mageagent/

mkdir -p ~/.claude/scripts
cp scripts/mageagent-server.sh ~/.claude/scripts/
chmod +x ~/.claude/scripts/mageagent-server.sh

# Create debug directory
mkdir -p ~/.claude/debug
```

## Step 4: Start the Server

```bash
~/.claude/scripts/mageagent-server.sh start
```

You should see:
```
MageAgent started (PID: xxxxx)
Server ready!
API: http://localhost:3457
```

## Step 5: Test It

```bash
# Quick test
curl -X POST http://localhost:3457/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mageagent:tools",
    "messages": [{"role": "user", "content": "Hello, what can you do?"}],
    "max_tokens": 100
  }'
```

## Step 6: Configure Claude Code Router (Optional)

Add MageAgent to your Claude Code router config:

```bash
# Edit config
nano ~/.claude-code-router/config.json
```

Add the mageagent provider:
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
        "mageagent:primary"
      ]
    }
  ]
}
```

## Usage in Claude Code

```bash
# Switch to MageAgent hybrid (best capability)
/model mageagent,mageagent:hybrid

# Or use natural language
"use mage hybrid"
"use best local"
"mage validated"
```

## Available Patterns

| Command | Description |
|---------|-------------|
| `mageagent:hybrid` | 72B reasoning + Hermes tools (recommended) |
| `mageagent:validated` | 72B + 7B validation + Hermes tools |
| `mageagent:compete` | 72B vs 32B + judge + Hermes tools |
| `mageagent:auto` | Automatic task routing |
| `mageagent:tools` | Fast Hermes-3 tool calling |
| `mageagent:primary` | Direct 72B reasoning |

## Troubleshooting

### Server won't start
```bash
# Check logs
tail -50 ~/.claude/debug/mageagent.log

# Check if port is in use
lsof -i :3457
```

### Model not found
```bash
# Verify models are downloaded
ls -la ~/.cache/mlx-models/
```

### Out of memory
- Close other applications
- Use smaller patterns (tools, validator)
- Consider 64GB or 128GB Mac

## Next Steps

- Read the [full README](README.md) for architecture details
- See [docs/PATTERNS.md](docs/PATTERNS.md) for pattern deep-dive
- Configure [auto-start on boot](docs/AUTOSTART.md)
