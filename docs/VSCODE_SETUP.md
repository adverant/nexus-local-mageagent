# VSCode Extension Setup Guide

This guide explains how to configure the Claude Code VSCode extension to work with MageAgent for multi-model orchestration.

## Overview

MageAgent runs as a local server on port 3457, providing an OpenAI-compatible API. The Claude Code VSCode extension can be configured to use MageAgent through the Claude Code Router.

## Prerequisites

1. MageAgent server running (`~/.claude/scripts/mageagent-server.sh start`)
2. Claude Code Router installed (`npm install -g @musistudio/claude-code-router`)
3. Claude Code VSCode extension installed

## Configuration Steps

### Step 1: Configure Router for MageAgent

Create or update `~/.claude-code-router/config.json`:

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
    },
    {
      "name": "anthropic",
      "api_base_url": "https://api.anthropic.com",
      "api_key": "${ANTHROPIC_API_KEY}",
      "models": [
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-5-20250929"
      ]
    }
  ],
  "Router": {
    "default": "mageagent,mageagent:hybrid"
  }
}
```

### Step 2: Configure Claude Settings

Add to `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:3456"
  }
}
```

This tells Claude Code to route all API calls through your local router.

### Step 3: Set System Environment Variable

Add to your `~/.zshrc`:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:3456
```

Then reload:
```bash
source ~/.zshrc
```

### Step 4: Restart VSCode Completely

**Important**: You must completely quit and restart VSCode (not just reload window) for environment variables to take effect.

```bash
# Quit VSCode completely, then relaunch
code .
```

### Step 5: Verify Services Running

Before using VSCode, ensure both services are running:

```bash
# Check MageAgent server
curl http://localhost:3457/health

# Check Claude Code Router
ccr status
```

## Using MageAgent in VSCode

### Switching Models

Use the `/model` command in any Claude Code session:

```bash
# Use MageAgent hybrid (recommended)
/model mageagent,mageagent:hybrid

# Use MageAgent validated for code quality
/model mageagent,mageagent:validated

# Use MageAgent compete for critical code
/model mageagent,mageagent:compete

# Switch back to Claude Opus
/model anthropic,claude-opus-4-5-20251101
```

### Natural Language Switching

Just say:
- "use mageagent" → Switches to hybrid pattern
- "mage validated" → Switches to validated pattern
- "use claude" → Switches back to Claude

### Pattern Selection Tips

| Pattern | When to Use |
|---------|-------------|
| `hybrid` | Default for most tasks |
| `validated` | Production code, need error checking |
| `compete` | Critical features, want best quality |
| `tools` | Quick file operations |
| `auto` | Let MageAgent decide |

## Troubleshooting

### Extension Not Using Router

1. **Check router is running**:
   ```bash
   ccr status
   ```

2. **Verify environment variable**:
   ```bash
   echo $ANTHROPIC_BASE_URL
   # Should output: http://127.0.0.1:3456
   ```

3. **Restart VSCode completely** - not just reload window

### MageAgent Server Not Responding

1. **Check server status**:
   ```bash
   ~/.claude/scripts/mageagent-server.sh status
   ```

2. **View logs**:
   ```bash
   tail -f ~/.claude/debug/mageagent.log
   ```

3. **Restart server**:
   ```bash
   ~/.claude/scripts/mageagent-server.sh stop
   ~/.claude/scripts/mageagent-server.sh start
   ```

### Model Loading Slow

Large models (72B) take 5-10 seconds to load initially. Subsequent requests are faster as models are cached in memory.

Monitor memory usage:
```bash
top -l 1 -o mem | head -10
```

### Out of Memory

If you see memory errors:
1. Close other applications
2. Use smaller patterns (`tools`, `validator`)
3. Restart MageAgent to clear cached models

## Performance Expectations

| Pattern | First Request | Subsequent |
|---------|---------------|------------|
| tools | 5-10s | 3-5s |
| hybrid | 30-45s | 25-35s |
| validated | 45-60s | 35-50s |
| compete | 60-90s | 50-70s |

## Direct API Access

You can also use MageAgent directly without the router:

```bash
curl -X POST http://localhost:3457/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mageagent:hybrid",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Memory Requirements

For full MageAgent capability, you need:

| Pattern | Peak Memory |
|---------|-------------|
| tools only | ~9 GB |
| validator only | ~5 GB |
| primary only | ~77 GB |
| hybrid | ~86 GB |
| validated | ~86 GB |
| compete | ~100 GB |

Recommended: 128GB unified memory for best experience.

---

*Made with care by [Adverant](https://github.com/adverant)*
