#!/bin/bash

# Nexus Local MageAgent - Automated Installation Script
# Installs and configures MageAgent multi-model orchestration for MLX

set -e  # Exit on error

echo "==============================================================="
echo "  Nexus Local MageAgent - Installation"
echo "  Multi-Model AI Orchestration for Apple Silicon"
echo "==============================================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}Error: This script requires macOS with Apple Silicon${NC}"
    exit 1
fi

# Check for Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo -e "${RED}Error: Apple Silicon (M1/M2/M3/M4) required${NC}"
    exit 1
fi

# Check memory
TOTAL_MEM=$(sysctl -n hw.memsize)
TOTAL_GB=$((TOTAL_MEM / 1024 / 1024 / 1024))
if [ $TOTAL_GB -lt 64 ]; then
    echo -e "${YELLOW}Warning: 128GB+ unified memory recommended for full MageAgent${NC}"
    echo -e "  Your system has ${TOTAL_GB}GB. Some patterns may not work.${NC}"
    echo ""
fi

echo "Checking Prerequisites..."
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    echo "  Install from: https://www.python.org/"
    exit 1
fi
echo -e "${GREEN}OK${NC} Python $(python3 --version | cut -d' ' -f2)"

# Check pip
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}Error: pip3 is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}OK${NC} pip3 installed"

echo ""
echo "==============================================================="
echo "  Installing MLX and Dependencies"
echo "==============================================================="
echo ""

# Install MLX
pip3 install --quiet mlx mlx-lm
echo -e "${GREEN}OK${NC} MLX framework installed"

# Install FastAPI for the server
pip3 install --quiet fastapi uvicorn pydantic
echo -e "${GREEN}OK${NC} FastAPI server dependencies installed"

# Install Hugging Face Hub for model downloads
pip3 install --quiet huggingface_hub
echo -e "${GREEN}OK${NC} Hugging Face Hub installed"

echo ""
echo "==============================================================="
echo "  Setting up MageAgent Server"
echo "==============================================================="
echo ""

# Create directories
mkdir -p ~/.claude/mageagent
mkdir -p ~/.claude/scripts
mkdir -p ~/.claude/debug
mkdir -p ~/.cache/mlx-models
mkdir -p ~/Library/LaunchAgents

echo -e "${GREEN}OK${NC} Directories created"

# Copy server files
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cp "$SCRIPT_DIR/mageagent/server.py" ~/.claude/mageagent/
echo -e "${GREEN}OK${NC} MageAgent server installed"

cp "$SCRIPT_DIR/scripts/mageagent-server.sh" ~/.claude/scripts/
chmod +x ~/.claude/scripts/mageagent-server.sh
echo -e "${GREEN}OK${NC} Server management script installed"

echo ""
echo "==============================================================="
echo "  Downloading MLX Models (~110GB total)"
echo "==============================================================="
echo ""
echo "MageAgent requires the following models:"
echo ""
echo -e "  ${BLUE}Hermes-3-Llama-3.1-8B-8bit${NC} (9GB)"
echo "    Role: Tool calling specialist (Q8 for reliable function calls)"
echo ""
echo -e "  ${BLUE}Qwen2.5-72B-Instruct-8bit${NC} (77GB)"
echo "    Role: Primary reasoning (Q8 for tool support)"
echo ""
echo -e "  ${BLUE}Qwen2.5-Coder-32B-Instruct-4bit${NC} (18GB)"
echo "    Role: Code generation competitor (Q4 for speed)"
echo ""
echo -e "  ${BLUE}Qwen2.5-Coder-7B-Instruct-4bit${NC} (5GB)"
echo "    Role: Fast validation and judging (Q4 for speed)"
echo ""

read -p "Download models now? This will take significant time. (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Downloading models from Hugging Face..."
    echo "(This may take 30-60 minutes depending on your connection)"
    echo ""

    python3 << 'PYTHON'
from huggingface_hub import snapshot_download
import os

models_dir = os.path.expanduser("~/.cache/mlx-models")

print("1/4 Downloading Hermes-3-Llama-3.1-8B-8bit (9GB)...")
snapshot_download(
    'mlx-community/Hermes-3-Llama-3.1-8B-8bit',
    local_dir=f'{models_dir}/Hermes-3-Llama-3.1-8B-8bit'
)
print("    Done!")

print("2/4 Downloading Qwen2.5-Coder-7B-Instruct-4bit (5GB)...")
snapshot_download(
    'mlx-community/Qwen2.5-Coder-7B-Instruct-4bit',
    local_dir=f'{models_dir}/Qwen2.5-Coder-7B-Instruct-4bit'
)
print("    Done!")

print("3/4 Downloading Qwen2.5-Coder-32B-Instruct-4bit (18GB)...")
snapshot_download(
    'mlx-community/Qwen2.5-Coder-32B-Instruct-4bit',
    local_dir=f'{models_dir}/Qwen2.5-Coder-32B-Instruct-4bit'
)
print("    Done!")

print("4/4 Downloading Qwen2.5-72B-Instruct-8bit (77GB)...")
print("    This is the largest model and will take the longest...")
snapshot_download(
    'mlx-community/Qwen2.5-72B-Instruct-8bit',
    local_dir=f'{models_dir}/Qwen2.5-72B-Instruct-8bit'
)
print("    Done!")

print("\nAll models downloaded successfully!")
PYTHON

    echo -e "${GREEN}OK${NC} All models downloaded"
else
    echo ""
    echo "Skipping model download. You can download later with:"
    echo "  python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('mlx-community/MODEL_NAME', local_dir='~/.cache/mlx-models/MODEL_NAME')\""
fi

echo ""
echo "==============================================================="
echo "  Configuring Claude Code Router (Optional)"
echo "==============================================================="
echo ""

# Check for Claude Code Router
if command -v ccr &> /dev/null; then
    echo -e "${GREEN}OK${NC} Claude Code Router detected"

    # Create router config directory
    mkdir -p ~/.claude-code-router

    # Add MageAgent provider to config
    if [ -f ~/.claude-code-router/config.json ]; then
        # Check if mageagent already configured
        if grep -q "mageagent" ~/.claude-code-router/config.json; then
            echo -e "${GREEN}OK${NC} MageAgent already in router config"
        else
            echo -e "${YELLOW}!${NC} Please add MageAgent to ~/.claude-code-router/config.json manually"
            echo "   See README.md for configuration example"
        fi
    else
        # Create new config
        cat > ~/.claude-code-router/config.json << 'EOF'
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
  ],
  "Router": {
    "default": "mageagent,mageagent:hybrid"
  }
}
EOF
        echo -e "${GREEN}OK${NC} Router config created"
    fi
else
    echo -e "${YELLOW}!${NC} Claude Code Router not found"
    echo "   You can use MageAgent directly via API at http://localhost:3457"
fi

echo ""
echo "==============================================================="
echo "  Setting up Auto-Start (Optional)"
echo "==============================================================="
echo ""

read -p "Configure MageAgent to start automatically on boot? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Create LaunchAgent
    cat > ~/Library/LaunchAgents/com.adverant.mageagent.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.adverant.mageagent</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>$HOME/.claude/mageagent/server.py</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$HOME/.claude/debug/mageagent.log</string>
    <key>StandardErrorPath</key>
    <string>$HOME/.claude/debug/mageagent.error.log</string>
    <key>WorkingDirectory</key>
    <string>$HOME/.claude/mageagent</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin</string>
    </dict>
</dict>
</plist>
EOF

    # Load the LaunchAgent
    launchctl unload ~/Library/LaunchAgents/com.adverant.mageagent.plist 2>/dev/null || true
    launchctl load ~/Library/LaunchAgents/com.adverant.mageagent.plist

    echo -e "${GREEN}OK${NC} LaunchAgent installed and loaded"
    echo "   MageAgent will start automatically on boot"
else
    echo "Skipping auto-start setup"
fi

echo ""
echo "==============================================================="
echo "  Starting MageAgent Server"
echo "==============================================================="
echo ""

~/.claude/scripts/mageagent-server.sh start

sleep 3

# Check if server is running
if curl -s http://localhost:3457/health > /dev/null 2>&1; then
    echo -e "${GREEN}OK${NC} MageAgent server is running!"
else
    echo -e "${YELLOW}!${NC} Server may still be starting. Check logs with:"
    echo "   tail -f ~/.claude/debug/mageagent.log"
fi

echo ""
echo "==============================================================="
echo "  Installation Complete!"
echo "==============================================================="
echo ""
echo "MageAgent is now running at: ${GREEN}http://localhost:3457${NC}"
echo ""
echo "Available orchestration patterns:"
echo ""
echo "  ${BLUE}mageagent:hybrid${NC}     - 72B reasoning + Hermes-3 tools (recommended)"
echo "  ${BLUE}mageagent:validated${NC}  - 72B + 7B validation + Hermes-3 tools"
echo "  ${BLUE}mageagent:compete${NC}    - 72B vs 32B + 7B judge + Hermes-3 tools"
echo "  ${BLUE}mageagent:auto${NC}       - Automatic task routing"
echo "  ${BLUE}mageagent:tools${NC}      - Fast Hermes-3 tool calling only"
echo "  ${BLUE}mageagent:primary${NC}    - Direct 72B access"
echo ""
echo "Quick Test:"
echo ""
echo "  curl -X POST http://localhost:3457/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"mageagent:tools\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'"
echo ""
echo "Server Management:"
echo ""
echo "  ~/.claude/scripts/mageagent-server.sh status"
echo "  ~/.claude/scripts/mageagent-server.sh stop"
echo "  ~/.claude/scripts/mageagent-server.sh start"
echo "  ~/.claude/scripts/mageagent-server.sh logs"
echo ""
echo "Documentation: ${GREEN}https://github.com/adverant/nexus-local-mageagent${NC}"
echo ""
echo "==============================================================="
