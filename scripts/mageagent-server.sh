#!/bin/bash
# MageAgent Server Launcher
# Multi-Model LLM Orchestrator for MLX

MAGEAGENT_DIR="$HOME/.claude/mageagent"
LOG_FILE="$HOME/.claude/debug/mageagent.log"
ERROR_LOG="$HOME/.claude/debug/mageagent.error.log"
PID_FILE="$MAGEAGENT_DIR/mageagent.pid"
PORT=3457

# Ensure directories exist
mkdir -p "$HOME/.claude/debug"
mkdir -p "$MAGEAGENT_DIR"

usage() {
    echo "MageAgent Server - Multi-Model LLM Orchestrator"
    echo ""
    echo "Usage: $0 {start|stop|restart|status|logs|test}"
    echo ""
    echo "Commands:"
    echo "  start   - Start the MageAgent server"
    echo "  stop    - Stop the MageAgent server"
    echo "  restart - Restart the MageAgent server"
    echo "  status  - Check if server is running"
    echo "  logs    - Tail the server logs"
    echo "  test    - Run a quick test against the server"
    echo ""
    echo "API Endpoints:"
    echo "  mageagent:auto      - Intelligent task routing"
    echo "  mageagent:validated - Generate + validate pattern"
    echo "  mageagent:compete   - Competing models with judge"
    echo "  mageagent:primary   - Direct 72B model access"
    echo "  mageagent:validator - Direct 7B validator access"
    echo "  mageagent:competitor - Direct 32B model access"
}

is_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

start_server() {
    if is_running; then
        echo "MageAgent already running (PID: $(cat $PID_FILE))"
        echo "API: http://localhost:$PORT"
        return 0
    fi

    echo "Starting MageAgent server on port $PORT..."
    echo "Log file: $LOG_FILE"

    cd "$MAGEAGENT_DIR"

    # Start with nohup to detach from terminal
    nohup python3 -u server.py > "$LOG_FILE" 2> "$ERROR_LOG" &
    echo $! > "$PID_FILE"

    # Wait a moment for server to start
    sleep 2

    if is_running; then
        echo "MageAgent started (PID: $(cat $PID_FILE))"
        echo ""
        echo "Waiting for server to be ready..."

        # Wait up to 30 seconds for server to respond
        for i in {1..30}; do
            if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
                echo "Server ready!"
                echo "API: http://localhost:$PORT"
                echo "Docs: http://localhost:$PORT/docs"
                return 0
            fi
            sleep 1
            echo -n "."
        done

        echo ""
        echo "Warning: Server started but not responding yet."
        echo "Check logs: tail -f $LOG_FILE"
    else
        echo "Failed to start MageAgent. Check error log:"
        tail -20 "$ERROR_LOG"
        return 1
    fi
}

stop_server() {
    if ! is_running; then
        echo "MageAgent not running"
        rm -f "$PID_FILE"
        return 0
    fi

    PID=$(cat "$PID_FILE")
    echo "Stopping MageAgent (PID: $PID)..."

    kill "$PID" 2>/dev/null

    # Wait for process to stop
    for i in {1..10}; do
        if ! kill -0 "$PID" 2>/dev/null; then
            rm -f "$PID_FILE"
            echo "MageAgent stopped"
            return 0
        fi
        sleep 1
    done

    # Force kill if still running
    echo "Force killing..."
    kill -9 "$PID" 2>/dev/null
    rm -f "$PID_FILE"
    echo "MageAgent stopped (forced)"
}

show_status() {
    if is_running; then
        PID=$(cat "$PID_FILE")
        echo "MageAgent: RUNNING (PID: $PID)"
        echo "Port: $PORT"
        echo ""

        # Check health endpoint
        HEALTH=$(curl -s "http://localhost:$PORT/health" 2>/dev/null)
        if [ -n "$HEALTH" ]; then
            echo "Health: $HEALTH"
        else
            echo "Health: Not responding"
        fi

        echo ""
        echo "Loaded models:"
        curl -s "http://localhost:$PORT/health" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print('  ' + '\n  '.join(d.get('loaded_models', [])))" 2>/dev/null || echo "  Unable to query"
    else
        echo "MageAgent: STOPPED"

        # Check if something else is using the port
        if lsof -i:$PORT > /dev/null 2>&1; then
            echo "Warning: Port $PORT is in use by another process"
            lsof -i:$PORT
        fi
    fi
}

show_logs() {
    echo "Showing MageAgent logs (Ctrl+C to exit)..."
    echo "---"
    tail -f "$LOG_FILE"
}

run_test() {
    if ! is_running; then
        echo "MageAgent not running. Start it first with: $0 start"
        return 1
    fi

    echo "Testing MageAgent server..."
    echo ""

    echo "1. Testing health endpoint..."
    curl -s "http://localhost:$PORT/health" | python3 -m json.tool
    echo ""

    echo "2. Testing models list..."
    curl -s "http://localhost:$PORT/v1/models" | python3 -m json.tool
    echo ""

    echo "3. Testing simple completion (validator model)..."
    echo "Request: Write a hello world in Python"

    RESPONSE=$(curl -s -X POST "http://localhost:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "mageagent:validator",
            "messages": [{"role": "user", "content": "Write a hello world in Python. Just the code, nothing else."}],
            "max_tokens": 100,
            "temperature": 0.3
        }')

    echo "Response:"
    echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" 2>/dev/null || echo "$RESPONSE"
    echo ""

    echo "Test complete!"
}

case "$1" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        stop_server
        sleep 2
        start_server
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    test)
        run_test
        ;;
    *)
        usage
        exit 1
        ;;
esac
