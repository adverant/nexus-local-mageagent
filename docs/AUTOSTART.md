# Auto-Start Configuration

> Configure MageAgent to start automatically on macOS boot

## Overview

MageAgent can be configured to start automatically when your Mac boots using macOS LaunchAgents. This ensures the server is always available without manual intervention.

## Quick Setup

The install script can set this up for you:

```bash
./scripts/install.sh
# Answer "y" when prompted about auto-start
```

## Manual Setup

### Step 1: Create the LaunchAgent

Create the plist file at `~/Library/LaunchAgents/com.adverant.mageagent.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.adverant.mageagent</string>

    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>/Users/YOUR_USERNAME/.claude/mageagent/server.py</string>
    </array>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>/Users/YOUR_USERNAME/.claude/debug/mageagent.log</string>

    <key>StandardErrorPath</key>
    <string>/Users/YOUR_USERNAME/.claude/debug/mageagent.error.log</string>

    <key>WorkingDirectory</key>
    <string>/Users/YOUR_USERNAME/.claude/mageagent</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin</string>
    </dict>
</dict>
</plist>
```

**Important**: Replace `YOUR_USERNAME` with your actual username (e.g., `/Users/don/`).

### Step 2: Load the LaunchAgent

```bash
# Unload if already loaded
launchctl unload ~/Library/LaunchAgents/com.adverant.mageagent.plist 2>/dev/null

# Load the agent
launchctl load ~/Library/LaunchAgents/com.adverant.mageagent.plist
```

### Step 3: Verify It's Running

```bash
# Check if loaded
launchctl list | grep mageagent

# Check if server is responding
curl http://localhost:3457/health
```

## LaunchAgent Options Explained

| Key | Value | Description |
|-----|-------|-------------|
| `Label` | `com.adverant.mageagent` | Unique identifier for this service |
| `ProgramArguments` | Array | Command to run (python3 + script path) |
| `RunAtLoad` | `true` | Start when agent is loaded (at boot) |
| `KeepAlive` | `true` | Restart if process exits |
| `StandardOutPath` | Path | Where to write stdout logs |
| `StandardErrorPath` | Path | Where to write stderr logs |
| `WorkingDirectory` | Path | Working directory for the process |
| `EnvironmentVariables` | Dict | Environment variables to set |

## Managing the Service

### Start the Service

```bash
launchctl start com.adverant.mageagent
```

### Stop the Service

```bash
launchctl stop com.adverant.mageagent
```

### Restart the Service

```bash
launchctl stop com.adverant.mageagent
launchctl start com.adverant.mageagent
```

### Disable Auto-Start

```bash
launchctl unload ~/Library/LaunchAgents/com.adverant.mageagent.plist
```

### Re-Enable Auto-Start

```bash
launchctl load ~/Library/LaunchAgents/com.adverant.mageagent.plist
```

### Remove Auto-Start Completely

```bash
launchctl unload ~/Library/LaunchAgents/com.adverant.mageagent.plist
rm ~/Library/LaunchAgents/com.adverant.mageagent.plist
```

## Viewing Logs

### Real-time Server Logs

```bash
tail -f ~/.claude/debug/mageagent.log
```

### Error Logs

```bash
tail -f ~/.claude/debug/mageagent.error.log
```

### System Logs (launchd)

```bash
log show --predicate 'subsystem == "com.apple.launchd"' --last 5m | grep mageagent
```

## Troubleshooting

### Service Won't Start

1. **Check the plist syntax**:
   ```bash
   plutil -lint ~/Library/LaunchAgents/com.adverant.mageagent.plist
   ```

2. **Verify paths exist**:
   ```bash
   ls -la ~/.claude/mageagent/server.py
   ls -la ~/.claude/debug/
   ```

3. **Check Python is accessible**:
   ```bash
   /usr/bin/python3 --version
   ```

4. **Check error logs**:
   ```bash
   cat ~/.claude/debug/mageagent.error.log
   ```

### Service Keeps Restarting

If `KeepAlive` is true and the service crashes, launchd will restart it. Check logs for crash reasons:

```bash
# View recent crashes
cat ~/.claude/debug/mageagent.error.log | tail -100
```

Common causes:
- Missing Python dependencies
- Port 3457 already in use
- Insufficient memory for models

### Port Already in Use

```bash
# Find what's using port 3457
lsof -i :3457

# Kill the process if needed
kill -9 <PID>
```

### Memory Issues

If the server crashes due to memory, you may need to:

1. Close other applications
2. Use smaller models (edit server.py)
3. Increase swap space

Check memory usage:
```bash
# Current memory pressure
memory_pressure

# Top memory consumers
top -l 1 -o mem | head -20
```

## Advanced Configuration

### Delay Start Until Network Ready

Add these keys to wait for network:

```xml
<key>NetworkState</key>
<true/>
```

### Run Only When Logged In

The default configuration in `~/Library/LaunchAgents/` only runs when you're logged in. For system-wide (all users), use `/Library/LaunchDaemons/` instead (requires admin).

### Resource Limits

Add resource limits if needed:

```xml
<key>HardResourceLimits</key>
<dict>
    <key>NumberOfFiles</key>
    <integer>1024</integer>
</dict>

<key>SoftResourceLimits</key>
<dict>
    <key>NumberOfFiles</key>
    <integer>1024</integer>
</dict>
```

### Throttle Restarts

Prevent rapid restart loops:

```xml
<key>ThrottleInterval</key>
<integer>30</integer>
```

This waits 30 seconds between restart attempts.

## Verification Checklist

After setup, verify everything works:

- [ ] LaunchAgent file exists at correct path
- [ ] File syntax is valid (`plutil -lint`)
- [ ] Agent is loaded (`launchctl list | grep mageagent`)
- [ ] Server responds (`curl http://localhost:3457/health`)
- [ ] Logs are being written (`ls -la ~/.claude/debug/`)
- [ ] Service survives reboot (restart and check)

## Alternative: Using the Management Script

Instead of launchctl directly, you can use the provided script:

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

The management script provides a simpler interface but doesn't persist across reboots unless combined with LaunchAgents.

---

*Made with care by [Adverant](https://github.com/adverant)*
