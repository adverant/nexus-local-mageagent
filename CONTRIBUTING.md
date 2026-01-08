# Contributing to Nexus Local MageAgent

Thank you for your interest in contributing to MageAgent! This document provides guidelines for contributors.

## Ways to Contribute

- **Report bugs** - Help us identify and fix issues
- **Suggest features** - Share ideas for new orchestration patterns
- **Improve documentation** - Clarify instructions, add examples
- **Submit code** - Fix bugs, add features, optimize performance
- **Add orchestration patterns** - Create new multi-model workflows
- **Test on different systems** - Verify compatibility across Mac configurations

## Getting Started

### 1. Fork the Repository

Click the "Fork" button at the top of this repository.

### 2. Clone Your Fork

```bash
git clone https://github.com/YOUR_USERNAME/nexus-local-mageagent.git
cd nexus-local-mageagent
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

## Development Setup

### Prerequisites

- Apple Silicon Mac (M1/M2/M3/M4)
- 64GB+ unified memory (128GB recommended)
- Python 3.9+
- pip3

### Install Development Dependencies

```bash
# Install MLX and server dependencies
pip3 install mlx mlx-lm fastapi uvicorn pydantic huggingface_hub

# Download required models (takes time)
python3 << 'EOF'
from huggingface_hub import snapshot_download

# Minimum: Hermes-3 for tools + Qwen-7B for validation
snapshot_download('mlx-community/Hermes-3-Llama-3.1-8B-8bit',
                  local_dir='~/.cache/mlx-models/Hermes-3-Llama-3.1-8B-8bit')
snapshot_download('mlx-community/Qwen2.5-Coder-7B-Instruct-4bit',
                  local_dir='~/.cache/mlx-models/Qwen2.5-Coder-7B-Instruct-4bit')
EOF
```

### Run the Development Server

```bash
# Copy server to development location
cp mageagent/server.py ~/.claude/mageagent/

# Start server
python3 ~/.claude/mageagent/server.py
```

## Testing

Before submitting changes, test thoroughly:

### 1. Test Server Health

```bash
curl http://localhost:3457/health
```

### 2. Test Each Pattern

```bash
# Test tools pattern
curl -X POST http://localhost:3457/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mageagent:tools", "messages": [{"role": "user", "content": "Hello"}]}'

# Test hybrid pattern
curl -X POST http://localhost:3457/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mageagent:hybrid", "messages": [{"role": "user", "content": "Explain Python decorators"}]}'

# Test validated pattern
curl -X POST http://localhost:3457/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mageagent:validated", "messages": [{"role": "user", "content": "Write a binary search function in Python"}]}'
```

### 3. Test Tool Extraction

```bash
# Should extract tool calls
curl -X POST http://localhost:3457/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mageagent:hybrid", "messages": [{"role": "user", "content": "Read the file at /tmp/test.txt"}]}'
```

### 4. Test Memory Handling

```bash
# Monitor memory during large model loads
top -l 1 -o mem | head -10
```

## Code Guidelines

### Python Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Include docstrings for functions and classes
- Handle errors gracefully with informative messages

Example:
```python
async def generate_with_model(
    model_type: str,
    messages: List[ChatMessage],
    max_tokens: int = 2048,
    temperature: float = 0.7
) -> str:
    """
    Generate response using specified model.

    Args:
        model_type: One of 'tools', 'primary', 'validator', 'competitor'
        messages: List of chat messages
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated response text

    Raises:
        ValueError: If model_type is unknown
        FileNotFoundError: If model files not found
    """
    ...
```

### Adding New Orchestration Patterns

When adding a new pattern:

1. Add the async function in `server.py`:
```python
async def generate_YOUR_PATTERN(
    messages: List[ChatMessage],
    max_tokens: int = 2048,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Your pattern description.

    Flow:
    1. Step one
    2. Step two
    3. Extract tools via Hermes-3 if needed
    """
    # Implementation
    ...
```

2. Add the endpoint handler in `chat_completions()`:
```python
elif model_name == "mageagent:YOUR_PATTERN":
    result = await generate_YOUR_PATTERN(...)
```

3. Add to models list in `list_models()`:
```python
ModelInfo(id="mageagent:YOUR_PATTERN", created=int(time.time())),
```

4. Document in `docs/PATTERNS.md`

5. Update README.md pattern table

### Bash Scripts

- Use `#!/bin/bash` shebang
- Include descriptive comments
- Handle errors gracefully
- Provide helpful error messages
- Test on macOS

### Documentation

- Write in Markdown
- Use clear, concise language
- Include code examples
- Add troubleshooting tips
- Test all commands before documenting

## Bug Reports

When reporting bugs, include:

1. **System Information**
   - macOS version
   - Apple Silicon chip (M1/M2/M3/M4)
   - Unified memory amount
   - Python version

2. **Steps to Reproduce**
   - Exact commands run
   - Configuration used
   - Expected vs actual behavior

3. **Logs**
   ```bash
   tail -100 ~/.claude/debug/mageagent.log
   tail -100 ~/.claude/debug/mageagent.error.log
   ```

### Bug Report Template

```markdown
**System:**
- macOS: [version]
- Chip: M4 Max / M3 Pro / etc.
- Memory: 128GB / 64GB / etc.
- Python: [version]

**Steps to Reproduce:**
1. Step one
2. Step two
3. Step three

**Expected Behavior:**
[What you expected]

**Actual Behavior:**
[What actually happened]

**Logs:**
```
[Paste relevant logs]
```
```

## Feature Requests

When suggesting features:

1. **Describe the problem** - What pain point does this address?
2. **Propose a solution** - How would this pattern/feature work?
3. **Show the flow** - Diagram the model orchestration
4. **Consider trade-offs** - Memory, latency, quality

### Feature Request Template

```markdown
**Problem:**
[Description of the limitation]

**Proposed Pattern:**
[How the orchestration would work]

**Flow Diagram:**
```
User Request
     │
     ▼
[Model A] → [Model B] → [Hermes-3 Tools]
```

**Trade-offs:**
- Memory: [impact]
- Latency: [impact]
- Quality: [expected improvement]
```

## Pull Request Process

### 1. Update Your Branch

```bash
git fetch upstream
git rebase upstream/main
```

### 2. Commit Your Changes

Use clear, descriptive commit messages:

```bash
git commit -m "feat: Add self-critique pattern with iterative refinement"
git commit -m "fix: Resolve memory leak in model unloading"
git commit -m "docs: Add troubleshooting guide for M3 Macs"
```

**Commit Message Format:**
- `feat:` New features or patterns
- `fix:` Bug fixes
- `docs:` Documentation changes
- `refactor:` Code refactoring
- `test:` Test additions/changes
- `perf:` Performance improvements

### 3. Create Pull Request

1. Go to your fork on GitHub
2. Click "New Pull Request"
3. Fill out the PR template
4. Link related issues

### Pull Request Template

```markdown
## Description
[Brief description of changes]

## Type of Change
- [ ] New orchestration pattern
- [ ] Bug fix
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Other: [describe]

## Testing
- [ ] Tested on Apple Silicon Mac
- [ ] All patterns work correctly
- [ ] Tool extraction works
- [ ] Memory usage acceptable
- [ ] Documentation updated

## Pattern Details (if new pattern)
- Models used: [list]
- Memory required: [amount]
- Expected latency: [range]
- Quality improvement: [estimate]

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
```

## Architecture Decisions

When making architectural changes, document the decision:

### Decision Record Template

```markdown
## ADR: [Title]

### Context
[Why is this decision needed?]

### Decision
[What was decided?]

### Consequences
[What are the implications?]

### Alternatives Considered
[What other options were evaluated?]
```

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited for patterns they create

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/adverant/nexus-local-mageagent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/adverant/nexus-local-mageagent/discussions)

## Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone.

### Our Standards

**Positive behaviors:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what's best for the community

**Unacceptable behaviors:**
- Trolling, insulting/derogatory comments
- Public or private harassment
- Publishing others' private information
- Other conduct which could reasonably be considered inappropriate

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to MageAgent!

Your efforts help make local AI orchestration more powerful and accessible for everyone.

*Made with care by [Adverant](https://github.com/adverant)*
