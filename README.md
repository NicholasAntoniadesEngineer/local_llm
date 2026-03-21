# 🤖 Self-Improving Agent System

An autonomous agent running on Apple Silicon (MLX) that improves itself iteratively through code analysis, testing, and implementation of its own improvements.

## Quick Start

### Prerequisites
- Python 3.9+
- MLX installed (`pip install mlx-lm`)
- Virtual environment set up (`mlx_agent_env`)

### Run Self-Improvement

```bash
source mlx_agent_env/bin/activate

# Single improvement cycle
python3 self_improve.py

# Multiple cycles
bash run_improvement_cycles.sh 3

# With specific goal
python3 agent.py "Your goal here"
```

## System Components

| File | Purpose |
|------|---------|
| `agent.py` | Core ReAct loop with tools (search, code, file I/O) |
| `config.py` | Configuration & model selection |
| `memory.py` | Session memory & learning tracking |
| `reflection.py` | Pattern detection & stuck-loop analysis |
| `config_manager.py` | Config evolution for self-improvement |
| `self_improve.py` | Self-improvement harness |

## How It Works

1. **Agent reads its own code** (`read_file` on agent.py, memory.py, reflection.py)
2. **Identifies an improvement** (better loop detection, caching, optimization)
3. **Implements the change** (`write_file` to update code)
4. **Tests the change** (`run_python` with test code)
5. **Measures improvement** (shows speedup, efficiency gain, etc.)
6. **Iterates** - Goes to step 1 for next improvement

## Features

✅ **Quality-Aware Search** - Measures relevance of search results (0-1 scale)
✅ **Intelligent Phase Forcing** - Decides research vs coding based on data quality
✅ **Refined Query Suggestions** - Automatically tries different search angles
✅ **Session Memory** - Tracks discoveries, failures, and learnings
✅ **Loop Detection** - Recognizes when stuck and suggests pivots
✅ **Self-Improvement** - Agent improves its own capabilities

## Architecture

```
Agent (ReAct Loop)
├─ Tools: web_search, run_python, bash, read_file, write_file
├─ Memory: Session tracking, discoveries, failures
└─ Reflection: Pattern detection, improvement suggestions
```

## Self-Improvement Cycle

The agent iteratively:

1. **Analyzes** its own code
2. **Plans** specific improvements
3. **Implements** code changes
4. **Tests** the changes
5. **Measures** the improvement
6. **Commits** if successful
7. **Repeats**

## Testing

```bash
# Test search quality detection
source mlx_agent_env/bin/activate
python3 test_search_quality.py
```

## Measuring Progress

After each improvement cycle:

```bash
# See what files were created
ls -lh agent_outputs/

# Check code changes
git diff agent.py reflection.py memory.py

# View session memory
ls -lh ~/.claude/sessions/
```

## Configuration

All settings in `config.py`:

```python
class AgentConfig:
    max_iterations = 50
    web_search_timeout = 10
    code_execution_timeout = 30
    max_search_results = 5
    low_quality_threshold = 0.4  # Search relevance score
    high_quality_threshold = 0.6
```

## Improvement Ideas

The agent can improve:
- Loop detection (detect similar results, not just tool repeats)
- Memory efficiency (compress old discoveries)
- Search optimization (cache results)
- Tool effectiveness (better Python code generation)
- Reflection accuracy (better learning from failures)
- Confidence scoring (know when to search vs code)

## Performance

- **Model**: Qwen3-14B-4bit via MLX
- **Speed**: ~60 tokens/second on Apple M4 Max
- **Context**: 16,384 tokens
- **Memory**: Efficient session-based tracking

## Next Steps

1. Start with: `python3 self_improve.py`
2. Watch it improve itself
3. Check improvements in `agent_outputs/`
4. Run more cycles: `bash run_improvement_cycles.sh 5`

## Documentation

- **SELF_IMPROVING_SYSTEM.md** - Architecture & detailed guide
- **IMPROVEMENTS.md** - Recent enhancements

## Status

✅ Core system ready
✅ Self-improvement harness implemented
✅ Quality-aware search active
✅ Ready for continuous improvement cycles

---

**Last Updated**: March 21, 2026
**System**: Self-improving autonomous agent on MLX
**Goal**: Continuously improve its own capabilities
