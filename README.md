# Self-Improving Agent System

An autonomous agent on Apple Silicon (MLX) that continuously improves itself by building new features each cycle.

## Quick Start

```bash
cd /Users/nicholasantoniades/Documents/GitHub/local_llm
source mlx_agent_env/bin/activate

# Run single improvement cycle
python3 improve.py 1

# Run continuous cycles
bash run.sh
```

## System

| Component | Purpose |
|-----------|---------|
| `agent.py` | Core ReAct loop with tools |
| `memory.py` | Session memory & discoveries |
| `reflection.py` | Pattern detection & analysis |
| `config.py` | Configuration & thresholds |
| `config_manager.py` | Config versioning |
| `improve.py` | Improvement cycle runner |
| `run.sh` | Continuous execution |

## How It Works

Each cycle:
1. Agent gets a specific feature to build
2. Agent writes production-quality code
3. Agent tests it works
4. Code committed to git
5. Repeat with next feature

## Features

- **Quality-Aware Search**: Results scored 0-1 for relevance
- **Intelligent Phasing**: Research vs code decisions based on data quality
- **Loop Detection**: Recognizes stuck patterns
- **Session Memory**: Tracks discoveries and learnings
- **Git Integration**: All improvements committed automatically

## What Gets Built

- **Cycle 1**: Search result caching
- **Cycle 2**: Performance metrics
- **Cycle 3**: Memory compression
- **Cycle 4+**: Agent chooses improvements

## Configuration

All settings in `config.py`:
- Model selection (8B, 14B, 32B)
- Timeout values
- Quality thresholds
- Context budgets

## Monitoring

```bash
# Watch new features
watch -n 2 'ls -lh *.py'

# Track git commits
watch -n 2 'git log --oneline | head -10'

# Check progress
git log --oneline | grep "feat(cycle"
```

## Performance

- **Model**: Qwen3-14B-4bit (MLX)
- **Speed**: ~60 tokens/second
- **Context**: 16,384 tokens
- **Platform**: Apple Silicon optimized

## Documentation

See `docs/` folder for detailed information.

---

**Status**: Production ready
**Last Updated**: March 21, 2026
