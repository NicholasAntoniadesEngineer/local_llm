# Architecture

## System Components

### Core Agent (`agent.py`)
- ReAct loop implementation
- Tool execution (web_search, run_python, bash, read/write files)
- Quality-aware search with relevance scoring
- Intelligent phase forcing based on data quality

### Memory System (`memory.py`)
- Session memory tracking
- Discovery recording
- Iteration history
- Failure analysis

### Reflection Engine (`reflection.py`)
- Loop detection
- Pattern analysis
- Strategy recommendations

### Configuration (`config.py`)
- Model selection (8B, 14B, 32B)
- Timeout settings
- Context budgets
- Quality thresholds

### Configuration Manager (`config_manager.py`)
- Config evolution
- Versioning
- Performance tracking

## Improvement Cycle

Each improvement cycle (`improve.py`):

1. **Define**: Agent gets a specific feature to build
2. **Implement**: Agent writes the code
3. **Test**: Agent verifies it works
4. **Commit**: Changes saved to git
5. **Repeat**: Next cycle starts

## Execution

```bash
# Single cycle
python3 improve.py 1

# Continuous cycles
bash run.sh
```

## Performance

- Model: Qwen3-14B-4bit (MLX)
- Speed: ~60 tokens/second
- Context: 16,384 tokens
- Inference: Apple Silicon optimized
