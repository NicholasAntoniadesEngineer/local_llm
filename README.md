# Local Self-Improving LLM Agent

A self-improving AI agent that runs locally on Apple Silicon using MLX. Solves coding challenges, learns from successes and failures, and measurably improves over time.

## Requirements

- Apple Silicon Mac (M4 Max recommended, 36GB+ unified memory)
- Python 3.11+
- MLX framework

## Setup

```bash
pip install -r requirements.txt
```

The model (`Qwen3-14B-4bit`) downloads automatically on first run (~8GB).

## Run

**Self-improvement loop** (primary usage):
```bash
python tools/improve.py 1 --loop
```

This will:
1. Load the Qwen3-14B model
2. Solve coding challenges (FizzBuzz, LRU Cache, expression parser, etc.)
3. Learn from successes (few-shot examples fed to next challenge)
4. Learn from failures (common error patterns injected into prompts)
5. Progress to harder challenges as solve rate improves

**Single cycle** (for testing):
```bash
python tools/improve.py 1
```

**Monitor performance** (separate terminal):
```bash
python tools/monitor.py
```

## How It Works

```
Challenge selected (easiest unsolved first)
    |
    v
LLM generates solution code (Qwen3-14B, single-shot)
    |
    v
Code extracted + syntax pre-checked
    |
    v
Tests run against solution
    |
    +---> PASS: Save solution as few-shot example for next challenge
    |           Update solve rates, advance difficulty
    |
    +---> FAIL: Classify error type (syntax, import, assertion, etc.)
                Accumulate failure patterns for future prompts
                Retry with higher temperature (0.0 -> 0.3 -> 0.6)
```

## Project Structure

```
tools/improve.py          # Entry point: run the self-improvement loop
src/
  agent.py                # MLX model loading + generation
  challenges.py           # 10 coding challenges with test suites
  config.py               # Model configurations (14B, 27B, 32B)
  skill_tree.py            # Skill dependency graph + UCB1 selection
  runtime/
    improve_runner.py      # Improvement cycle orchestration + feedback loops
    verifier.py            # Code validation gates
    llm_text.py            # Response parsing + code extraction
    controller.py          # Multi-step agent controller (advanced)
    tools.py               # Tool execution (web search, file I/O)
skills/                    # 13 generated Python modules
runs/                      # Persistent state (skill tree DB)
run_output_data/           # Per-run logs, metrics, challenge results
```

## Key Files

| What you want to do | File |
|---------------------|------|
| Run the system | `tools/improve.py` |
| Add new challenges | `src/challenges.py` |
| Change the model | `src/config.py` (edit `AGENT_MODEL` env var) |
| See results | `run_output_data/metrics.json` |
| See challenge history | `run_output_data/challenge_results.jsonl` |
| See what the model generated | `run_output_data/successful_generations.jsonl` |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_MODEL` | `fast` | Model profile: `fast` (14B), `balanced` (27B), `primary` (32B) |
| `IMPROVE_LOOP_SLEEP_SEC` | `3` | Seconds between cycles |
| `MLX_METAL_SAFE_PROMPT_TOKENS` | `18432` | Max prompt tokens (Metal safety) |
