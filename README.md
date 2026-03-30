# Local Self-Improving LLM Agent

A self-improving AI agent running locally on Apple Silicon using MLX. Solves coding challenges, learns from successes and failures, generates new challenges, and collects training data for LoRA fine-tuning.

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

```bash
python tools/improve.py 1 --loop
```

This starts the self-improvement loop. Each cycle:

1. Picks a coding challenge (difficulty calibrated to model capability)
2. Generates a solution using Qwen3-14B
3. Tests the solution against assertions
4. On success: saves solution as few-shot example for future challenges
5. On failure: classifies error, accumulates patterns, retries with higher temperature
6. When all challenges solved: generates NEW challenges at the appropriate difficulty
7. Every 15 successes: prompts to run LoRA fine-tuning

Output shows live metrics:
```
CYCLE #5
CHALLENGE: LRU Cache
  Solve rates: d1=100%, d2=67%
  Few-shot: 340 chars from prior success
  Feedback injected (120 chars)
  Attempt 1/3 (temp=0.0)...
    SOLVED in 2.3s
Session: 4/5 (80%) | challenges: 80% of 5 [d1=1.0 d2=0.667]
```

## Self-Improvement Loops

Five feedback loops are active:

| Loop | What it does | Data flow |
|------|-------------|-----------|
| **Few-shot** | Successful solutions injected as examples in future prompts | solve → save → prompt next challenge |
| **Failure patterns** | Common error types warn future attempts | fail → classify → accumulate → prompt |
| **Adaptive difficulty** | Solve rates drive challenge selection | rates → pick harder/easier |
| **Cross-session memory** | Learnings persist across restarts | save session → retrieve next run |
| **LoRA training data** | Successes collected for model fine-tuning | solve → collect → export → train |

## LoRA Fine-Tuning

After accumulating 15+ successful solutions:

```bash
# Check readiness
python -m src.training --check

# Export training data
python -m src.training --export-only

# Run fine-tuning (requires MLX LoRA support)
python -m src.training --model mlx-community/Qwen3-14B-4bit --rank 8 --lr 1e-5
```

Training data is formatted as chat JSONL with system/user/assistant messages.

## Dynamic Challenge Generation

When all existing challenges are solved, the agent generates new ones:
- Uses the LLM to create novel coding problems
- Validates that tests aren't trivially passing
- Persists generated challenges to `run_output_data/generated_challenges.json`
- Difficulty scales with the model's demonstrated capability

## Project Structure

```
tools/improve.py              # Entry point: self-improvement loop
src/
  agent.py                    # MLX model loading + generation
  challenges.py               # 10 hardcoded challenges + test runner
  challenge_generator.py      # Dynamic challenge creation via LLM
  training.py                 # LoRA fine-tuning pipeline
  config.py                   # Model configurations
  memory.py                   # Cross-session learning
  skill_tree.py               # Skill dependency graph
  runtime/
    improve_runner.py          # Cycle orchestration + all feedback loops
    verifier.py                # Code validation gates
    llm_text.py                # Response parsing + code extraction
    controller.py              # Multi-step agent controller (advanced)
    tools.py                   # Tool execution (web search, file I/O)
skills/                        # 13 validated Python skill modules
run_output_data/               # Persistent state:
  metrics.json                 #   Live solve rates + difficulty tracking
  challenge_results.jsonl      #   Every challenge attempt with code
  successful_generations.jsonl #   Successful solutions for few-shot + training
  generated_challenges.json    #   LLM-generated challenges
  training_data.jsonl          #   Formatted data for LoRA fine-tuning
  sessions/                    #   Per-session memory for cross-session learning
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_MODEL` | `fast` | Model profile: `fast` (14B), `balanced` (27B) |
| `IMPROVE_LOOP_SLEEP_SEC` | `3` | Seconds between cycles |
| `MLX_METAL_SAFE_PROMPT_TOKENS` | `18432` | Max prompt tokens (Metal safety) |
