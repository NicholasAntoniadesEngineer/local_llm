# Project Context for Claude

## Overview

Self-improving agent system running on Apple Silicon (MLX). Agent autonomously builds features each cycle.

## Core Files (Do NOT Delete)

- **agent.py** - Main ReAct loop with tools (web_search, run_python, bash, read/write files)
- **memory.py** - Session memory system
- **reflection.py** - Loop detection and pattern analysis
- **config.py** - All configuration in one place
- **config_manager.py** - Config versioning

## Improvement System

- **improve.py** - Single script to run improvement cycles
- **run.sh** - Bash script for continuous cycles

## How to Use

```bash
# Single cycle
python3 improve.py 1

# Continuous
bash run.sh
```

## Production Standards

- No magic numbers (all in config.py)
- Type hints on all functions
- Docstrings on all public methods
- Error handling with clear messages
- Git commits for all changes

## Key Features

1. **Quality-Aware Search** - Scores results 0-1, detects low-quality research
2. **Intelligent Phasing** - Decides when to research vs code based on data
3. **Loop Detection** - Recognizes stuck patterns and suggests pivots
4. **Session Memory** - Tracks discoveries, learnings, failures

## Do NOT

- Add new improvement scripts (only improve.py)
- Add run scripts (only run.sh)
- Create duplicate documentation
- Leave output files in root (use agent_outputs/)
- Commit generated code/logs

## Do

- Keep config.py as single source of truth
- Clean up old/unused files
- Add to docs/ folder for documentation
- Test code before committing
- Use clear commit messages

## Current State

Production-ready. Agent can build features autonomously.
