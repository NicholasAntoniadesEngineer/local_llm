# Agent Improvements - March 21, 2026

## Overview

Enhanced the MLX agent system with **quality-aware search and intelligent phase forcing**. The agent is now smarter about research and won't waste time trying to code from bad information.

## Key Improvements

### 1. Search Result Quality Scoring (NEW)

**What it does:**
- Scores each search result on a 0-1 scale for relevance
- Calculates average relevance across all results
- Shows quality indicator: 🔴 LOW / 🟡 MEDIUM / 🟢 HIGH

**Implementation:**
```python
def _score_relevance(self, title: str, url: str, query: str) -> float:
    # Boosts: Keywords in title/URL, official docs, tutorials, API docs
    # Penalizes: Generic search pages, unrelated domains
    # Returns: 0-1 score showing how relevant this result is
```

**Benefits:**
- Agent can detect when search results are irrelevant (< 0.4 score)
- Recognizes high-quality results (> 0.6 score)
- Makes data-driven decisions instead of guessing

### 2. Smart Refined Query Suggestions (ENHANCED)

**What it does:**
- When search results are low quality, suggests completely different search angles
- Rotates through 6 different strategies: tutorial → docs → github → examples → api → guide

**Example:**
```
Original query: "prediction markets"

Strategy 1: "prediction markets' tutorial"
Strategy 2: "prediction markets' documentation"
Strategy 3: "prediction markets' github"
Strategy 4: "prediction markets' example code"
Strategy 5: "prediction markets' api"
Strategy 6: "prediction' markets guide"
```

**Benefits:**
- Each failed search tries a completely different angle
- Avoids repetitive, ineffective searches
- Increases chance of finding relevant information

### 3. Quality-Based Phase Forcing (NEW)

**Before:** Force to code phase after 3 web_searches regardless of quality

**After:** Make intelligent decisions based on research quality:

```
IF search_quality < 0.4 (LOW):
  → Try refined search (strategy 1, 2, 3...)
  → Only force to code after 2 refined attempts fail

IF search_quality 0.4-0.6 (MEDIUM):
  → Can move to code if research_count >= 3

IF search_quality > 0.6 (HIGH):
  → Good to move to code immediately
```

**Benefits:**
- No longer codes from poor research
- Adapts search strategy before giving up
- Makes decisions based on actual data quality

### 4. Search Quality Tracking

**What it tracks:**
- Average relevance score from last search
- Quality level (LOW/MEDIUM/HIGH)
- Number of junk results filtered

**Code:**
```python
self._last_search_quality = {
    'average_relevance': 0.75,
    'quality_level': '🟢 HIGH'
}
```

**Benefits:**
- Phase forcing logic can check quality
- Run loop knows when to persist vs. pivot
- Memory system can learn from quality patterns

### 5. Enhanced Initial System Prompt

**New education for the model:**
- Explains why search quality matters
- Shows that 🔴 LOW quality = need to try different query
- Emphasizes specific searches > generic searches
- Helps model understand it should adapt strategy

## Performance Impact

### Before (Old Agent)
- Got stuck in search loops (same query repeatedly)
- Forced to code from irrelevant research
- No quality measurement
- Failed when generic searches returned junk

### After (New Agent)
- Detects low-quality results immediately
- Automatically tries different search angles
- Only moves to code when research is solid
- Succeeds when initial searches fail

### Example Scenario

**Goal:** "Build a prediction market trading bot"

**Old behavior:**
1. Search: "prediction markets" → Gets generic results
2. Search: "prediction markets api" → Still generic
3. Search: "prediction markets python" → Still not great
4. Force to code phase → Write code from bad research → Fails

**New behavior:**
1. Search: "prediction markets" → Results score 0.35 (LOW) 🔴
2. Agent suggests: "prediction markets' tutorial"
3. Search: "prediction markets' tutorial" → Results score 0.72 (HIGH) 🟢
4. Move to code phase → Write code from GOOD research → Works!

## Next Iterations

### Planned Improvements

1. **Loop Detection Enhancement**
   - Track not just "same tool 3x" but "same tool producing same quality 3x"
   - Detect repeated failures more reliably

2. **Memory Compression**
   - Old discoveries are compressed/forgotten
   - Recent discoveries have more weight
   - System stays sharp as memory grows

3. **Result Caching**
   - Cache search results to avoid repeating identical searches
   - Detect when agent is asking same question differently

4. **Confidence Scoring**
   - Agent assigns confidence to its own beliefs
   - Uses this to decide whether to search again
   - Balances exploration vs. exploitation

### Self-Improvement System

**New script:** `self_improve.py`

This allows the agent to:
- Read its own source code
- Identify improvements
- Implement changes
- Test results
- Iterate autonomously

**Kick off self-improvement:**
```bash
source mlx_agent_env/bin/activate
python3 self_improve.py
```

The agent will improve itself iteratively, creating better versions over time.

## Testing

### Test Search Quality Detection

```bash
source mlx_agent_env/bin/activate
python3 test_search_quality.py
```

Shows:
- How quality scores work
- What gets flagged as LOW/MEDIUM/HIGH
- How refined queries are suggested

### Verify Syntax

```bash
python3 -m py_compile agent.py
```

## Files Changed

| File | Changes |
|------|---------|
| `agent.py` | Added quality scoring, refined queries, quality-based phase forcing |
| `test_search_quality.py` | NEW - Test quality detection |
| `self_improve.py` | NEW - Self-improvement harness |
| `IMPROVEMENTS.md` | This file |

## Configuration

All quality thresholds in `config.py`:
- LOW quality threshold: 0.4
- MEDIUM quality threshold: 0.6
- Max refined query attempts: 2
- Max search results: 5

These can be tuned if needed.

## Deployment

Agent is ready to use:

```bash
source mlx_agent_env/bin/activate

# Standard goal
python3 agent.py "Build a prediction market bot"

# Self-improvement (for agent to improve itself)
python3 self_improve.py
```

## Key Insight

The agent is now **resilient to bad search results**. Instead of "search once and code", it now follows:

**RESEARCH QUALITY LOOP:**
1. Search with initial query
2. Measure quality
3. If bad quality, try different angle
4. Repeat until quality is good
5. THEN code with confidence

This is how expert researchers actually work - they don't use the first result, they keep searching until they find good sources.

---

**Status:** Ready for testing and iteration
**Last Updated:** March 21, 2026
