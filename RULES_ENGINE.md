# Constitutional AI Rules Engine — Implementation Guide

## Overview

The rules engine is a complete Constitutional AI system for enforcing research quality standards. It implements:

1. **Hard rules** (immutable, blocking)
2. **Soft rules** (learnable, advisory)
3. **Meta-rules** (conflict resolution)
4. **Learning rules** (agent-generated from failures)
5. **Self-improvement via A/B testing**
6. **Prompt optimization with DSPy**

## Architecture

```
RuleLoader (YAML parsing, XML compilation)
    ↓
RulesEngine (critique-revise loop)
    ├─ Hard rules (parallel checking)
    ├─ Soft rules (sequential checking)
    └─ Caching (>70% target hit rate)
    ↓
RuleLearner (failure analysis, A/B testing)
    └─ RuleOptimizer (prompt optimization)
```

## File Structure

```
src/rules/
├── __init__.py          # Package exports
├── models.py            # Pydantic v2 models
├── loader.py            # YAML parsing + XML compilation
├── engine.py            # Critique-revise loop
├── learner.py           # A/B testing + learning
└── optimizer.py         # DSPy prompt optimization
```

## Usage Examples

### Basic Usage: Load and Compile Rules

```python
from src.rules import RuleLoader

# Load rules from YAML
loader = RuleLoader("config/rules.yaml")
rules = loader.load()

# Compile to XML for system prompts
xml_rules = loader.compile_to_xml()

# Access specific rules
hard_rules = loader.get_hard_rules()
soft_rules = loader.get_soft_rules()
learning_rules = loader.get_learning_rules()
```

### Enforce Rules on Response

```python
from src.rules import RulesEngine, ResponseType
from src.llm.router import ModelRouter

# Initialize engine with LLM router for critique
router = ModelRouter("config/model_config.yaml")
engine = RulesEngine("config/rules.yaml", model_router=router)

# Enforce rules (critique-revise loop)
final_response, violations = await engine.enforce(
    response_text="Here's my research...",
    response_type=ResponseType.RESEARCH_FINDING
)

# Check statistics
stats = engine.get_statistics()
print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
print(f"Violations found: {stats['violations_found']}")
print(f"Revisions triggered: {stats['revisions_triggered']}")
```

### Learn from Failures

```python
from src.rules import RuleLearner

learner = RuleLearner("config/rules.yaml", model_router=router)

# Analyze failures
proposals = await learner.analyze_failures(
    task_id="task_uuid_123",
    violations=violations,
    task_context="Researching LLM inference techniques"
)

# A/B test proposals
for proposal in proposals:
    improvement = await learner.test_proposal(
        proposal,
        baseline_test_set=test_cases
    )

    # Commit if improvement > 5%
    if improvement > 0.05:
        await learner.commit_proposal(proposal, improvement)
```

### Optimize Prompts

```python
from src.rules import RuleOptimizer

optimizer = RuleOptimizer("config/prompts", model_router=router)

# Register quality metrics
optimizer.register_metric(
    "critique_accuracy",
    lambda example: example.get("score", 0.0)
)

# Optimize critique prompt with BootstrapFewShot
optimized = await optimizer.optimize_critique_prompt(
    training_examples=critique_examples,
    metric_name="critique_accuracy",
    num_shots=3
)

# Create default templates
optimizer.create_default_templates()

# Render templates with Jinja2
result = optimizer.render_template(
    "critique.j2",
    {"rule": rule, "response": response, "response_type": "research_finding"}
)
```

## Data Models

### Rule Models

```python
# Base rule
class Rule:
    id: str                     # Unique ID (H1, S2, L3)
    rule: str                   # Rule statement
    rationale: str              # Why this rule exists
    priority: Priority          # critical, high, medium, low

# Hard rule (immutable)
class HardRule(Rule):
    enforcement: Enforcement    # block_if_violated
    immutable: bool            # Always True

# Soft rule (learnable)
class SoftRule(Rule):
    confidence: float          # [0.0, 1.0]
    effectiveness_score: float # Historical score
    derived_from: str | None   # Task UUID

# Learning rule (agent-generated)
class LearningRule(Rule):
    confidence: float          # Starts at 0.5
    effectiveness_score: float # Updated by A/B testing
    derived_from: str          # Task UUID
    failure_case: str          # Example failure
```

### Evaluation Models

```python
# Critique of rule violation
class Critique:
    rule_id: str
    violates: bool
    explanation: str
    severity: "low" | "medium" | "high"
    confidence: float
    suggestions: list[str]

# Recorded violation
class RuleViolation:
    rule_id: str
    response_text: str
    response_type: ResponseType
    critique: Critique
    timestamp: datetime
    resolved: bool
    revision_attempt: int

# Proposed rule change
class ProposedRuleChange:
    change_type: "add" | "modify" | "remove"
    rule: LearningRule | SoftRule
    rationale: str
    based_on_failures: list[str]
    status: "proposed" | "testing" | "approved" | "rejected"
    test_results: dict  # {baseline_score, candidate_score, improvement_pct}
```

## Enforcement Process

### Phase 1: Hard Rule Checking (Parallel)

```
1. Load all hard rules
2. Evaluate in parallel using qwen3:8b
3. Early stop if any violations found
4. Attempt revision (max 3 times)
5. Re-check hard rules after revision
```

### Phase 2: Soft Rule Checking (Sequential)

```
1. Sort soft rules by priority
2. Check each rule in order
3. Log violations but don't block
4. Continue to next rule
```

### Caching Strategy

```
Cache key: SHA256(response[:100]) + rule_id + response_type
TTL: 24 hours
Target hit rate: >70%

Hit rate formula:
  cache_hits / total_evaluations
```

## Configuration

### rules.yaml Structure

```yaml
version: 1
updated: 2026-03-20

# Meta-rules: conflict resolution
meta_rules:
  - id: M1
    priority: critical
    rule: "When rules conflict, prefer accuracy over completeness"

# Hard rules: immutable, blocking
hard_rules:
  - id: H1
    rule: "Verify claims against minimum 2 independent sources"
    enforcement: "block_if_violated"
    rationale: "..."

# Soft rules: learnable, advisory
soft_rules:
  - id: S1
    priority: high
    confidence: 0.85
    rule: "Prefer primary sources..."
    effectiveness_score: 0.82

# Learning rules: agent-generated
learning_rules:
  - id: L1
    confidence: 0.6
    derived_from: "task_uuid_abc"
    rule: "..."
```

## Integration with Agent

### In Research Node

```python
# 1. Get rules XML for system prompt
rules_xml = self.engine.loader.compile_to_xml()

# 2. Enforce rules on research response
response, violations = await self.engine.enforce(
    response_text=research_result,
    response_type=ResponseType.RESEARCH_FINDING
)

# 3. Log violations to memory
await self.memory.store_violations(violations, task_id)
```

### In Reflect Node

```python
# 1. Analyze recent failures
violations = await self.memory.get_recent_violations(limit=20)

proposals = await self.learner.analyze_failures(
    task_id=self.task_id,
    violations=violations,
    task_context=self.context
)

# 2. A/B test proposals
for proposal in proposals:
    improvement = await self.learner.test_proposal(
        proposal,
        baseline_test_set=recent_test_cases
    )

    if improvement > 0.05:
        await self.learner.commit_proposal(proposal, improvement)
        await self.memory.store_proposal(proposal)
```

## Performance Characteristics

### LLM Calls

| Task | Model | Role | Tokens | Latency |
|------|-------|------|--------|---------|
| Hard rule critique | qwen3:8b | orchestrate | ~500 | ~0.5-1s |
| Soft rule critique | qwen3:8b | orchestrate | ~500 | ~0.5-1s |
| Response revision | qwen3:32b | reason | ~2000 | ~2-4s |
| Failure analysis | qwen3:32b | reason | ~1000 | ~1-2s |
| Proposal generation | qwen3:32b | reason | ~1000 | ~1-2s |

### Cache Performance

```
Without cache: 50+ API calls per response
With 70% hit rate: ~15 API calls per response
Average savings: 70% reduction in latency
```

## Extension Points

### Custom Metrics

```python
def accuracy_metric(example):
    """Score how accurate critique was."""
    expected = example["expected_violation"]
    predicted = example["critique"].violates
    return 1.0 if expected == predicted else 0.0

optimizer.register_metric("accuracy", accuracy_metric)
```

### Custom Rules

Add to `config/rules.yaml`:

```yaml
soft_rules:
  - id: S6
    priority: medium
    confidence: 0.5
    rule: "Your custom rule here"
    rationale: "Why this rule matters"
```

### Custom Templates

Add to `config/prompts/`:

```jinja2
{# config/prompts/custom.j2 #}
Custom prompt with {{ variable }}
```

## Testing

Run the comprehensive test suite:

```bash
python -m pytest tests/test_rules.py -v
```

Tests cover:
- Model creation and validation
- Rule loading and parsing
- YAML structure validation
- Unique ID enforcement
- XML compilation
- Cache operations
- Engine initialization
- Statistics tracking
- Learner workflows
- Optimizer functionality

## Debugging

### Enable Structured Logging

```python
import structlog

# Logs go to logs/agent.log
logger = structlog.get_logger("rules_engine")
logger.info("rule_violation", rule_id="H1", severity="high")
```

### Check Cache Hit Rate

```python
stats = engine.get_statistics()
print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
print(f"Target: {stats['cache_hit_target']:.1%}")
print(f"Cache size: {stats['cache_size']}")
```

### Inspect Violations

```python
for violation in violations:
    print(f"Rule: {violation.rule_id}")
    print(f"Severity: {violation.critique.severity}")
    print(f"Explanation: {violation.critique.explanation}")
    print(f"Suggestions: {violation.critique.suggestions}")
```

## Performance Tips

1. **Reuse ModelRouter**: Create once, share across engine/learner
2. **Batch rule checks**: Parallelize hard rules, use cache for repeats
3. **Monitor cache hit rate**: Aim for >70%
4. **Limit revisions**: Max 3 attempts to prevent infinite loops
5. **Use soft rules wisely**: Check only most important ones
6. **A/B test carefully**: Use holdout set (10%), 5% improvement threshold

## Future Enhancements

- [ ] Distributed cache (Redis) for multi-process
- [ ] Rule conflict detection and resolution
- [ ] Automatic rule deprecation for low-confidence rules
- [ ] Rule importance scoring via gradient-based attribution
- [ ] Multi-language rule support
- [ ] Rule versioning and rollback

## References

- **Constitutional AI**: Bai et al., 2022 (anthropic.com/constitutional-ai)
- **DSPy**: Khattab et al., 2023 (github.com/stanfordnlp/dspy)
- **LangGraph**: LangChain framework for agent orchestration
- **Pydantic v2**: Data validation and serialization

## Support

For issues, check:
1. `logs/agent.log` for structured logs
2. `src/rules/*.py` docstrings for API details
3. `tests/test_rules.py` for usage examples
4. `config/rules.yaml` for rule definitions
