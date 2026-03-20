"""Comprehensive tests for Constitutional AI rules engine."""

import asyncio
import pytest
from pathlib import Path

from src.rules.models import (
    HardRule,
    SoftRule,
    MetaRule,
    LearningRule,
    Critique,
    RuleViolation,
    ProposedRuleChange,
    Priority,
    Enforcement,
    ResponseType,
)
from src.rules.loader import RuleLoader, RuleCache
from src.rules.engine import RulesEngine
from src.rules.learner import RuleLearner
from src.rules.optimizer import RuleOptimizer


class TestModels:
    """Test Pydantic models."""

    def test_hard_rule_creation(self):
        """Test HardRule model creation."""
        rule = HardRule(
            id="H1",
            rule="Verify claims against minimum 2 independent sources",
            rationale="Single-source claims have high failure rate",
            priority=Priority.HIGH,
            enforcement=Enforcement.BLOCK_IF_VIOLATED,
        )
        assert rule.id == "H1"
        assert rule.enforcement == Enforcement.BLOCK_IF_VIOLATED
        assert rule.immutable is True

    def test_soft_rule_creation(self):
        """Test SoftRule model creation."""
        rule = SoftRule(
            id="S1",
            rule="Prefer primary sources over secondary summaries",
            confidence=0.85,
            effectiveness_score=0.82,
            priority=Priority.HIGH,
        )
        assert rule.confidence == 0.85
        assert rule.effectiveness_score == 0.82

    def test_learning_rule_creation(self):
        """Test LearningRule model creation."""
        rule = LearningRule(
            id="L1",
            rule="When researching AI models, check the official model card first",
            derived_from="task_uuid_abc123",
            confidence=0.6,
            failure_case="Claimed model had feature it didn't actually have",
        )
        assert rule.id == "L1"
        assert rule.derived_from == "task_uuid_abc123"
        assert rule.confidence == 0.6

    def test_critique_model(self):
        """Test Critique model."""
        critique = Critique(
            rule_id="H1",
            violates=True,
            explanation="Response only cites one source",
            severity="high",
            confidence=0.95,
            suggestions=["Add at least one more independent source"],
        )
        assert critique.violates is True
        assert critique.severity == "high"

    def test_rule_violation_model(self):
        """Test RuleViolation model."""
        critique = Critique(rule_id="H1", violates=True)
        violation = RuleViolation(
            rule_id="H1",
            response_text="This is a response",
            response_type=ResponseType.RESEARCH_FINDING,
            critique=critique,
        )
        assert violation.rule_id == "H1"
        assert violation.resolved is False

    def test_proposed_rule_change_model(self):
        """Test ProposedRuleChange model."""
        rule = LearningRule(
            id="L1",
            rule="New rule",
            derived_from="task_123",
        )
        proposal = ProposedRuleChange(
            change_type="add",
            rule=rule,
            rationale="Improves accuracy",
        )
        assert proposal.change_type == "add"
        assert proposal.status == "proposed"

    def test_rule_id_validation(self):
        """Test rule ID format validation."""
        with pytest.raises(ValueError):
            HardRule(id="INVALID", rule="Test rule")

    def test_confidence_bounds(self):
        """Test confidence bounds enforcement."""
        with pytest.raises(ValueError):
            SoftRule(
                id="S1",
                rule="Test",
                confidence=1.5,  # Invalid
            )

        with pytest.raises(ValueError):
            SoftRule(
                id="S1",
                rule="Test",
                confidence=-0.1,  # Invalid
            )


class TestRuleCache:
    """Test RuleCache functionality."""

    def test_cache_set_and_get(self):
        """Test cache operations."""
        cache = RuleCache(ttl_hours=24)
        cache.set("hash123", "H1", "research_finding", '{"violates": true}')
        result = cache.get("hash123", "H1", "research_finding")
        assert result == '{"violates": true}'

    def test_cache_miss(self):
        """Test cache miss."""
        cache = RuleCache()
        result = cache.get("nonexistent", "H1", "research_finding")
        assert result is None


class TestRuleLoader:
    """Test RuleLoader functionality."""

    def test_loader_initialization(self):
        """Test loader can be initialized."""
        loader = RuleLoader(rules_path="config/rules.yaml")
        assert loader.rules_path == Path("config/rules.yaml")
        assert isinstance(loader.cache, RuleCache)

    def test_load_rules(self):
        """Test loading rules from YAML."""
        loader = RuleLoader(rules_path="config/rules.yaml")
        rules = loader.load()

        assert "hard_rules" in rules
        assert "soft_rules" in rules
        assert "learning_rules" in rules
        assert len(rules["hard_rules"]) > 0
        assert len(rules["soft_rules"]) > 0

    def test_unique_id_validation(self):
        """Test unique ID validation."""
        loader = RuleLoader(rules_path="config/rules.yaml")
        rules = loader.load()

        all_ids = []
        for rule_list in rules.values():
            for rule in rule_list:
                all_ids.append(rule.id)

        # Should have no duplicates
        assert len(all_ids) == len(set(all_ids))

    def test_compile_to_xml(self):
        """Test XML compilation."""
        loader = RuleLoader(rules_path="config/rules.yaml")
        loader.load()
        xml = loader.compile_to_xml()

        assert "<rules>" in xml
        assert "<hard_rules>" in xml
        assert "<soft_rules>" in xml
        assert 'id="H' in xml or 'id="S' in xml  # At least one rule ID

    def test_get_rules_by_id(self):
        """Test retrieving rules by ID."""
        loader = RuleLoader(rules_path="config/rules.yaml")
        loader.load()

        rule = loader.get_rules_by_id("H1")
        assert rule is not None
        assert rule.id == "H1"

    def test_get_rule_lists(self):
        """Test getting rule lists by type."""
        loader = RuleLoader(rules_path="config/rules.yaml")
        loader.load()

        hard_rules = loader.get_hard_rules()
        soft_rules = loader.get_soft_rules()

        assert isinstance(hard_rules, list)
        assert isinstance(soft_rules, list)


class TestRulesEngine:
    """Test RulesEngine functionality."""

    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = RulesEngine(rules_path="config/rules.yaml")
        assert engine.loader is not None
        assert engine.max_revisions == 3
        assert engine.cache_hit_target == 0.7

    def test_statistics_tracking(self):
        """Test statistics tracking."""
        engine = RulesEngine(rules_path="config/rules.yaml")
        stats = engine.get_statistics()

        assert "total_evaluations" in stats
        assert "cache_hits" in stats
        assert "violations_found" in stats
        assert stats["cache_hit_target"] == 0.7

    def test_statistics_reset(self):
        """Test statistics reset."""
        engine = RulesEngine(rules_path="config/rules.yaml")
        engine._stats["total_evaluations"] = 10
        engine.reset_statistics()
        assert engine._stats["total_evaluations"] == 0


class TestRuleLearner:
    """Test RuleLearner functionality."""

    def test_learner_initialization(self):
        """Test learner initialization."""
        learner = RuleLearner(rules_path="config/rules.yaml")
        assert learner.loader is not None
        assert learner.improvement_threshold == 0.05
        assert learner.holdout_fraction == 0.1

    def test_proposed_rules_management(self):
        """Test managing proposed rules."""
        learner = RuleLearner(rules_path="config/rules.yaml")

        rule = LearningRule(
            id="L1",
            rule="Test rule",
            derived_from="task_123",
        )
        proposal = ProposedRuleChange(
            change_type="add",
            rule=rule,
            rationale="Improves quality",
        )

        learner._proposed_changes.append(proposal)
        proposals = learner.get_proposed_rules()
        assert len(proposals) == 1

    def test_get_approved_rules(self):
        """Test getting approved rules."""
        learner = RuleLearner(rules_path="config/rules.yaml")

        rule = LearningRule(id="L1", rule="Test", derived_from="task_123")
        proposal = ProposedRuleChange(
            change_type="add",
            rule=rule,
            rationale="Test",
        )
        proposal.status = "approved"
        learner._proposed_changes.append(proposal)

        approved = learner.get_approved_rules()
        assert len(approved) == 1
        assert approved[0].status == "approved"


class TestRuleOptimizer:
    """Test RuleOptimizer functionality."""

    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = RuleOptimizer(prompts_dir="config/prompts")
        assert optimizer.prompts_dir == Path("config/prompts")

    def test_metric_registration(self):
        """Test metric registration."""
        optimizer = RuleOptimizer()

        def test_metric(example):
            return 0.8

        optimizer.register_metric("test_metric", test_metric)
        assert "test_metric" in optimizer._metrics

    def test_create_default_templates(self):
        """Test creating default templates."""
        optimizer = RuleOptimizer(prompts_dir="/tmp/test_prompts")
        optimizer.create_default_templates()

        # Check files were created
        assert (Path("/tmp/test_prompts") / "critique.j2").exists()
        assert (Path("/tmp/test_prompts") / "proposal.j2").exists()
        assert (Path("/tmp/test_prompts") / "system.j2").exists()

    def test_template_rendering(self):
        """Test template rendering."""
        optimizer = RuleOptimizer(prompts_dir="/tmp/test_prompts")
        optimizer.create_default_templates()

        result = optimizer.render_template(
            "critique.j2",
            {
                "rule": "Test rule",
                "rationale": "Test rationale",
                "response_type": "research_finding",
                "response": "Test response",
            },
        )

        assert "Test rule" in result
        assert "Test response" in result


@pytest.mark.asyncio
async def test_rules_engine_enforce_async():
    """Test async rule enforcement (requires mock router)."""
    engine = RulesEngine(rules_path="config/rules.yaml")

    # Without router, should just track evaluation
    response, violations = await engine.enforce(
        "This is a test response.",
        ResponseType.GENERAL,
    )

    assert response is not None
    assert isinstance(violations, list)


def test_integration_rules_workflow():
    """Test complete rules workflow integration."""
    # Load rules
    loader = RuleLoader(rules_path="config/rules.yaml")
    rules = loader.load()

    assert len(rules["hard_rules"]) > 0
    assert len(rules["soft_rules"]) > 0

    # Compile to XML
    xml = loader.compile_to_xml()
    assert "<rules>" in xml

    # Create learner
    learner = RuleLearner(rules_path="config/rules.yaml")
    assert learner.loader is not None

    # Create optimizer
    optimizer = RuleOptimizer()
    optimizer.register_metric(
        "test",
        lambda x: 0.8,
    )
    assert "test" in optimizer._metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
