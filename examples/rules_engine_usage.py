"""
Example usage of the Constitutional AI Rules Engine.

This file demonstrates:
1. Loading rules from YAML
2. Enforcing rules on responses
3. Learning from failures
4. Optimizing prompts
"""

import asyncio
from src.rules import (
    RuleLoader,
    RulesEngine,
    RuleLearner,
    RuleOptimizer,
    ResponseType,
)
from src.llm.router import ModelRouter


async def example_1_load_and_compile_rules():
    """Example 1: Load rules and compile to XML."""
    print("\n=== Example 1: Load and Compile Rules ===")

    # Load rules from YAML
    loader = RuleLoader("config/rules.yaml")
    rules = loader.load()

    print(f"Loaded {len(rules['hard_rules'])} hard rules")
    print(f"Loaded {len(rules['soft_rules'])} soft rules")
    print(f"Loaded {len(rules['learning_rules'])} learning rules")

    # Compile to XML for system prompts
    xml_rules = loader.compile_to_xml()
    print(f"\nCompiled XML size: {len(xml_rules)} bytes")

    # Access specific rules
    hard_rules = loader.get_hard_rules()
    print(f"\nHard rules ({len(hard_rules)}):")
    for rule in hard_rules[:3]:
        print(f"  {rule.id}: {rule.rule[:60]}...")

    soft_rules = loader.get_soft_rules()
    print(f"\nSoft rules ({len(soft_rules)}):")
    for rule in soft_rules[:3]:
        print(f"  {rule.id}: {rule.rule[:60]}... (confidence={rule.confidence:.2f})")

    return loader


async def example_2_enforce_rules(loader: RuleLoader):
    """Example 2: Enforce rules on a research response."""
    print("\n=== Example 2: Enforce Rules on Response ===")

    # Initialize router for LLM calls
    router = ModelRouter("config/model_config.yaml")

    # Create rules engine
    engine = RulesEngine(
        "config/rules.yaml",
        model_router=router,
        max_revisions=3,
    )

    # Sample response
    response_text = """
    According to my research, Machine Learning is becoming more important.
    This is because models can now process large amounts of data efficiently.
    """

    print(f"Original response ({len(response_text)} chars)")
    print(f"Response preview: {response_text[:100]}...")

    # Enforce rules
    final_response, violations = await engine.enforce(
        response_text=response_text,
        response_type=ResponseType.RESEARCH_FINDING,
    )

    # Check results
    stats = engine.get_statistics()
    print(f"\nEnforcement results:")
    print(f"  Violations found: {len(violations)}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")
    print(f"  Total evaluations: {stats['total_evaluations']}")

    if violations:
        print(f"\nViolations:")
        for v in violations:
            print(f"  {v.rule_id}: {v.critique.explanation}")

    return engine, violations


async def example_3_learn_from_failures(engine: RulesEngine, violations: list):
    """Example 3: Learn from failures via A/B testing."""
    print("\n=== Example 3: Learn from Failures ===")

    if not violations:
        print("No violations to learn from in this example")
        return

    # Initialize learner
    router = ModelRouter("config/model_config.yaml")
    learner = RuleLearner(
        "config/rules.yaml",
        model_router=router,
        improvement_threshold=0.05,
    )

    # Analyze failures
    task_id = "example_task_001"
    context = "Research on machine learning best practices"

    proposals = await learner.analyze_failures(
        task_id=task_id,
        violations=violations,
        task_context=context,
    )

    print(f"Analyzed {len(violations)} violations")
    print(f"Generated {len(proposals)} rule proposals")

    for proposal in proposals:
        print(f"\nProposal:")
        print(f"  Change type: {proposal.change_type}")
        print(f"  Rule: {proposal.rule.rule[:60]}...")
        print(f"  Rationale: {proposal.rationale}")

    # A/B test proposals (simplified)
    for proposal in proposals:
        # Create mock test set
        test_set = [
            {
                "response": "Test response 1",
                "violations": violations[:1] if violations else [],
            },
        ]

        # Test proposal
        improvement = await learner.test_proposal(proposal, test_set)
        print(f"\nA/B Test results for {proposal.rule.id}:")
        print(f"  Improvement: {improvement:.1%}")
        print(f"  Threshold: 5%")

        # Commit if improvement meets threshold
        if improvement > 0.05:
            await learner.commit_proposal(proposal, improvement)
            print(f"  Status: APPROVED")
        else:
            print(f"  Status: REJECTED")

    # Get approval history
    approved = learner.get_approved_rules()
    print(f"\nTotal approved rules: {len(approved)}")


async def example_4_optimize_prompts():
    """Example 4: Optimize rule critique prompts."""
    print("\n=== Example 4: Optimize Prompts ===")

    # Create optimizer
    optimizer = RuleOptimizer("config/prompts")

    # Register quality metrics
    def accuracy_metric(example):
        """Measure critique accuracy."""
        expected = example.get("expected_violation", False)
        predicted = example.get("critique", {}).get("violates", False)
        return 1.0 if expected == predicted else 0.0

    def clarity_metric(example):
        """Measure explanation clarity (simpler heuristic)."""
        explanation = example.get("critique", {}).get("explanation", "")
        return min(1.0, len(explanation) / 100)  # Reward longer explanations

    optimizer.register_metric("accuracy", accuracy_metric)
    optimizer.register_metric("clarity", clarity_metric)

    print("Registered metrics: accuracy, clarity")

    # Create default templates
    optimizer.create_default_templates()
    print("Created default prompt templates:")
    print("  - config/prompts/critique.j2")
    print("  - config/prompts/proposal.j2")
    print("  - config/prompts/system.j2")
    print("  - config/prompts/reflect.j2")

    # Render a template
    template_result = optimizer.render_template(
        "critique.j2",
        {
            "rule": "Verify claims against multiple sources",
            "rationale": "Reduces false positives",
            "response_type": "research_finding",
            "response": "Machine learning is useful for many tasks.",
        },
    )

    print(f"\nRendered critique template ({len(template_result)} chars):")
    print(template_result[:200] + "...")


async def example_5_statistics_and_monitoring():
    """Example 5: Monitor system statistics."""
    print("\n=== Example 5: Statistics and Monitoring ===")

    loader = RuleLoader("config/rules.yaml")
    loader.load()

    engine = RulesEngine("config/rules.yaml")

    # Get statistics
    stats = engine.get_statistics()
    print("Engine statistics:")
    print(f"  Total evaluations: {stats['total_evaluations']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")
    print(f"  Cache target: {stats['cache_hit_target']:.1%}")
    print(f"  Cache size: {stats['cache_size']}")
    print(f"  Violations found: {stats['violations_found']}")
    print(f"  Revisions triggered: {stats['revisions_triggered']}")

    # Get rule summary
    rules = loader.load()
    print(f"\nRule summary:")
    print(f"  Hard rules: {len(rules['hard_rules'])}")
    print(f"  Soft rules: {len(rules['soft_rules'])}")
    print(f"  Meta rules: {len(rules['meta_rules'])}")
    print(f"  Learning rules: {len(rules['learning_rules'])}")
    print(f"  Output rules: {len(rules['output_rules'])}")
    print(f"  Tool rules: {len(rules['tool_rules'])}")


async def main():
    """Run all examples."""
    print("=" * 60)
    print("Constitutional AI Rules Engine - Usage Examples")
    print("=" * 60)

    try:
        # Example 1: Load and compile
        loader = await example_1_load_and_compile_rules()

        # Example 2: Enforce rules
        # NOTE: Requires Ollama running and qwen3:8b loaded
        # Commented out for demo purposes
        # engine, violations = await example_2_enforce_rules(loader)

        # Example 3: Learn from failures
        # NOTE: Requires violations from example 2
        # await example_3_learn_from_failures(engine, violations)

        # Example 4: Optimize prompts (no LLM required)
        await example_4_optimize_prompts()

        # Example 5: Monitor statistics
        await example_5_statistics_and_monitoring()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
