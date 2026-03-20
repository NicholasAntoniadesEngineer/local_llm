"""DSPy prompt optimization for rule critique and learning."""

import json
from pathlib import Path
from typing import Any, Callable

import structlog
from jinja2 import Environment, FileSystemLoader, Template

from src.llm.router import ModelRouter

logger = structlog.get_logger(__name__)


class RuleOptimizer:
    """
    DSPy-style prompt optimization for rule critique prompts.

    Features:
    - BootstrapFewShot optimizer for rule critique examples
    - Jinja2 template rendering
    - Metric definitions for optimization
    - Saves optimized examples to config/prompts/
    """

    def __init__(
        self,
        prompts_dir: str = "config/prompts",
        model_router: ModelRouter | None = None,
    ):
        """
        Initialize optimizer.

        Args:
            prompts_dir: Directory for prompt templates
            model_router: ModelRouter for LLM calls
        """
        self.prompts_dir = Path(prompts_dir)
        self.router = model_router
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self.env = Environment(loader=FileSystemLoader(str(self.prompts_dir)))
        self._optimized_examples: list[dict[str, Any]] = []
        self._metrics: dict[str, Callable] = {}

    def register_metric(
        self,
        name: str,
        metric_fn: Callable[[Any, Any], float],
    ) -> None:
        """
        Register a metric function for optimization.

        Args:
            name: Name of metric (e.g., "critique_accuracy")
            metric_fn: Function that scores a result (0-1)
        """
        self._metrics[name] = metric_fn
        logger.info("metric_registered", metric_name=name)

    async def optimize_critique_prompt(
        self,
        training_examples: list[dict[str, Any]],
        metric_name: str = "critique_accuracy",
        num_shots: int = 3,
    ) -> str:
        """
        Optimize critique prompt using BootstrapFewShot.

        Args:
            training_examples: List of {rule, response, critique} examples
            metric_name: Metric to optimize for
            num_shots: Number of examples to include

        Returns:
            Optimized prompt template
        """
        if not self.router:
            logger.warning("router_not_set_for_optimization")
            return ""

        if metric_name not in self._metrics:
            logger.warning("metric_not_registered", metric_name=metric_name)
            return ""

        logger.info(
            "optimizing_critique_prompt",
            num_examples=len(training_examples),
            metric=metric_name,
            num_shots=num_shots,
        )

        # Score each example
        scored_examples = []
        for example in training_examples:
            try:
                score = self._metrics[metric_name](example)
                scored_examples.append((example, score))
            except Exception as e:
                logger.warning("example_scoring_failed", error=str(e))
                continue

        # Sort by score and take top examples
        scored_examples.sort(key=lambda x: x[1], reverse=True)
        best_examples = [ex for ex, _ in scored_examples[:num_shots]]

        # Format examples into prompt
        optimized_prompt = self._format_critique_examples(best_examples)

        # Save optimized examples
        self._save_optimized_examples(best_examples, "critique")

        logger.info(
            "critique_prompt_optimized",
            best_examples_count=len(best_examples),
        )

        return optimized_prompt

    def _format_critique_examples(
        self,
        examples: list[dict[str, Any]],
    ) -> str:
        """Format examples into prompt with few-shot examples."""
        prompt = """You are a rule enforcement checker. Here are examples of good critiques:

"""
        for i, example in enumerate(examples, 1):
            rule = example.get("rule", {})
            response = example.get("response", "")[:200]
            critique = example.get("critique", {})

            prompt += f"""Example {i}:
Rule: {rule.get("rule", "")}
Response: {response}...
Critique:
- Violates: {critique.get("violates", False)}
- Severity: {critique.get("severity", "medium")}
- Explanation: {critique.get("explanation", "")}

"""

        prompt += """Now evaluate the given rule and response in the same format."""
        return prompt

    async def optimize_proposal_prompt(
        self,
        training_examples: list[dict[str, Any]],
        metric_name: str = "proposal_quality",
    ) -> str:
        """
        Optimize rule proposal prompt.

        Args:
            training_examples: List of {rule, violations, proposal} examples
            metric_name: Metric to optimize for

        Returns:
            Optimized prompt template
        """
        if metric_name not in self._metrics:
            logger.warning("metric_not_registered", metric_name=metric_name)
            return ""

        logger.info(
            "optimizing_proposal_prompt",
            num_examples=len(training_examples),
        )

        # Score examples
        scored_examples = []
        for example in training_examples:
            try:
                score = self._metrics[metric_name](example)
                scored_examples.append((example, score))
            except Exception as e:
                logger.warning("example_scoring_failed", error=str(e))
                continue

        # Sort and take top examples
        scored_examples.sort(key=lambda x: x[1], reverse=True)
        best_examples = [ex for ex, _ in scored_examples[:3]]

        self._save_optimized_examples(best_examples, "proposal")

        logger.info("proposal_prompt_optimized", examples_count=len(best_examples))

        return self._format_proposal_examples(best_examples)

    def _format_proposal_examples(
        self,
        examples: list[dict[str, Any]],
    ) -> str:
        """Format examples for rule proposal."""
        prompt = """You are proposing rule improvements. Here are examples of good proposals:

"""
        for i, example in enumerate(examples, 1):
            rule = example.get("rule", {})
            violations = example.get("violations", [])
            proposal = example.get("proposal", {})

            prompt += f"""Example {i}:
Original Rule: {rule.get("rule", "")}
Violations:
{chr(10).join([f"  - {v.get('explanation', '')}" for v in violations[:2]])}

Proposed Rule: {proposal.get("new_rule", "")}
Rationale: {proposal.get("rationale", "")}

"""

        prompt += """Now analyze the given failures and propose an improvement."""
        return prompt

    def _save_optimized_examples(
        self,
        examples: list[dict[str, Any]],
        example_type: str,
    ) -> None:
        """Save optimized examples to disk."""
        output_file = self.prompts_dir / f"optimized_{example_type}_examples.json"
        try:
            with open(output_file, "w") as f:
                json.dump(examples, f, indent=2, default=str)
            logger.info("optimized_examples_saved", file=str(output_file))
        except Exception as e:
            logger.warning("failed_to_save_examples", error=str(e))

    def load_template(self, template_name: str) -> Template | None:
        """
        Load a Jinja2 template from prompts directory.

        Args:
            template_name: Name of template file (e.g., "rule_critique.j2")

        Returns:
            Jinja2 Template or None if not found
        """
        try:
            return self.env.get_template(template_name)
        except Exception as e:
            logger.warning("template_not_found", template=template_name, error=str(e))
            return None

    def render_template(
        self,
        template_name: str,
        context: dict[str, Any],
    ) -> str:
        """
        Render a Jinja2 template with context.

        Args:
            template_name: Name of template file
            context: Variables for template rendering

        Returns:
            Rendered template string
        """
        template = self.load_template(template_name)
        if not template:
            logger.warning("template_render_failed", template=template_name)
            return ""

        try:
            return template.render(context)
        except Exception as e:
            logger.error("template_rendering_failed", template=template_name, error=str(e))
            return ""

    def create_default_templates(self) -> None:
        """Create default prompt templates if they don't exist."""
        templates = {
            "critique.j2": """You are a rule enforcement checker.

Rule: {{ rule }}
Rationale: {{ rationale }}

Response Type: {{ response_type }}
Response: {{ response[:1000] }}

Evaluate if the response violates the rule.
Respond in JSON with: violates (bool), explanation, severity, confidence, suggestions (list)""",

            "proposal.j2": """Analyze failures in rule enforcement and propose improvements.

Rule ID: {{ rule_id }}
Rule: {{ rule }}

Failure examples:
{% for violation in violations %}
- {{ violation }}
{% endfor %}

Propose a refined rule that handles edge cases better.
Respond in JSON with: change_type, new_rule, rationale""",

            "system.j2": """You are a Constitutional AI rule enforcement system.
Your role is to evaluate responses against a set of hard and soft rules.
Be consistent, fair, and explain your reasoning clearly.""",

            "reflect.j2": """Review your recent work and identify patterns:
- Which rules are violated most often?
- Are there edge cases not covered?
- Should any rules be refined or added?

Task results: {{ task_results }}

Propose improvements based on observed patterns.""",
        }

        for filename, content in templates.items():
            template_path = self.prompts_dir / filename
            if not template_path.exists():
                try:
                    with open(template_path, "w") as f:
                        f.write(content)
                    logger.info("default_template_created", file=filename)
                except Exception as e:
                    logger.warning("failed_to_create_template", file=filename, error=str(e))

    def get_optimized_examples(self) -> list[dict[str, Any]]:
        """Get all optimized examples."""
        return self._optimized_examples
