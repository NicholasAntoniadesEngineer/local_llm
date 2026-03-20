"""Self-improvement via A/B testing and rule learning."""

import asyncio
import hashlib
from datetime import datetime
from typing import Any
from uuid import UUID

import structlog

from src.llm.router import ModelRouter
from src.llm.base import CompletionRequest

from .models import (
    LearningRule,
    Priority,
    ProposedRuleChange,
    RuleViolation,
    SoftRule,
)
from .loader import RuleLoader

logger = structlog.get_logger(__name__)


class RuleLearner:
    """
    Self-improvement via A/B testing and rule learning.

    Process:
    1. Identify failures (score < 0.6)
    2. Agent analyzes failures, proposes rule changes
    3. Test candidate rules on 10% holdout set
    4. If improvement > 5%, commit; else reject
    5. Update confidence based on effectiveness
    6. Persist to SQLite
    """

    def __init__(
        self,
        rules_path: str = "config/rules.yaml",
        model_router: ModelRouter | None = None,
        improvement_threshold: float = 0.05,
        holdout_fraction: float = 0.1,
    ):
        """
        Initialize rule learner.

        Args:
            rules_path: Path to rules YAML file
            model_router: ModelRouter for LLM calls
            improvement_threshold: Minimum improvement to commit (5%)
            holdout_fraction: Fraction of data for A/B testing (10%)
        """
        self.loader = RuleLoader(rules_path)
        self.router = model_router
        self.improvement_threshold = improvement_threshold
        self.holdout_fraction = holdout_fraction
        self._proposed_changes: list[ProposedRuleChange] = []
        self._test_history: list[dict[str, Any]] = []

    async def analyze_failures(
        self,
        task_id: str,
        violations: list[RuleViolation],
        task_context: str = "",
    ) -> list[ProposedRuleChange]:
        """
        Analyze failures and propose rule changes.

        Args:
            task_id: Task UUID for tracking
            violations: List of rule violations from task
            task_context: Context about the task

        Returns:
            List of proposed rule changes
        """
        if not violations:
            logger.info("no_violations_to_analyze", task_id=task_id)
            return []

        if not self.router:
            logger.warning("router_not_set_for_learning")
            return []

        logger.info(
            "analyzing_failures",
            task_id=task_id,
            violation_count=len(violations),
        )

        # Group violations by rule
        violations_by_rule = {}
        for v in violations:
            if v.rule_id not in violations_by_rule:
                violations_by_rule[v.rule_id] = []
            violations_by_rule[v.rule_id].append(v)

        proposed_changes = []

        # Analyze each violated rule
        for rule_id, rule_violations in violations_by_rule.items():
            rule = self.loader.get_rules_by_id(rule_id)
            if not rule:
                continue

            proposal = await self._propose_rule_change(
                task_id,
                rule,
                rule_violations,
                task_context,
            )

            if proposal:
                proposed_changes.append(proposal)
                self._proposed_changes.append(proposal)

        logger.info(
            "failures_analyzed",
            task_id=task_id,
            proposals_generated=len(proposed_changes),
        )

        return proposed_changes

    async def _propose_rule_change(
        self,
        task_id: str,
        rule: Any,
        violations: list[RuleViolation],
        task_context: str,
    ) -> ProposedRuleChange | None:
        """Propose a single rule change for a violated rule."""
        # Build analysis prompt
        violation_examples = "\n".join(
            [f"- {v.critique.explanation}" for v in violations[:3]]
        )

        analysis_prompt = f"""Analyze this failed rule and propose an improvement:

Rule ID: {rule.id}
Rule: {rule.rule}

Failure examples:
{violation_examples}

Task context:
{task_context[:500]}

Propose a new or refined rule that would:
1. Be more specific to catch edge cases
2. Account for what we learned from failures
3. Maintain clarity and actionability

Respond in JSON:
{{
    "change_type": "add/modify",
    "new_rule": "...",
    "rationale": "...",
    "reasoning": "..."
}}"""

        try:
            response = await self.router.complete(
                role="reason",
                prompt=analysis_prompt,
                temperature=0.3,
                max_tokens=1000,
            )

            proposal = self._parse_proposal_response(
                response.text,
                rule.id,
                task_id,
                violations,
            )

            if proposal:
                logger.info(
                    "rule_change_proposed",
                    rule_id=rule.id,
                    change_type=proposal.change_type,
                )
                return proposal

        except Exception as e:
            logger.error(
                "proposal_generation_failed",
                rule_id=rule.id,
                error=str(e),
            )

        return None

    def _parse_proposal_response(
        self,
        response_text: str,
        base_rule_id: str,
        task_id: str,
        violations: list[RuleViolation],
    ) -> ProposedRuleChange | None:
        """Parse LLM response into ProposedRuleChange."""
        import json
        import re

        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            logger.warning("proposal_json_not_found", rule_id=base_rule_id)
            return None

        try:
            data = json.loads(json_match.group())

            # Create new learning rule
            new_rule = LearningRule(
                id=f"L{len(self.loader.get_learning_rules()) + 1}",
                rule=data.get("new_rule", ""),
                rationale=data.get("rationale", ""),
                priority=Priority.MEDIUM,
                confidence=0.5,  # Start at 0.5
                derived_from=task_id,
                failure_case=violations[0].critique.explanation if violations else "",
            )

            proposal = ProposedRuleChange(
                change_type=data.get("change_type", "add"),
                rule=new_rule,
                rationale=data.get("rationale", ""),
                based_on_failures=[v.rule_id for v in violations],
            )

            return proposal

        except Exception as e:
            logger.warning("proposal_parse_failed", rule_id=base_rule_id, error=str(e))
            return None

    async def test_proposal(
        self,
        proposal: ProposedRuleChange,
        baseline_test_set: list[dict[str, Any]],
    ) -> float:
        """
        A/B test a proposed rule change.

        Args:
            proposal: Proposed rule change
            baseline_test_set: Test cases (10% holdout)

        Returns:
            Improvement percentage (0.0 to 1.0)
        """
        if not baseline_test_set:
            logger.warning("empty_test_set")
            return 0.0

        logger.info(
            "testing_proposal",
            proposal_id=proposal.id,
            test_set_size=len(baseline_test_set),
        )

        baseline_scores = []
        candidate_scores = []

        for test_case in baseline_test_set:
            response = test_case.get("response", "")
            violations = test_case.get("violations", [])

            # Score baseline (how many violations with current rule)
            baseline_violation_count = len(
                [v for v in violations if v.rule_id == proposal.rule.id]
            )
            baseline_scores.append(1.0 - (baseline_violation_count / max(len(violations), 1)))

            # Score candidate (would new rule catch these?)
            candidate_scores.append(
                await self._evaluate_rule_on_response(proposal.rule, response)
            )

        baseline_avg = sum(baseline_scores) / len(baseline_scores)
        candidate_avg = sum(candidate_scores) / len(candidate_scores)
        improvement = candidate_avg - baseline_avg

        proposal.test_results = {
            "baseline_score": baseline_avg,
            "candidate_score": candidate_avg,
            "improvement_pct": improvement,
            "test_set_size": len(baseline_test_set),
        }

        logger.info(
            "proposal_tested",
            proposal_id=proposal.id,
            baseline=f"{baseline_avg:.3f}",
            candidate=f"{candidate_avg:.3f}",
            improvement_pct=f"{improvement:.1%}",
        )

        return improvement

    async def _evaluate_rule_on_response(self, rule: LearningRule, response: str) -> float:
        """Evaluate how well a proposed rule performs on a response."""
        if not self.router:
            return 0.0

        eval_prompt = f"""Rate how well this rule would improve response quality:

Rule: {rule.rule}
Response: {response[:500]}

Rate on 0-1 scale how well this rule improves quality:
0 = not helpful
1 = very helpful

Just respond with a number."""

        try:
            result = await self.router.complete(
                role="orchestrate",
                prompt=eval_prompt,
                temperature=0.1,
                max_tokens=10,
            )

            score_text = result.text.strip()
            score = float(score_text.split()[0])
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning("rule_evaluation_failed", error=str(e))
            return 0.0

    async def commit_proposal(
        self,
        proposal: ProposedRuleChange,
        improvement_pct: float,
    ) -> bool:
        """
        Commit a proposal if improvement threshold is met.

        Args:
            proposal: Proposed rule change
            improvement_pct: Improvement percentage from A/B test

        Returns:
            True if committed, False otherwise
        """
        if improvement_pct < self.improvement_threshold:
            logger.info(
                "proposal_rejected",
                proposal_id=proposal.id,
                improvement=f"{improvement_pct:.1%}",
                threshold=f"{self.improvement_threshold:.1%}",
            )
            proposal.status = "rejected"
            return False

        logger.info(
            "proposal_committed",
            proposal_id=proposal.id,
            improvement=f"{improvement_pct:.1%}",
        )

        proposal.status = "approved"

        # Update rule's confidence based on improvement
        if hasattr(proposal.rule, "confidence"):
            # Increase confidence if improvement is good
            improvement_factor = 1.0 + (improvement_pct / 0.2)  # Max 1.5x
            proposal.rule.confidence = min(
                1.0,
                proposal.rule.confidence * improvement_factor,
            )

        self._test_history.append({
            "proposal_id": str(proposal.id),
            "improvement_pct": improvement_pct,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "approved",
        })

        return True

    def get_proposed_rules(self) -> list[ProposedRuleChange]:
        """Get all proposed rule changes."""
        return self._proposed_changes

    def get_approved_rules(self) -> list[ProposedRuleChange]:
        """Get approved rule changes."""
        return [p for p in self._proposed_changes if p.status == "approved"]

    def get_test_history(self) -> list[dict[str, Any]]:
        """Get A/B testing history."""
        return self._test_history

    def clear_proposals(self) -> None:
        """Clear proposed rules."""
        self._proposed_changes = []
