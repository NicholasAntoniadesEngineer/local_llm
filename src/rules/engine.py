"""Constitutional AI rules engine with critique-revise loop."""

import asyncio
import hashlib
from typing import Any

import structlog

from src.llm.router import ModelRouter
from src.llm.base import CompletionRequest

from .models import (
    Critique,
    HardRule,
    RuleViolation,
    ResponseType,
    SoftRule,
)
from .loader import RuleLoader

logger = structlog.get_logger(__name__)


class RulesEngine:
    """
    Enforce Constitutional AI rules via critique-revise loop.

    Features:
    - Parallel hard rule checking (early stop if any violated)
    - Sequential soft rule checking (by priority)
    - Response caching with >70% target hit rate
    - Max 3 revision attempts per response
    - LLM-based rule violation checking via qwen3:8b
    """

    def __init__(
        self,
        rules_path: str = "config/rules.yaml",
        model_router: ModelRouter | None = None,
        max_revisions: int = 3,
        cache_hit_target: float = 0.7,
    ):
        """
        Initialize rules engine.

        Args:
            rules_path: Path to rules YAML file
            model_router: ModelRouter instance for LLM calls
            max_revisions: Maximum revision attempts per response
            cache_hit_target: Target cache hit rate
        """
        self.loader = RuleLoader(rules_path)
        self.loader.load()
        self.loader.compile_to_xml()
        self.router = model_router
        self.max_revisions = max_revisions
        self.cache_hit_target = cache_hit_target
        self._critique_cache: dict[str, Critique] = {}
        self._stats = {
            "total_evaluations": 0,
            "cache_hits": 0,
            "revisions_triggered": 0,
            "violations_found": 0,
        }

    async def enforce(
        self,
        response_text: str,
        response_type: ResponseType = ResponseType.GENERAL,
    ) -> tuple[str, list[RuleViolation]]:
        """
        Enforce all rules on a response via critique-revise loop.

        Process:
        1. Run hard rules in parallel (blocking)
        2. If violations, attempt revision (max 3 times)
        3. Run soft rules sequentially by priority
        4. Return revised response and violation list

        Args:
            response_text: The response to evaluate
            response_type: Type of response being evaluated

        Returns:
            Tuple of (final_response_text, list_of_violations)
        """
        self._stats["total_evaluations"] += 1
        violations: list[RuleViolation] = []
        current_response = response_text
        revision_count = 0

        logger.info(
            "rules_enforcement_started",
            response_type=response_type.value,
            response_len=len(response_text),
        )

        # Phase 1: Check hard rules in parallel
        hard_rules = self.loader.get_hard_rules()
        if hard_rules:
            hard_violations = await self._check_hard_rules_parallel(
                current_response,
                hard_rules,
                response_type,
            )
            violations.extend(hard_violations)

            # Attempt revisions if hard rule violations found
            while hard_violations and revision_count < self.max_revisions:
                logger.info(
                    "attempting_revision",
                    violation_count=len(hard_violations),
                    attempt=revision_count + 1,
                )
                current_response = await self._revise_response(
                    current_response,
                    hard_violations,
                )
                revision_count += 1

                # Re-check hard rules
                hard_violations = await self._check_hard_rules_parallel(
                    current_response,
                    hard_rules,
                    response_type,
                )

                if not hard_violations:
                    logger.info("hard_rules_satisfied", after_revision=revision_count)
                    break

        # Phase 2: Check soft rules sequentially (by priority)
        soft_rules = self.loader.get_soft_rules()
        if soft_rules:
            # Sort by priority
            sorted_soft = sorted(
                soft_rules,
                key=lambda r: self._priority_rank(r.priority),
            )

            for soft_rule in sorted_soft:
                critique = await self._check_rule(
                    current_response,
                    soft_rule,
                    response_type,
                )

                if critique and critique.violates:
                    violations.append(
                        RuleViolation(
                            rule_id=soft_rule.id,
                            response_text=current_response,
                            response_type=response_type,
                            critique=critique,
                        )
                    )
                    self._stats["violations_found"] += 1
                    logger.warning(
                        "soft_rule_violation",
                        rule_id=soft_rule.id,
                        severity=critique.severity,
                    )

        logger.info(
            "rules_enforcement_completed",
            violations_found=len(violations),
            revisions_used=revision_count,
            final_len=len(current_response),
        )

        return current_response, violations

    async def _check_hard_rules_parallel(
        self,
        response_text: str,
        hard_rules: list[HardRule],
        response_type: ResponseType,
    ) -> list[RuleViolation]:
        """
        Check hard rules in parallel (blocking on violations).

        Args:
            response_text: Response to check
            hard_rules: List of hard rules
            response_type: Type of response

        Returns:
            List of violations found
        """
        tasks = [
            self._check_rule(response_text, rule, response_type)
            for rule in hard_rules
        ]

        critiques = await asyncio.gather(*tasks, return_exceptions=True)

        violations = []
        for i, critique in enumerate(critiques):
            if isinstance(critique, Exception):
                logger.warning(
                    "rule_check_exception",
                    rule_id=hard_rules[i].id,
                    error=str(critique),
                )
                continue

            if critique and critique.violates:
                violations.append(
                    RuleViolation(
                        rule_id=hard_rules[i].id,
                        response_text=response_text,
                        response_type=response_type,
                        critique=critique,
                    )
                )
                self._stats["violations_found"] += 1

        return violations

    async def _check_rule(
        self,
        response_text: str,
        rule: HardRule | SoftRule,
        response_type: ResponseType,
    ) -> Critique | None:
        """
        Check if response violates a rule using LLM.

        Uses qwen3:8b for fast rule critique. Caches results by
        response_hash + rule_id + response_type.

        Args:
            response_text: Response to check
            rule: Rule to evaluate against
            response_type: Type of response

        Returns:
            Critique object or None if check failed
        """
        # Check cache first
        response_hash = hashlib.sha256(response_text[:100].encode()).hexdigest()
        cached = self.loader.cache.get(
            response_hash,
            rule.id,
            response_type.value,
        )
        if cached:
            self._stats["cache_hits"] += 1
            logger.debug("critique_cache_hit", rule_id=rule.id)
            return Critique.model_validate_json(cached)

        if not self.router:
            logger.warning("router_not_set", skipping_critique=True)
            return None

        # Build critique prompt
        critique_prompt = self._build_critique_prompt(
            response_text,
            rule,
            response_type,
        )

        try:
            response = await self.router.complete(
                role="orchestrate",
                prompt=critique_prompt,
                temperature=0.1,
                max_tokens=500,
            )

            # Parse critique from response
            critique = self._parse_critique_response(
                response.text,
                rule.id,
            )

            if critique:
                # Cache the result
                self.loader.cache.set(
                    response_hash,
                    rule.id,
                    response_type.value,
                    critique.model_dump_json(),
                )

            return critique

        except Exception as e:
            logger.error(
                "critique_generation_failed",
                rule_id=rule.id,
                error=str(e),
            )
            return None

    def _build_critique_prompt(
        self,
        response_text: str,
        rule: HardRule | SoftRule,
        response_type: ResponseType,
    ) -> str:
        """Build prompt for LLM to critique response against rule."""
        return f"""You are a rule enforcement checker. Evaluate if the following response violates the given rule.

Rule ID: {rule.id}
Rule: {rule.rule}
Rationale: {rule.rationale}

Response Type: {response_type.value}
Response:
{response_text[:1000]}

Respond in JSON format:
{{
    "violates": true/false,
    "explanation": "...",
    "severity": "low/medium/high",
    "confidence": 0.0-1.0,
    "suggestions": ["..."]
}}

Be strict but fair. Only flag clear violations."""

    def _parse_critique_response(self, response_text: str, rule_id: str) -> Critique | None:
        """Parse LLM response into Critique object."""
        import json
        import re

        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            logger.warning("critique_json_not_found", rule_id=rule_id)
            return None

        try:
            data = json.loads(json_match.group())
            return Critique(
                rule_id=rule_id,
                violates=data.get("violates", False),
                explanation=data.get("explanation", ""),
                severity=data.get("severity", "medium"),
                confidence=data.get("confidence", 0.5),
                suggestions=data.get("suggestions", []),
            )
        except Exception as e:
            logger.warning("critique_parse_failed", rule_id=rule_id, error=str(e))
            return None

    async def _revise_response(
        self,
        response_text: str,
        violations: list[RuleViolation],
    ) -> str:
        """
        Revise response to address violations.

        Args:
            response_text: Original response
            violations: List of violations to address

        Returns:
            Revised response text
        """
        if not self.router:
            logger.warning("router_not_set_for_revision")
            return response_text

        violation_summaries = "\n".join(
            [
                f"- {v.rule_id}: {v.critique.explanation}"
                for v in violations
            ]
        )

        revision_prompt = f"""Revise the following response to address these violations:

{violation_summaries}

Original response:
{response_text}

Provide a revised response that addresses all violations while maintaining the core message."""

        try:
            response = await self.router.complete(
                role="reason",
                prompt=revision_prompt,
                temperature=0.3,
                max_tokens=2000,
            )
            logger.info("response_revised", original_len=len(response_text), revised_len=len(response.text))
            return response.text
        except Exception as e:
            logger.error("revision_failed", error=str(e))
            return response_text

    def _priority_rank(self, priority: Any) -> int:
        """Get rank for priority ordering (lower = higher priority)."""
        rank_map = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        return rank_map.get(priority.value if hasattr(priority, "value") else str(priority), 2)

    def get_statistics(self) -> dict[str, Any]:
        """Get enforcement statistics."""
        cache_hit_rate = (
            self._stats["cache_hits"] / self._stats["total_evaluations"]
            if self._stats["total_evaluations"] > 0
            else 0.0
        )
        return {
            **self._stats,
            "cache_hit_rate": cache_hit_rate,
            "cache_hit_target": self.cache_hit_target,
            "cache_size": len(self.loader.cache.entries),
        }

    def reset_statistics(self) -> None:
        """Reset enforcement statistics."""
        self._stats = {
            "total_evaluations": 0,
            "cache_hits": 0,
            "revisions_triggered": 0,
            "violations_found": 0,
        }
