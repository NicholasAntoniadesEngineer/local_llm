"""YAML rule loader with validation, compilation, and hot-reload support."""

import hashlib
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import structlog
import yaml

from .models import (
    HardRule,
    MetaRule,
    SoftRule,
    LearningRule,
    Priority,
    Enforcement,
)

logger = structlog.get_logger(__name__)


class RuleCache:
    """Cache for compiled rule XML and validation results."""

    def __init__(self, ttl_hours: int = 24):
        """
        Initialize rule cache.

        Args:
            ttl_hours: Time-to-live for cache entries in hours
        """
        self.ttl = timedelta(hours=ttl_hours)
        self.entries: dict[str, tuple[str, datetime]] = {}  # {cache_key: (value, timestamp)}

    def _generate_key(
        self,
        response_hash: str,
        rule_id: str,
        response_type: str,
    ) -> str:
        """Generate cache key from response hash, rule ID, and type."""
        return f"{response_hash}_{rule_id}_{response_type}"

    def get(self, response_hash: str, rule_id: str, response_type: str) -> str | None:
        """Get cached value if exists and not expired."""
        key = self._generate_key(response_hash, rule_id, response_type)
        if key in self.entries:
            value, timestamp = self.entries[key]
            if datetime.utcnow() - timestamp < self.ttl:
                return value
            else:
                del self.entries[key]
        return None

    def set(self, response_hash: str, rule_id: str, response_type: str, value: str) -> None:
        """Cache a value."""
        key = self._generate_key(response_hash, rule_id, response_type)
        self.entries[key] = (value, datetime.utcnow())

    def hit_rate(self) -> float:
        """Get cache hit rate (placeholder for full implementation)."""
        # Tracked in engine, but cache size can indicate quality
        return min(len(self.entries) / 1000, 1.0)  # Rough estimate


class RuleLoader:
    """
    Load, validate, and compile rules from YAML.

    Features:
    - Parse config/rules.yaml with validation
    - Unique ID enforcement
    - Confidence bounds checking
    - XML compilation for rule blocks
    - Hot-reload capability
    - Cache management
    """

    def __init__(self, rules_path: str = "config/rules.yaml"):
        """
        Initialize rule loader.

        Args:
            rules_path: Path to rules YAML file
        """
        self.rules_path = Path(rules_path)
        self.cache = RuleCache()
        self._loaded_rules: dict[str, Any] = {}
        self._compiled_xml: str = ""
        self._file_hash = ""

    def load(self) -> dict[str, list[HardRule | SoftRule | MetaRule | LearningRule]]:
        """
        Load and validate rules from YAML file.

        Returns:
            Dictionary with keys: 'meta_rules', 'hard_rules', 'soft_rules', 'learning_rules'

        Raises:
            FileNotFoundError: If rules file not found
            ValueError: If validation fails
        """
        if not self.rules_path.exists():
            logger.error("rules_file_not_found", path=str(self.rules_path))
            raise FileNotFoundError(f"Rules file not found: {self.rules_path}")

        try:
            with open(self.rules_path, "r") as f:
                raw_rules = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error("yaml_parse_error", path=str(self.rules_path), error=str(e))
            raise ValueError(f"Failed to parse rules YAML: {e}")

        # Validate structure
        self._validate_structure(raw_rules)

        # Parse each rule type
        parsed_rules = {
            "meta_rules": self._parse_meta_rules(raw_rules.get("meta_rules", [])),
            "hard_rules": self._parse_hard_rules(raw_rules.get("hard_rules", [])),
            "soft_rules": self._parse_soft_rules(raw_rules.get("soft_rules", [])),
            "learning_rules": self._parse_learning_rules(raw_rules.get("learning_rules", [])),
            "output_rules": self._parse_output_rules(raw_rules.get("output_rules", [])),
            "tool_rules": self._parse_tool_rules(raw_rules.get("tool_rules", [])),
        }

        # Validate unique IDs across all rule types
        self._validate_unique_ids(parsed_rules)

        self._loaded_rules = parsed_rules
        logger.info(
            "rules_loaded",
            hard_count=len(parsed_rules["hard_rules"]),
            soft_count=len(parsed_rules["soft_rules"]),
            learning_count=len(parsed_rules["learning_rules"]),
            meta_count=len(parsed_rules["meta_rules"]),
        )

        return parsed_rules

    def _validate_structure(self, raw_rules: dict[str, Any]) -> None:
        """Validate YAML structure."""
        if not isinstance(raw_rules, dict):
            raise ValueError("Rules file must be a dictionary")

        required_sections = ["hard_rules", "soft_rules"]
        for section in required_sections:
            if section not in raw_rules:
                raise ValueError(f"Missing required section: {section}")

    def _parse_meta_rules(self, rules_list: list[dict[str, Any]]) -> list[MetaRule]:
        """Parse meta-rules from YAML."""
        parsed = []
        for rule_data in rules_list:
            try:
                rule = MetaRule(
                    id=rule_data["id"],
                    rule=rule_data["rule"],
                    rationale=rule_data.get("rationale", ""),
                    priority=Priority(rule_data.get("priority", "medium")),
                )
                parsed.append(rule)
            except Exception as e:
                logger.warning("meta_rule_parse_failed", rule_id=rule_data.get("id"), error=str(e))
        return parsed

    def _parse_hard_rules(self, rules_list: list[dict[str, Any]]) -> list[HardRule]:
        """Parse hard rules from YAML."""
        parsed = []
        for rule_data in rules_list:
            try:
                rule = HardRule(
                    id=rule_data["id"],
                    rule=rule_data["rule"],
                    rationale=rule_data.get("rationale", ""),
                    priority=Priority(rule_data.get("priority", "high")),
                    enforcement=Enforcement(rule_data.get("enforcement", "block_if_violated")),
                    immutable=rule_data.get("immutable", True),
                )
                parsed.append(rule)
            except Exception as e:
                logger.warning("hard_rule_parse_failed", rule_id=rule_data.get("id"), error=str(e))
        return parsed

    def _parse_soft_rules(self, rules_list: list[dict[str, Any]]) -> list[SoftRule]:
        """Parse soft rules from YAML."""
        parsed = []
        for rule_data in rules_list:
            try:
                # Validate confidence is in [0, 1]
                confidence = rule_data.get("confidence", 0.5)
                if not (0.0 <= confidence <= 1.0):
                    raise ValueError(f"Confidence must be in [0,1], got {confidence}")

                rule = SoftRule(
                    id=rule_data["id"],
                    rule=rule_data["rule"],
                    rationale=rule_data.get("rationale", ""),
                    priority=Priority(rule_data.get("priority", "medium")),
                    confidence=confidence,
                    effectiveness_score=rule_data.get("effectiveness_score", 0.0),
                    derived_from=rule_data.get("derived_from"),
                )
                parsed.append(rule)
            except Exception as e:
                logger.warning("soft_rule_parse_failed", rule_id=rule_data.get("id"), error=str(e))
        return parsed

    def _parse_learning_rules(self, rules_list: list[dict[str, Any]]) -> list[LearningRule]:
        """Parse learning rules from YAML."""
        parsed = []
        for rule_data in rules_list:
            try:
                confidence = rule_data.get("confidence", 0.5)
                if not (0.0 <= confidence <= 1.0):
                    raise ValueError(f"Confidence must be in [0,1], got {confidence}")

                rule = LearningRule(
                    id=rule_data["id"],
                    rule=rule_data["rule"],
                    rationale=rule_data.get("rationale", ""),
                    priority=Priority(rule_data.get("priority", "medium")),
                    confidence=confidence,
                    effectiveness_score=rule_data.get("effectiveness_score", 0.0),
                    derived_from=rule_data["derived_from"],
                    failure_case=rule_data.get("failure_case", ""),
                )
                parsed.append(rule)
            except Exception as e:
                logger.warning("learning_rule_parse_failed", rule_id=rule_data.get("id"), error=str(e))
        return parsed

    def _parse_output_rules(self, rules_list: list[dict[str, Any]]) -> list[MetaRule]:
        """Parse output formatting rules as meta-rules."""
        parsed = []
        for rule_data in rules_list:
            try:
                rule = MetaRule(
                    id=rule_data["id"],
                    rule=rule_data["rule"],
                    rationale=rule_data.get("rationale", ""),
                    priority=Priority(rule_data.get("priority", "medium")),
                )
                parsed.append(rule)
            except Exception as e:
                logger.warning("output_rule_parse_failed", rule_id=rule_data.get("id"), error=str(e))
        return parsed

    def _parse_tool_rules(self, rules_list: list[dict[str, Any]]) -> list[MetaRule]:
        """Parse tool usage rules as meta-rules."""
        parsed = []
        for rule_data in rules_list:
            try:
                rule = MetaRule(
                    id=rule_data["id"],
                    rule=rule_data["rule"],
                    rationale=rule_data.get("rationale", ""),
                    priority=Priority(rule_data.get("priority", "medium")),
                )
                parsed.append(rule)
            except Exception as e:
                logger.warning("tool_rule_parse_failed", rule_id=rule_data.get("id"), error=str(e))
        return parsed

    def _validate_unique_ids(self, parsed_rules: dict[str, list[Any]]) -> None:
        """Ensure all rule IDs are unique."""
        all_ids = []
        for rule_list in parsed_rules.values():
            for rule in rule_list:
                all_ids.append(rule.id)

        if len(all_ids) != len(set(all_ids)):
            duplicates = [id for id in set(all_ids) if all_ids.count(id) > 1]
            raise ValueError(f"Duplicate rule IDs: {duplicates}")

    def compile_to_xml(self) -> str:
        """
        Compile rules to XML block format for inclusion in system prompts.

        Returns:
            XML string containing all rules
        """
        if not self._loaded_rules:
            self.load()

        root = ET.Element("rules")

        # Add meta-rules section
        meta_elem = ET.SubElement(root, "meta_rules")
        for rule in self._loaded_rules.get("meta_rules", []):
            self._rule_to_xml(meta_elem, rule)

        # Add hard-rules section
        hard_elem = ET.SubElement(root, "hard_rules")
        hard_elem.set("enforcement", "block_if_violated")
        for rule in self._loaded_rules.get("hard_rules", []):
            self._rule_to_xml(hard_elem, rule)

        # Add soft-rules section
        soft_elem = ET.SubElement(root, "soft_rules")
        for rule in self._loaded_rules.get("soft_rules", []):
            rule_elem = self._rule_to_xml(soft_elem, rule)
            if isinstance(rule, SoftRule):
                rule_elem.set("confidence", f"{rule.confidence:.2f}")
                rule_elem.set("priority", rule.priority.value)

        # Add learning-rules section
        learning_elem = ET.SubElement(root, "learning_rules")
        for rule in self._loaded_rules.get("learning_rules", []):
            rule_elem = self._rule_to_xml(learning_elem, rule)
            if isinstance(rule, LearningRule):
                rule_elem.set("confidence", f"{rule.confidence:.2f}")
                rule_elem.set("derived_from", rule.derived_from)

        self._compiled_xml = ET.tostring(root, encoding="unicode")
        logger.info("rules_compiled_to_xml", size_bytes=len(self._compiled_xml))
        return self._compiled_xml

    def _rule_to_xml(self, parent: ET.Element, rule: Any) -> ET.Element:
        """Convert a single rule to XML element."""
        elem = ET.SubElement(parent, "rule")
        elem.set("id", rule.id)

        rule_text = ET.SubElement(elem, "text")
        rule_text.text = rule.rule

        if rule.rationale:
            rationale = ET.SubElement(elem, "rationale")
            rationale.text = rule.rationale

        return elem

    def hot_reload(self) -> bool:
        """
        Check if rules file changed and reload if necessary.

        Returns:
            True if rules were reloaded, False otherwise
        """
        if not self.rules_path.exists():
            logger.warning("hot_reload_file_not_found", path=str(self.rules_path))
            return False

        with open(self.rules_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        if file_hash != self._file_hash:
            logger.info("rules_reloading", path=str(self.rules_path))
            self._file_hash = file_hash
            self.load()
            self.compile_to_xml()
            return True

        return False

    def get_compiled_xml(self) -> str:
        """Get last compiled XML, compile if needed."""
        if not self._compiled_xml:
            self.compile_to_xml()
        return self._compiled_xml

    def get_rules_by_id(self, rule_id: str) -> Any:
        """Get a specific rule by ID."""
        for rule_list in self._loaded_rules.values():
            for rule in rule_list:
                if rule.id == rule_id:
                    return rule
        return None

    def get_hard_rules(self) -> list[HardRule]:
        """Get all hard rules."""
        return self._loaded_rules.get("hard_rules", [])

    def get_soft_rules(self) -> list[SoftRule]:
        """Get all soft rules."""
        return self._loaded_rules.get("soft_rules", [])

    def get_learning_rules(self) -> list[LearningRule]:
        """Get all learning rules."""
        return self._loaded_rules.get("learning_rules", [])

    def get_cache(self) -> RuleCache:
        """Get cache object."""
        return self.cache
