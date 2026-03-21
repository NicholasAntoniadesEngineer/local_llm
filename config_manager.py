"""Configuration management and evolution for self-improving agents."""

import json
import subprocess
from pathlib import Path
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional

from config import CONFIG, AgentConfig


@dataclass
class ConfigVersion:
    """Tracked configuration version with performance metrics."""

    version_id: str  # e.g., "v0", "v1", "v2"
    timestamp: str  # ISO format creation time
    config_file: Path  # Path to config_vX.py
    parent_version: Optional[str]  # e.g., "v0" for improvements
    description: str  # What changed
    performance_score: float = 0.0  # 0-1, higher is better
    test_results: dict = None  # Results from A/B testing

    def __post_init__(self):
        """Initialize test_results dict."""
        if self.test_results is None:
            self.test_results = {}


class ConfigManager:
    """Manages config versioning, evolution, and A/B testing."""

    def __init__(self, versions_dir: Path = Path("./.claude/config_versions")):
        """Initialize config manager.

        Args:
            versions_dir: Directory to store config versions
        """
        self.versions_dir = versions_dir
        self.versions_dir.mkdir(parents=True, exist_ok=True)

        self.history_file = self.versions_dir / "history.json"
        self.versions: dict[str, ConfigVersion] = self._load_history()

    def _load_history(self) -> dict[str, ConfigVersion]:
        """Load config version history from disk."""
        if not self.history_file.exists():
            return {}

        try:
            with open(self.history_file, "r") as f:
                data = json.load(f)

            versions = {}
            for vid, v in data.items():
                # Convert config_file string back to Path
                v["config_file"] = Path(v["config_file"])
                versions[vid] = ConfigVersion(**v)
            return versions
        except Exception as e:
            print(f"⚠️  Could not load config history: {e}")
            return {}

    def _save_history(self) -> None:
        """Persist config history to disk."""
        data = {}
        for vid, v in self.versions.items():
            v_dict = asdict(v)
            # Convert Path objects to strings for JSON serialization
            v_dict["config_file"] = str(v_dict["config_file"])
            data[vid] = v_dict

        with open(self.history_file, "w") as f:
            json.dump(data, f, indent=2)

    def create_version(
        self,
        description: str,
        config_changes: dict,
        parent_version: Optional[str] = None,
    ) -> str:
        """Create a new config version.

        Args:
            description: What changed in this version
            config_changes: Dict of config parameters to modify
            parent_version: Version this is based on (for tracking evolution)

        Returns:
            New version ID (e.g., "v1")
        """
        # Generate version ID
        next_num = len(self.versions)
        version_id = f"v{next_num}"

        # Generate config file
        config_file = self.versions_dir / f"config_{version_id}.py"

        # Write new config (modifying existing CONFIG)
        new_config_text = self._generate_config_file(config_changes)
        with open(config_file, "w") as f:
            f.write(new_config_text)

        # Track version
        version = ConfigVersion(
            version_id=version_id,
            timestamp=datetime.now().isoformat(),
            config_file=config_file,
            parent_version=parent_version,
            description=description,
        )

        self.versions[version_id] = version
        self._save_history()

        return version_id

    def _generate_config_file(self, changes: dict) -> str:
        """Generate new config file with modifications.

        Args:
            changes: Dict like {"max_tokens": 1024, "max_iterations": 25}

        Returns:
            Python code for new config
        """
        # Get current values
        max_tokens = changes.get("max_tokens", CONFIG.models["fast"].max_tokens)
        context_window = changes.get("context_window", CONFIG.models["fast"].context_window)
        max_iterations = changes.get("max_iterations", CONFIG.max_iterations)
        web_search_timeout = changes.get("web_search_timeout", CONFIG.web_search_timeout)
        code_execution_timeout = changes.get("code_execution_timeout", CONFIG.code_execution_timeout)
        max_search_results = changes.get("max_search_results", CONFIG.max_search_results)

        budget = CONFIG.context_budget
        system_prompt = changes.get("system_prompt", budget.system_prompt)
        tool_definitions = changes.get("tool_definitions", budget.tool_definitions)
        conversation_history = changes.get("conversation_history", budget.conversation_history)
        retrieved_memory = changes.get("retrieved_memory", budget.retrieved_memory)
        workspace = changes.get("workspace", budget.workspace)
        response_buffer = changes.get("response_buffer", budget.response_buffer)

        # Generate Python code
        code = (
            '"""Auto-generated config - evolved version."""\n\n'
            'from pathlib import Path\n'
            'from config import AgentConfig, MLXModelConfig, ContextBudgetConfig\n\n'
            'CONFIG = AgentConfig(\n'
            '    models={\n'
            '        "fast": MLXModelConfig(\n'
            f'            name="mlx-community/Qwen3-8B-4bit",\n'
            f'            max_tokens={max_tokens},\n'
            f'            context_window={context_window},\n'
            '        ),\n'
            '    },\n'
            '    default_model="fast",\n'
            '    context_budget=ContextBudgetConfig(\n'
            f'        system_prompt={system_prompt},\n'
            f'        tool_definitions={tool_definitions},\n'
            f'        conversation_history={conversation_history},\n'
            f'        retrieved_memory={retrieved_memory},\n'
            f'        workspace={workspace},\n'
            f'        response_buffer={response_buffer},\n'
            '    ),\n'
            f'    max_iterations={max_iterations},\n'
            f'    output_dir=Path("./agent_outputs"),\n'
            f'    web_search_timeout={web_search_timeout},\n'
            f'    code_execution_timeout={code_execution_timeout},\n'
            f'    max_search_results={max_search_results},\n'
            ')\n'
        )

        return code

    def evaluate_version(
        self, version_id: str, score: float, results: dict
    ) -> None:
        """Record performance metrics for a version.

        Args:
            version_id: Version to evaluate (e.g., "v1")
            score: Performance score (0-1, higher is better)
            results: Detailed test results dict
        """
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")

        version = self.versions[version_id]
        version.performance_score = score
        version.test_results = results

        self._save_history()

    def get_best_version(self) -> Optional[str]:
        """Get the best performing version ID."""
        if not self.versions:
            return None

        return max(self.versions.keys(), key=lambda v: self.versions[v].performance_score)

    def compare_versions(self, v1: str, v2: str) -> dict:
        """Compare two versions.

        Args:
            v1, v2: Version IDs

        Returns:
            Comparison dict with winner, scores, etc.
        """
        if v1 not in self.versions or v2 not in self.versions:
            raise ValueError("One or both versions not found")

        ver1 = self.versions[v1]
        ver2 = self.versions[v2]

        winner = v1 if ver1.performance_score > ver2.performance_score else v2
        improvement = abs(ver1.performance_score - ver2.performance_score)

        return {
            "winner": winner,
            "improvement_delta": improvement,
            "v1_score": ver1.performance_score,
            "v2_score": ver2.performance_score,
            "v1_description": ver1.description,
            "v2_description": ver2.description,
        }

    def promote_version(self, version_id: str) -> None:
        """Promote a version to become the default (overwrites config.py).

        Args:
            version_id: Version to promote
        """
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")

        version = self.versions[version_id]

        # Backup current config
        backup_file = self.versions_dir / f"config_backup_{datetime.now().isoformat()[:10]}.py"
        import shutil
        shutil.copy("config.py", backup_file)

        # Copy promoted version to config.py
        shutil.copy(version.config_file, "config.py")

        print(f"✅ Promoted {version_id} to default config")
        print(f"   Backup saved: {backup_file}")
