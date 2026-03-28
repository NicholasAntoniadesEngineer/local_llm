import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.context_manager import BudgetResult
from src.logger import AgentLogger
from src.runtime.controller import AgentController
from src.runtime.state_store import PersistentStateStore
from src.runtime.task_state import TASK_PHASE_PLAN, TaskState
from src.runtime.tools import ToolExecutionResult
from src.runtime.verifier import RuntimeVerifier, VerificationResult


class FakeLogger:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    def step_start(self, *args, **kwargs):
        return None

    def tool_call(self, *args, **kwargs):
        return None

    def tool_result(self, *args, **kwargs):
        return None

    def validation(self, *args, **kwargs):
        return None

    def loop_detected(self, *args, **kwargs):
        return None

    def phase_change(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


class FakeContextGuard:
    def __init__(self) -> None:
        self.formatter_calls = 0

    def enforce_budget(self, messages, formatter):
        self.formatter_calls += 1
        formatter(messages)
        return BudgetResult(messages=messages, prompt_tokens=7, action="none")


class FakePolicyEngine:
    class Config:
        no_tool_retry_limit = 1
        idle_budget_s = 0.0
        stuck_abort_limit = 2

    def __init__(self) -> None:
        self.config = self.Config()

    def build_step_policy(self, **kwargs):
        return type(
            "FakeStepPolicy",
            (),
            {
                "guidance_messages": [],
                "suggested_tool": "",
                "confidence": 1.0,
                "action": "inspect",
                "active_skill_name": "",
            },
        )()

    def select_tool_batch(self, tool_calls):
        return tool_calls

    def fallback_tool_call(self, phase, goal, suggested_tool=""):
        return {"name": "list_dir", "arguments": {"path": "."}}

    def reward_from_outcome(self, verification, loop_detected=False):
        return 0.0


class FakeVerifier:
    def evaluate_tool_result(self, tool_name, result_text, written_path=None):
        return VerificationResult(
            status="observation",
            accepted=False,
            should_stop=False,
            summary=result_text[:80],
            target_path="",
        )

    def evaluate_completion_signal(self, response_text, last_verification):
        return VerificationResult(
            status="accepted_completion",
            accepted=True,
            should_stop=True,
            summary="accepted",
            target_path="artifact.py",
        )


class VerifierThatRaises(FakeVerifier):
    def evaluate_tool_result(self, tool_name, result_text, written_path=None):
        raise RuntimeError("simulated verifier fault")


class FakeStateStoreMinimal:
    run_id = "contract-test"
    goal = "g"

    def record_tool_attempt(self, **kwargs):
        return None

    def record_validation(self, **kwargs):
        return None


class FakeToolExecutorListDir:
    def execute(self, tool_name, tool_args):
        return ToolExecutionResult(
            tool_name=tool_name,
            success=True,
            output="listing ok",
            summary="listing ok",
            result_kind="ok",
            details={},
            written_path=None,
        )


class FakeIdleSchedulerMinimal:
    def enqueue(self, *args, **kwargs):
        return None

    def run_pending(self, *args, **kwargs):
        return None

    def get_result(self, _key):
        return None


class FakeSkillTree:
    def peek_next_skill(self):
        return None

    def update_impact_from_result(self, *args, **kwargs):
        return None


class ControllerContractTests(unittest.TestCase):
    def test_controller_uses_injected_format_prompt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            state_file = Path(temp_dir) / "controller_state.json"
            with patch("src.runtime.state_store.STATE_FILE", state_file):
                state_store = PersistentStateStore("run-id", "goal", run_dir)
                context_guard = FakeContextGuard()
                controller = AgentController(
                    goal="goal",
                    config_model=type("ConfigModel", (), {"name": "model", "max_tokens": 10, "context_window": 100})(),
                    logger=FakeLogger(run_dir),
                    memory_manager=None,
                    state_store=state_store,
                    policy_engine=FakePolicyEngine(),
                    verifier=FakeVerifier(),
                    tool_executor=type("ToolExecutor", (), {"files_written": 0})(),
                    skill_tree=FakeSkillTree(),
                    idle_scheduler=type("IdleScheduler", (), {})(),
                    context_guard=context_guard,
                    compress_context=lambda messages: messages,
                    format_prompt=lambda messages: "formatted-prompt",
                    generate_response=lambda messages: "DONE",
                    extract_tool_calls=lambda response: [],
                    build_memory_context=lambda: "",
                    load_skill_instance=lambda skill_name, class_name: None,
                    pre_validate=lambda path: "OK",
                    evaluate_written_file=lambda path: {},
                    perf={"tool_success": {"total": 0, "success": 0}},
                    max_iterations=1,
                    resource_sampler=lambda run_dir_value, stop_event: None,
                )

                controller.run(resume=False)

                self.assertEqual(context_guard.formatter_calls, 1)

    def test_tool_pipeline_survives_verifier_exception(self):
        controller = AgentController(
            goal="goal",
            config_model=type("ConfigModel", (), {"name": "model", "max_tokens": 10, "context_window": 100})(),
            logger=FakeLogger(Path(tempfile.mkdtemp()) / "r"),
            memory_manager=None,
            state_store=FakeStateStoreMinimal(),
            policy_engine=FakePolicyEngine(),
            verifier=VerifierThatRaises(),
            tool_executor=FakeToolExecutorListDir(),
            skill_tree=FakeSkillTree(),
            idle_scheduler=FakeIdleSchedulerMinimal(),
            context_guard=FakeContextGuard(),
            compress_context=lambda messages: messages,
            format_prompt=lambda messages: "p",
            generate_response=lambda messages: "",
            extract_tool_calls=lambda response: [],
            build_memory_context=lambda: "",
            load_skill_instance=lambda skill_name, class_name: None,
            pre_validate=lambda path: "OK",
            evaluate_written_file=lambda path: {},
            perf={"tool_success": {"total": 0, "success": 0}},
            max_iterations=1,
            resource_sampler=lambda run_dir_value, stop_event: None,
        )
        task_state = TaskState(task_id="t", goal_text="goal")
        task_state.phase = TASK_PHASE_PLAN
        abort, verification = controller._run_single_tool_iteration(
            step=1,
            task_state=task_state,
            tool_call={"name": "list_dir", "arguments": {"path": "."}},
            loop_detector=None,
        )
        self.assertFalse(abort)
        self.assertIsNone(verification)
        self.assertEqual(task_state.phase, TASK_PHASE_PLAN)
        self.assertTrue(any("Tool pipeline error" in r for r in task_state.failure_reasons))


class ValidateModuleCrashTests(unittest.TestCase):
    def test_evaluate_tool_result_survives_validate_generated_module_crash(self):
        skill_file = Path(tempfile.mkdtemp()) / "some_skill.py"
        skill_file.write_text("def run():\n    return 1\n", encoding="utf-8")
        verifier = RuntimeVerifier(skill_tree=None)
        with patch(
            "src.runtime.verifier.validate_generated_module",
            side_effect=RuntimeError("validator blew up"),
        ):
            result = verifier.evaluate_tool_result("write_file", "wrote", written_path=skill_file)
        self.assertFalse(result.accepted)
        self.assertIn("crashed", result.summary)


class LoggerRunIdTests(unittest.TestCase):
    def test_logger_writes_run_id_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_run_dir = Path(temp_dir) / "logger-run"
            fake_run_dir.mkdir(parents=True, exist_ok=True)
            with patch("src.logger.get_run_dir", return_value=fake_run_dir):
                logger = AgentLogger("test-run-id")
            self.assertEqual((logger.run_dir / "run_id.txt").read_text(), "test-run-id")


if __name__ == "__main__":
    unittest.main()
