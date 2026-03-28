import unittest

from src.runtime.llm_text import extract_python_code_block, strip_thinking_tags
from src.runtime.prompt_builder import (
    build_plan_items_from_policy,
    build_prompt_messages,
    build_task_hypothesis,
)
from src.runtime.task_state import TaskState


class FakePolicy:
    def __init__(self) -> None:
        self.guidance_messages = [
            "Inspect the controller contract first.",
            "Compare the current phase against recent verifier failures.",
            "Prefer observation before mutation when uncertain.",
            "Keep the next write tightly scoped.",
            "This extra message should be truncated.",
        ]
        self.suggested_tool = "read_file"
        self.confidence = 0.625
        self.action = "inspect"
        self.active_skill_name = "Prompt Builder"


class PromptBuilderTests(unittest.TestCase):
    def test_build_plan_items_from_policy_limits_output(self) -> None:
        policy = FakePolicy()

        plan_items = build_plan_items_from_policy(policy)

        self.assertEqual(len(plan_items), 6)
        self.assertEqual(plan_items[0], "Advance active skill target: Prompt Builder")
        self.assertEqual(plan_items[1], "Preferred next tool: read_file")
        self.assertEqual(plan_items[2], "Primary controller action: inspect")
        self.assertNotIn("This extra message should be truncated.", plan_items)

    def test_build_task_hypothesis_uses_phase_confidence_and_next_action(self) -> None:
        task_state = TaskState(task_id="run-1", goal_text="Refactor controller prompt assembly")
        task_state.transition_phase("plan", "Need a concrete extraction plan.")
        policy = FakePolicy()

        hypothesis = build_task_hypothesis(task_state, policy)

        self.assertIn("Phase=plan", hypothesis)
        self.assertIn("confidence=0.62", hypothesis)
        self.assertIn("next=read_file", hypothesis)
        self.assertIn("Inspect the controller contract first.", hypothesis)

    def test_build_prompt_messages_preserves_protected_contract_and_context_sections(self) -> None:
        task_state = TaskState(task_id="run-2", goal_text="Extract prompt builder")
        task_state.set_plan_items([f"Plan item {index}" for index in range(1, 4)])
        task_state.current_hypothesis = "Phase=plan; confidence=0.62; next=read_file; inspect the controller"
        for index in range(8):
            task_state.add_target_file(f"src/file_{index}.py")
        for index in range(7):
            task_state.add_failure_reason(f"failure {index}")
        task_state.update_verification("rejected_write", False, "Tests failed in prompt builder", "test")
        for index in range(6):
            task_state.mark_step(index + 1)
            task_state.add_action_record(
                tool_name="read_file",
                success=index % 2 == 0,
                args_preview=f"args-{index}",
                result_preview=f"result-{index}",
            )
        policy = FakePolicy()

        prompt_messages = build_prompt_messages(
            task_state,
            policy,
            memory_context="Memory summary block",
            retrieval_context=["Validation: old failure", "Memory: prior attempt"],
            past_failures=["failure A", "failure B", "failure C", "failure D", "failure E"],
        )

        self.assertEqual(len(prompt_messages), 2)
        self.assertEqual(prompt_messages[0]["role"], "system")
        self.assertEqual(prompt_messages[1]["role"], "user")
        self.assertTrue(all(message["protected"] for message in prompt_messages))

        system_prompt = prompt_messages[0]["content"]
        user_prompt = prompt_messages[1]["content"]
        self.assertIn("/nothink", system_prompt)
        self.assertIn("Goal: Extract prompt builder", system_prompt)
        self.assertIn("Current hypothesis:\n- Phase=plan; confidence=0.62; next=read_file; inspect the controller", system_prompt)
        self.assertIn("Latest verifier result:\n- status=rejected_write", system_prompt)
        self.assertIn("Relevant past records:\n- Validation: old failure\n- Memory: prior attempt", system_prompt)
        self.assertIn("Historical verifier failures:\n- failure A\n- failure B\n- failure C\n- failure D", system_prompt)
        self.assertNotIn("failure E", system_prompt)
        self.assertIn("Memory summary block", system_prompt)
        self.assertIn("src/file_2.py", system_prompt)
        self.assertNotIn("src/file_1.py", system_prompt)
        self.assertIn("failure 3", system_prompt)
        self.assertNotIn("failure 2", system_prompt)
        self.assertIn("Tests failed in prompt builder", system_prompt)
        self.assertIn("result=result-5", system_prompt)
        self.assertNotIn("result=result-0", system_prompt)
        self.assertIn("Controller guidance:\n- Inspect the controller contract first.", user_prompt)
        self.assertNotIn("This extra message should be truncated.", user_prompt)


class LLMTextTests(unittest.TestCase):
    def test_strip_thinking_tags_removes_hidden_reasoning(self) -> None:
        self.assertEqual(strip_thinking_tags("<think>hidden</think>\n\nVisible answer"), "Visible answer")

    def test_extract_python_code_block_returns_first_fenced_block(self) -> None:
        text = "before```python\nprint('hello')\n```after"
        self.assertEqual(extract_python_code_block(text), "print('hello')")
        self.assertIsNone(extract_python_code_block("no code fence here"))


if __name__ == "__main__":
    unittest.main()
