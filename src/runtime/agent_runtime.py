"""Bundled runtime dependencies for the verifier-driven controller (self-improve / skill loops)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from src.config import CONFIG
from src.runtime.controller import AgentController
from src.runtime.runtime_support import resource_sampler

if TYPE_CHECKING:
    from src.agent import MLXAgent


@dataclass(frozen=True)
class AgentRuntimeKernel:
    """Single object carrying everything `AgentController` needs from `MLXAgent`.

    Keeps `run_loop` to one line and makes it obvious what the controller depends on.
    """

    goal: str
    config_model: Any
    logger: Any
    memory_manager: Any
    state_store: Any
    policy_engine: Any
    verifier: Any
    tool_executor: Any
    skill_tree: Any
    idle_scheduler: Any
    context_guard: Any
    compress_context: Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
    format_prompt: Callable[..., str]
    generate_response: Callable[[list[dict[str, Any]]], str]
    extract_tool_calls: Callable[[str], list[dict[str, Any]]]
    build_memory_context: Callable[[], str]
    load_skill_instance: Callable[[str, str], Any]
    pre_validate: Callable[[str], str]
    evaluate_written_file: Callable[..., dict[str, Any]]
    perf: dict[str, Any]
    max_iterations: int
    resource_sampler: Callable[[Any, threading.Event], None]

    def build_controller(self) -> AgentController:
        return AgentController(
            goal=self.goal,
            config_model=self.config_model,
            logger=self.logger,
            memory_manager=self.memory_manager,
            state_store=self.state_store,
            policy_engine=self.policy_engine,
            verifier=self.verifier,
            tool_executor=self.tool_executor,
            skill_tree=self.skill_tree,
            idle_scheduler=self.idle_scheduler,
            context_guard=self.context_guard,
            compress_context=self.compress_context,
            format_prompt=self.format_prompt,
            generate_response=self.generate_response,
            extract_tool_calls=self.extract_tool_calls,
            build_memory_context=self.build_memory_context,
            load_skill_instance=self.load_skill_instance,
            pre_validate=self.pre_validate,
            evaluate_written_file=self.evaluate_written_file,
            perf=self.perf,
            max_iterations=self.max_iterations,
            resource_sampler=self.resource_sampler,
        )

    @staticmethod
    def from_mlx_agent(agent: MLXAgent, goal: str) -> AgentRuntimeKernel:
        return AgentRuntimeKernel(
            goal=goal,
            config_model=agent.config_model,
            logger=agent.logger,
            memory_manager=agent.memory_manager,
            state_store=agent.state_store,
            policy_engine=agent.policy_engine,
            verifier=agent.verifier,
            tool_executor=agent.tool_executor,
            skill_tree=agent.skill_tree,
            idle_scheduler=agent._idle_scheduler,
            context_guard=agent._context_guard,
            compress_context=agent._compress_context,
            format_prompt=agent._format_prompt,
            generate_response=agent._generate_response,
            extract_tool_calls=agent._extract_tool_calls,
            build_memory_context=agent._build_memory_context,
            load_skill_instance=agent._load_skill_instance,
            pre_validate=agent._pre_validate,
            evaluate_written_file=agent._evaluate_written_file,
            perf=agent._perf,
            max_iterations=CONFIG.max_iterations,
            resource_sampler=resource_sampler,
        )
