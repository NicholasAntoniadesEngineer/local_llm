"""LangGraph-based orchestrator for autonomous research agent."""

import asyncio
from datetime import datetime
from typing import Any
import uuid

import structlog
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from src.llm.router import ModelRouter
from src.memory import MemoryManager, SessionInfo
from .state import AgentState
from .nodes.plan import plan_node
from .nodes.think import think_node
from .nodes.act import act_node
from .nodes.observe import observe_node
from .nodes.reflect import reflect_node
from .nodes.synthesize import synthesize_node
from .nodes.enforce_rules import enforce_rules_node

logger = structlog.get_logger(__name__)


def create_agent_graph(
    memory_manager: MemoryManager,
    model_router: ModelRouter,
    checkpoint_dir: str = "data/checkpoints",
) -> tuple[Any, SqliteSaver]:
    """
    Create LangGraph state machine for research agent.

    Graph flow:
        START → plan → think → act → observe → reflect → {continue | synthesize} → enforce_rules → END

    The reflect node decides whether to:
    - Continue research loop (plan next research goal)
    - Move to synthesis (found enough information)
    - Force synthesis (max steps reached)

    Args:
        memory_manager: Initialized MemoryManager for persistence
        model_router: ModelRouter for LLM selection
        checkpoint_dir: Directory for SqliteSaver checkpoints

    Returns:
        Tuple of (compiled graph, checkpoint saver) for execution and resumability

    Raises:
        ValueError: If graph construction fails
    """

    # Create graph
    graph = StateGraph(AgentState)

    # Bind memory_manager and model_router to node functions
    # (Nodes use closure to access these)
    def _plan(state: AgentState) -> AgentState:
        return plan_node(state, memory_manager, model_router)

    def _think(state: AgentState) -> AgentState:
        return think_node(state, memory_manager, model_router)

    def _act(state: AgentState) -> AgentState:
        return asyncio.run(act_node(state, memory_manager, model_router))

    def _observe(state: AgentState) -> AgentState:
        return asyncio.run(observe_node(state, memory_manager, model_router))

    def _reflect(state: AgentState) -> AgentState:
        return reflect_node(state, memory_manager, model_router)

    def _synthesize(state: AgentState) -> AgentState:
        return synthesize_node(state, memory_manager, model_router)

    def _enforce_rules(state: AgentState) -> AgentState:
        return asyncio.run(enforce_rules_node(state, memory_manager, model_router))

    # Add nodes
    graph.add_node("plan", _plan)
    graph.add_node("think", _think)
    graph.add_node("act", _act)
    graph.add_node("observe", _observe)
    graph.add_node("reflect", _reflect)
    graph.add_node("synthesize", _synthesize)
    graph.add_node("enforce_rules", _enforce_rules)

    # Add edges
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "think")
    graph.add_edge("think", "act")
    graph.add_edge("act", "observe")
    graph.add_edge("observe", "reflect")

    # Conditional edge from reflect: continue_research vs synthesize
    def should_continue_research(state: AgentState) -> str:
        """Decide whether to continue research or synthesize."""
        if state.get("should_continue"):
            return "plan"
        else:
            return "synthesize"

    graph.add_conditional_edges("reflect", should_continue_research)

    graph.add_edge("synthesize", "enforce_rules")
    graph.add_edge("enforce_rules", END)

    # Create checkpoint saver
    saver = SqliteSaver(f"{checkpoint_dir}/agent_state.db")

    # Compile with checkpointing
    compiled_graph = graph.compile(checkpointer=saver)

    logger.info("agent_graph_created", nodes=7, checkpointer="sqlite")

    return compiled_graph, saver


class ResearchAgent:
    """
    High-level interface for autonomous research execution.

    Manages session lifecycle, checkpointing, memory, and orchestrates
    graph execution with proper error handling and resumability.

    Example:
        agent = ResearchAgent(memory_manager, model_router)
        result = await agent.research(
            objective="What are latest quantum computing breakthroughs?",
            max_steps=15
        )
        print(result.final_response)
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        model_router: ModelRouter,
        checkpoint_dir: str = "data/checkpoints",
    ):
        """
        Initialize research agent.

        Args:
            memory_manager: Initialized MemoryManager
            model_router: Initialized ModelRouter
            checkpoint_dir: Directory for checkpoints
        """
        self.memory_manager = memory_manager
        self.model_router = model_router
        self.checkpoint_dir = checkpoint_dir
        self.graph, self.saver = create_agent_graph(
            memory_manager, model_router, checkpoint_dir
        )

    async def research(
        self,
        objective: str,
        max_steps: int = 15,
        session_id: str | None = None,
        parent_message_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute autonomous research for given objective.

        Initializes session, creates initial state, runs graph with checkpointing,
        and returns final findings with synthesis.

        Args:
            objective: Research objective/query
            max_steps: Maximum research steps (hard limit)
            session_id: Resume from existing session (optional)
            parent_message_id: For multi-turn conversation threading
            metadata: Custom metadata (tags, source, etc.)

        Returns:
            Result dict with:
                - session_id: Session identifier
                - final_response: Synthesized response
                - findings: List of research findings
                - messages: Conversation history
                - context_tokens: Final token count
                - execution_status: "complete" or "failed"
                - error_message: Error details if failed

        Raises:
            ValueError: If objective is empty
            Exception: Execution errors (caught and returned in result)
        """
        if not objective or not objective.strip():
            raise ValueError("Objective cannot be empty")

        try:
            # Create or resume session
            if session_id is None:
                session_id = str(uuid.uuid4())
                session = SessionInfo(
                    id=session_id,
                    objective=objective,
                    max_steps=max_steps,
                    status="active",
                    metadata=metadata or {},
                )
                await self.memory_manager.save_session(session)
                logger.info("session_created", session_id=session_id, objective=objective)
            else:
                # Resume from checkpoint
                session = await self.memory_manager.load_session(session_id)
                logger.info("session_resumed", session_id=session_id)

            # Create initial state
            initial_state: AgentState = {
                "objective": objective,
                "session_id": session_id,
                "sub_goals": [],
                "current_goal_index": 0,
                "step_number": 0,
                "max_steps": max_steps,
                "findings": [],
                "messages": [],
                "context_tokens": 0,
                "max_context_tokens": 16384,
                "rule_violations": [],
                "proposed_rule_changes": [],
                "last_tool_results": {},
                "tool_error": None,
                "checkpoint_data": {},
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "should_continue": True,
                "synthesis_complete": False,
                "final_response": "",
                "error_message": None,
                "execution_status": "planning",
                "parent_message_id": parent_message_id,
                "metadata": metadata or {},
            }

            # Execute graph with checkpointing
            config = {"configurable": {"thread_id": session_id}}

            logger.info("agent_execution_start", session_id=session_id, objective=objective)

            # Run the graph (synchronous wrapper around async execution)
            output = await asyncio.to_thread(
                self.graph.invoke, initial_state, config
            )

            # Update session with final state
            session.status = "completed" if output.get("execution_status") == "complete" else "failed"
            session.current_step = output.get("step_number", 0)
            session.findings_count = len(output.get("findings", []))
            session.messages_count = len(output.get("messages", []))
            session.total_tokens = output.get("context_tokens", 0)
            await self.memory_manager.save_session(session)

            logger.info(
                "agent_execution_complete",
                session_id=session_id,
                status=session.status,
                steps=session.current_step,
                findings=session.findings_count,
            )

            return {
                "session_id": session_id,
                "final_response": output.get("final_response", ""),
                "findings": output.get("findings", []),
                "messages": output.get("messages", []),
                "context_tokens": output.get("context_tokens", 0),
                "execution_status": output.get("execution_status", "failed"),
                "error_message": output.get("error_message"),
            }

        except Exception as e:
            logger.error("agent_execution_failed", session_id=session_id, error=str(e))
            return {
                "session_id": session_id,
                "final_response": "",
                "findings": [],
                "messages": [],
                "context_tokens": 0,
                "execution_status": "failed",
                "error_message": str(e),
            }

    async def resume_session(self, session_id: str) -> dict[str, Any]:
        """
        Resume a paused research session.

        Loads checkpoint and continues execution from last step.

        Args:
            session_id: Session ID to resume

        Returns:
            Same as research() method

        Raises:
            ValueError: If session not found
        """
        session = await self.memory_manager.load_session(session_id)

        if not session:
            raise ValueError(f"Session {session_id} not found")

        if session.status != "paused":
            logger.warning("session_not_paused", session_id=session_id, status=session.status)

        # Resume with existing session_id (will trigger checkpoint load)
        return await self.research(
            objective=session.objective,
            max_steps=session.max_steps,
            session_id=session_id,
            metadata=session.metadata,
        )

    async def query_memory(
        self,
        query: str,
        session_id: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Query research memory across sessions.

        Args:
            query: Natural language query
            session_id: Limit to specific session (optional)
            top_k: Number of top results to return

        Returns:
            List of matching findings with relevance scores
        """
        # This would use memory manager's retrieval interface
        # Placeholder for now
        logger.info("memory_query", query=query, top_k=top_k)
        return []

    async def cleanup(self) -> None:
        """Clean up resources (models, connections, etc.)."""
        await self.model_router.cleanup()
        logger.info("agent_cleanup_complete")
