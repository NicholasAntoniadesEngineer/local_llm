"""
Example: Running autonomous research with the agent.

This example demonstrates how to:
1. Initialize the research agent
2. Run research on an objective
3. Resume a paused session
4. Query research memory

Requirements:
- Ollama running locally with models loaded
- TAVILY_API_KEY environment variable set
- .env file with configuration
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.router import ModelRouter
from src.memory import MemoryManager
from src.agent import ResearchAgent


async def main():
    """Run example research."""

    # Initialize components
    print("Initializing components...")
    model_router = ModelRouter(config_path="config/model_config.yaml")
    memory_manager = MemoryManager()
    agent = ResearchAgent(memory_manager, model_router)

    # Run research
    print("\n" + "=" * 80)
    print("RUNNING AUTONOMOUS RESEARCH")
    print("=" * 80)

    objective = "What are the latest breakthroughs in quantum computing and quantum error correction?"

    print(f"\nObjective: {objective}")
    print("\nStarting research (max 15 steps)...")

    result = await agent.research(
        objective=objective,
        max_steps=15,
        metadata={
            "source": "example",
            "tags": ["quantum", "computing"],
        },
    )

    # Display results
    print("\n" + "=" * 80)
    print("RESEARCH COMPLETE")
    print("=" * 80)

    print(f"\nSession ID: {result['session_id']}")
    print(f"Status: {result['execution_status']}")
    print(f"Findings collected: {len(result['findings'])}")
    print(f"Tokens used: {result['context_tokens']}")

    if result["final_response"]:
        print("\n" + "-" * 80)
        print("SYNTHESIZED RESPONSE")
        print("-" * 80)
        print(result["final_response"])

    if result["error_message"]:
        print("\n" + "-" * 80)
        print("ERRORS ENCOUNTERED")
        print("-" * 80)
        print(result["error_message"])

    # Cleanup
    await agent.cleanup()
    print("\n" + "=" * 80)
    print("Research complete!")


async def example_resume():
    """Example: Resume a paused session."""

    model_router = ModelRouter(config_path="config/model_config.yaml")
    memory_manager = MemoryManager()
    agent = ResearchAgent(memory_manager, model_router)

    # This would be a previously saved session ID
    session_id = "abc123def456"

    print(f"Resuming session {session_id}...")

    try:
        result = await agent.resume_session(session_id)
        print(f"Session status: {result['execution_status']}")
    except ValueError as e:
        print(f"Session not found: {e}")

    await agent.cleanup()


async def example_query_memory():
    """Example: Query research memory."""

    model_router = ModelRouter(config_path="config/model_config.yaml")
    memory_manager = MemoryManager()
    agent = ResearchAgent(memory_manager, model_router)

    query = "quantum error correction techniques"

    print(f"Querying memory for: {query}")

    findings = await agent.query_memory(query, top_k=5)

    print(f"\nFound {len(findings)} relevant findings:")
    for i, finding in enumerate(findings, 1):
        print(f"  {i}. {finding.get('title', 'Finding')}")

    await agent.cleanup()


if __name__ == "__main__":
    # Run main example
    asyncio.run(main())

    # Uncomment to run other examples:
    # asyncio.run(example_resume())
    # asyncio.run(example_query_memory())
