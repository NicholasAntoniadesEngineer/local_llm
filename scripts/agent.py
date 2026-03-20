"""CLI entry point for autonomous research agent."""

import asyncio
import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

import structlog
from dotenv import load_dotenv

from src.llm.router import ModelRouter
from src.memory import MemoryManager
from src.agent import ResearchAgent

# Setup structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


async def run_research(args: argparse.Namespace) -> int:
    """
    Run autonomous research for given objective.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    try:
        # Initialize components
        model_router = ModelRouter(config_path="config/model_config.yaml")
        memory_manager = MemoryManager()  # Would initialize with proper config

        agent = ResearchAgent(memory_manager, model_router)

        logger.info(
            "research_start",
            objective=args.objective,
            max_steps=args.max_steps,
        )

        # Run research
        result = await agent.research(
            objective=args.objective,
            max_steps=args.max_steps,
            metadata={
                "source": "cli",
                "tags": args.tags.split(",") if args.tags else [],
            },
        )

        # Print results
        print("\n" + "=" * 80)
        print("RESEARCH COMPLETE")
        print("=" * 80)
        print(f"\nSession ID: {result['session_id']}")
        print(f"Status: {result['execution_status']}")
        print(f"Steps: {len(result['messages']) // 2}")
        print(f"Findings: {len(result['findings'])}")
        print(f"Tokens used: {result['context_tokens']}")

        print("\n" + "-" * 80)
        print("FINAL RESPONSE")
        print("-" * 80)
        print(result["final_response"])

        if result["error_message"]:
            print("\n" + "-" * 80)
            print("ERRORS")
            print("-" * 80)
            print(result["error_message"])

        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            logger.info("results_saved", path=args.output)
            print(f"\nResults saved to: {args.output}")

        await agent.cleanup()

        return 0 if result["execution_status"] == "complete" else 1

    except Exception as e:
        logger.error("research_failed", error=str(e))
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


async def resume_session(args: argparse.Namespace) -> int:
    """
    Resume a paused research session.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code
    """
    try:
        model_router = ModelRouter(config_path="config/model_config.yaml")
        memory_manager = MemoryManager()

        agent = ResearchAgent(memory_manager, model_router)

        logger.info("resuming_session", session_id=args.session_id)

        result = await agent.resume_session(args.session_id)

        print("\n" + "=" * 80)
        print("RESEARCH RESUMED AND COMPLETE")
        print("=" * 80)
        print(f"\nSession ID: {result['session_id']}")
        print(f"Status: {result['execution_status']}")
        print(f"Findings: {len(result['findings'])}")
        print("\n" + "-" * 80)
        print("FINAL RESPONSE")
        print("-" * 80)
        print(result["final_response"])

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\nResults saved to: {args.output}")

        await agent.cleanup()

        return 0 if result["execution_status"] == "complete" else 1

    except Exception as e:
        logger.error("session_resume_failed", error=str(e))
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


async def query_memory(args: argparse.Namespace) -> int:
    """
    Query research memory.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code
    """
    try:
        memory_manager = MemoryManager()
        model_router = ModelRouter(config_path="config/model_config.yaml")

        agent = ResearchAgent(memory_manager, model_router)

        logger.info("memory_query", query=args.query)

        results = await agent.query_memory(
            query=args.query,
            session_id=args.session_id,
            top_k=args.top_k,
        )

        print("\n" + "=" * 80)
        print("MEMORY QUERY RESULTS")
        print("=" * 80)

        if not results:
            print("\nNo matching findings.")
        else:
            for i, result in enumerate(results, 1):
                print(f"\n[{i}] {result.get('title', 'Finding')}")
                print(f"    Score: {result.get('confidence', 0):.2f}")
                print(f"    Content: {result.get('content', '')[:200]}...")

        await agent.cleanup()
        return 0

    except Exception as e:
        logger.error("query_failed", error=str(e))
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


def main() -> int:
    """
    Main CLI entry point.

    Returns:
        Exit code
    """
    # Load environment
    load_dotenv()

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Autonomous research agent for local LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.agent run \\
    --objective "Find latest quantum computing advances" \\
    --max-steps 10 \\
    --output results.json

  python -m scripts.agent resume \\
    --session-id abc123def456 \\
    --output results.json

  python -m scripts.agent query \\
    --query "quantum error correction" \\
    --top-k 5
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run new research")
    run_parser.add_argument(
        "--objective",
        required=True,
        help="Research objective/query",
    )
    run_parser.add_argument(
        "--max-steps",
        type=int,
        default=15,
        help="Maximum research steps (default: 15)",
    )
    run_parser.add_argument(
        "--tags",
        help="Comma-separated tags for metadata",
    )
    run_parser.add_argument(
        "--output",
        help="Output file for results (JSON)",
    )
    run_parser.set_defaults(func=run_research)

    # Resume command
    resume_parser = subparsers.add_parser("resume", help="Resume paused session")
    resume_parser.add_argument(
        "--session-id",
        required=True,
        help="Session ID to resume",
    )
    resume_parser.add_argument(
        "--output",
        help="Output file for results (JSON)",
    )
    resume_parser.set_defaults(func=resume_session)

    # Query command
    query_parser = subparsers.add_parser("query", help="Query research memory")
    query_parser.add_argument(
        "--query",
        required=True,
        help="Natural language query",
    )
    query_parser.add_argument(
        "--session-id",
        help="Limit to specific session",
    )
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results (default: 5)",
    )
    query_parser.set_defaults(func=query_memory)

    # Parse and execute
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Run async function
    return asyncio.run(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
