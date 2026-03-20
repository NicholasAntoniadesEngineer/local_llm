#!/usr/bin/env python3
"""
Autonomous Trading Agent Launcher

Usage:
    python -m scripts.run_trader --capital 50 --hours 24
    python -m scripts.run_trader --capital 100 --strategy aggressive
    python -m scripts.run_trader stats
    python -m scripts.run_trader portfolio
"""

import asyncio
import argparse
import json
from pathlib import Path
import structlog

from src.agent.traders import AutonomousTrader
from src.llm.router import ModelRouter
from src.memory.manager import MemoryManager

logger = structlog.get_logger(__name__)

# Results directory
RESULTS_DIR = Path(__file__).parent.parent / "trading_results"
RESULTS_DIR.mkdir(exist_ok=True)


async def run_trader(capital: float = 50.0, hours: int = 24, strategy: str = "balanced"):
    """
    Run autonomous trader.

    Args:
        capital: Starting capital in EUR
        hours: How long to run (24 = 1 day, 168 = 1 week)
        strategy: Trading strategy (conservative, balanced, aggressive)
    """
    logger.info(
        "starting_trader",
        capital=capital,
        hours=hours,
        strategy=strategy,
    )

    # Initialize components
    router = ModelRouter("config/model_config.yaml")
    memory = MemoryManager("data")

    # Configure strategy
    risk_per_trade = {"conservative": 0.02, "balanced": 0.05, "aggressive": 0.10}.get(
        strategy, 0.05
    )

    # Create trader
    trader = AutonomousTrader(
        router=router,
        memory=memory,
        initial_capital=capital,
        risk_per_trade=risk_per_trade,
    )

    print(f"""
╔════════════════════════════════════════════════════╗
║         AUTONOMOUS TRADING AGENT STARTING            ║
╠════════════════════════════════════════════════════╣
║  Capital:        €{capital:.2f}                         ║
║  Duration:       {hours} hours                             ║
║  Strategy:       {strategy.upper()}                          ║
║  Risk per Trade: {risk_per_trade:.1%}                           ║
║                                                    ║
║  Agent will:                                       ║
║  1. Discover available platforms & markets         ║
║  2. Research trading opportunities                 ║
║  3. Execute trades (with safeguards)               ║
║  4. Monitor positions                              ║
║  5. Learn from results                             ║
║                                                    ║
║  🔴 DO NOT CLOSE THIS WINDOW - Agent is working    ║
╚════════════════════════════════════════════════════╝
    """)

    try:
        # Run autonomous trading cycle
        stats = await trader.run_autonomous_cycle(hours=hours)

        # Get final portfolio
        portfolio = trader.get_portfolio_summary()

        # Print results
        print(f"""
╔════════════════════════════════════════════════════╗
║              TRADING SESSION COMPLETE              ║
╠════════════════════════════════════════════════════╣
║  Trades Executed:    {stats.get('trades_executed', 0):<33}║
║  Platforms Found:    {stats.get('platforms_discovered', 0):<33}║
║  Opportunities:      {stats.get('opportunities_found', 0):<33}║
║  Learning Updates:   {stats.get('learning_updates', 0):<33}║
║                                                    ║
║  Final Capital:      €{portfolio['capital']:<31.2f}║
║  Total P&L:          €{portfolio['total_pnl']:<31.2f}║
║  ROI:                {portfolio['roi']:<34}║
║  Win Rate:           {portfolio['win_rate']:<34}║
╚════════════════════════════════════════════════════╝
        """)

        # Save results
        result_file = RESULTS_DIR / f"session_{int(asyncio.get_event_loop().time())}.json"
        with open(result_file, "w") as f:
            json.dump(
                {
                    "session_stats": stats,
                    "portfolio": portfolio,
                    "timestamp": json.dumps(stats, indent=2, default=str),
                },
                f,
                indent=2,
                default=str,
            )

        logger.info(
            "session_complete",
            capital=portfolio["capital"],
            pnl=portfolio["total_pnl"],
            roi=portfolio["roi"],
        )

        return portfolio

    except Exception as e:
        logger.error("trader_failed", error=str(e))
        print(f"\n❌ ERROR: {str(e)}")
        raise


def show_portfolio_stats():
    """Show cumulative portfolio stats."""
    print("""
╔════════════════════════════════════════════════════╗
║          PORTFOLIO STATISTICS (ALL SESSIONS)       ║
╠════════════════════════════════════════════════════╣

Your autonomous agent has been learning and improving.

View detailed results in:
  ./trading_results/session_*.json

Each session contains:
  - Trades executed & P&L
  - Markets discovered
  - Rules applied & violated
  - Learning updates
  - Portfolio evolution

To view latest session:
  cat trading_results/session_*.json | tail -1
╚════════════════════════════════════════════════════╝
    """)


async def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Autonomous Trading Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start trading with €50 for 24 hours
  python -m scripts.run_trader --capital 50 --hours 24

  # Run aggressive strategy for 1 week
  python -m scripts.run_trader --capital 100 --hours 168 --strategy aggressive

  # View portfolio stats
  python -m scripts.run_trader stats

  # Run continuously (loops every 24 hours)
  python -m scripts.run_trader --capital 100 --loop
        """,
    )

    parser.add_argument("--capital", type=float, default=50.0, help="Starting capital (EUR)")
    parser.add_argument("--hours", type=int, default=24, help="Run duration (hours)")
    parser.add_argument(
        "--strategy",
        choices=["conservative", "balanced", "aggressive"],
        default="balanced",
        help="Trading strategy",
    )
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("stats", nargs="?", help="Show portfolio stats")

    args = parser.parse_args()

    if args.stats == "stats":
        show_portfolio_stats()
        return

    # Run trader
    try:
        if args.loop:
            # Run continuously
            iteration = 1
            while True:
                print(f"\n📊 Starting iteration {iteration}...")
                await run_trader(
                    capital=args.capital,
                    hours=args.hours,
                    strategy=args.strategy,
                )
                iteration += 1
                print(f"\n⏰ Sleeping 1 hour before next iteration...")
                await asyncio.sleep(3600)
        else:
            # Run once
            await run_trader(
                capital=args.capital,
                hours=args.hours,
                strategy=args.strategy,
            )
    except KeyboardInterrupt:
        print("\n\n⏹️  Trader stopped by user")
        logger.info("trader_stopped_by_user")
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {str(e)}")
        logger.error("fatal_error", error=str(e))
        raise


if __name__ == "__main__":
    asyncio.run(main())
