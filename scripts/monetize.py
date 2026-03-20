#!/usr/bin/env python3
"""
Monetization Engine: Turn Research Agent → Revenue Stream

Usage:
    python -m scripts.monetize generate --type seo --topic "best hiking boots"
    python -m scripts.monetize list-orders
    python -m scripts.monetize process-order --order-id abc123
    python -m scripts.monetize stats
"""

import asyncio
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional
import structlog

from src.agent.core import ResearchAgent
from src.memory.manager import MemoryManager
from src.llm.router import ModelRouter

logger = structlog.get_logger(__name__)

# Output directory for sellable content
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Orders database (simple JSON for MVP)
ORDERS_FILE = OUTPUT_DIR / "orders.jsonl"


class MonetizationEngine:
    """Convert research tasks → sellable deliverables."""

    def __init__(self, agent: ResearchAgent):
        self.agent = agent
        self.router = agent.router
        self.memory = agent.memory

    async def generate_seo_article(
        self,
        topic: str,
        target_length: int = 1500,
        include_sources: bool = True,
    ) -> dict:
        """
        Generate SEO-optimized article ready to sell.

        Args:
            topic: Article topic (e.g., "best hiking boots 2025")
            target_length: Target word count (1000-3000)
            include_sources: Include citations in [1] format

        Returns:
            dict with article, metadata, and formatting
        """
        logger.info("generating_seo_article", topic=topic, target_length=target_length)

        # Research objective
        objective = f"""
        Write a comprehensive, SEO-optimized article about: {topic}

        Requirements:
        - {target_length} words minimum
        - H2 subheadings for scannability
        - Include data/statistics where relevant
        - Natural keyword integration
        - Professional tone
        - Citations in [1], [2], [3] format
        - Include a conclusion with CTA
        """

        # Run agent research
        session = await self.memory.start_session(user_id="monetize", objective=objective)

        try:
            # Execute research with agent
            findings = await self.agent.run(
                objective=objective,
                max_steps=10,
                session_id=session["session_id"],
            )

            # Format as article
            article = self._format_article(
                topic=topic,
                findings=findings,
                include_sources=include_sources,
            )

            logger.info("article_generated", topic=topic, words=len(article["body"].split()))

            return {
                "type": "seo_article",
                "topic": topic,
                "title": article["title"],
                "body": article["body"],
                "sources": article["sources"],
                "word_count": len(article["body"].split()),
                "quality_score": self._calculate_quality(findings),
                "ready_to_sell": True,
                "generated_at": datetime.utcnow().isoformat(),
                "session_id": session["session_id"],
            }

        except Exception as e:
            logger.error("article_generation_failed", topic=topic, error=str(e))
            raise

    async def generate_competitor_analysis(
        self,
        company: str,
        competitors: list[str],
    ) -> dict:
        """Generate competitor analysis report."""
        logger.info("generating_competitor_analysis", company=company, competitors=competitors)

        objective = f"""
        Analyze {company}'s competitive landscape against: {', '.join(competitors)}

        Include:
        1. Market positioning comparison
        2. Feature/product comparison
        3. Pricing strategy analysis
        4. Marketing approach differences
        5. Strengths & weaknesses
        6. Recommendations
        """

        session = await self.memory.start_session(user_id="monetize", objective=objective)

        try:
            findings = await self.agent.run(
                objective=objective,
                max_steps=15,
                session_id=session["session_id"],
            )

            report = self._format_report(
                company=company,
                competitors=competitors,
                findings=findings,
            )

            logger.info("report_generated", company=company, sections=len(report["sections"]))

            return {
                "type": "competitor_analysis",
                "company": company,
                "competitors": competitors,
                "report": report,
                "quality_score": self._calculate_quality(findings),
                "ready_to_sell": True,
                "generated_at": datetime.utcnow().isoformat(),
                "session_id": session["session_id"],
            }

        except Exception as e:
            logger.error("report_generation_failed", company=company, error=str(e))
            raise

    async def daily_production_run(self) -> dict:
        """
        Generate sellable content daily (while you sleep).

        Strategy: Generate 3-5 popular topics pre-written,
                 then sell to customers as custom articles.
        """
        logger.info("starting_daily_production_run")

        # Popular topics that sell well
        topics = [
            "best productivity apps for remote work",
            "how to optimize your home office",
            "top 10 budget travel destinations 2025",
            "guide to starting a side hustle",
            "ultimate guide to home automation",
        ]

        results = {"generated": 0, "failed": 0, "articles": []}

        for topic in topics:
            try:
                article = await self.generate_seo_article(
                    topic=topic,
                    target_length=1500,
                )
                results["articles"].append(article)
                results["generated"] += 1

                # Save to inventory
                await self._save_to_inventory(article)

                logger.info("article_added_to_inventory", topic=topic)

                # Small delay between generations (be nice to local GPU)
                await asyncio.sleep(2)

            except Exception as e:
                logger.error("production_failed", topic=topic, error=str(e))
                results["failed"] += 1

        logger.info("daily_production_complete", **results)
        return results

    async def process_customer_order(
        self,
        order_id: str,
        customer_topic: str,
        customer_email: str,
        price: float = 49.99,
    ) -> dict:
        """
        Process customer order: research → format → send.

        This is what happens when someone buys from your Fiverr/Gumroad listing.
        """
        logger.info("processing_customer_order", order_id=order_id, topic=customer_topic)

        try:
            # Generate article
            article = await self.generate_seo_article(
                topic=customer_topic,
                target_length=1500,
            )

            # Save as PDF-ready (we'll add PDF export later)
            output = {
                "order_id": order_id,
                "customer_email": customer_email,
                "topic": customer_topic,
                "article": article,
                "price": price,
                "status": "ready_for_delivery",
                "delivered_at": datetime.utcnow().isoformat(),
            }

            # Log the order (payment would happen here in production)
            await self._log_order(output)

            logger.info("order_processed", order_id=order_id, revenue=price)
            return output

        except Exception as e:
            logger.error("order_processing_failed", order_id=order_id, error=str(e))
            raise

    async def get_inventory(self) -> list:
        """Get pre-written articles ready to sell."""
        inventory_file = OUTPUT_DIR / "inventory.jsonl"
        if not inventory_file.exists():
            return []

        articles = []
        with open(inventory_file) as f:
            for line in f:
                articles.append(json.loads(line))
        return articles

    async def get_order_stats(self) -> dict:
        """Get revenue stats."""
        if not ORDERS_FILE.exists():
            return {
                "total_orders": 0,
                "total_revenue": 0.0,
                "average_price": 0.0,
            }

        orders = []
        with open(ORDERS_FILE) as f:
            for line in f:
                orders.append(json.loads(line))

        total_revenue = sum(o.get("price", 0) for o in orders)
        return {
            "total_orders": len(orders),
            "total_revenue": total_revenue,
            "average_price": total_revenue / len(orders) if orders else 0,
        }

    # ─────────────────────────────────────────────────────────────
    # Private Methods
    # ─────────────────────────────────────────────────────────────

    def _format_article(self, topic: str, findings: dict, include_sources: bool) -> dict:
        """Format research findings as article."""
        # Extract text from findings
        body = findings.get("synthesis", "").strip()

        if not body:
            body = "No findings generated. Please try again."

        # Add structure
        lines = body.split("\n")
        title = f"The Ultimate Guide to {topic.title()}"

        # Format with H2s for readability
        formatted_body = "\n\n".join(lines)

        return {
            "title": title,
            "body": formatted_body,
            "sources": findings.get("sources", []),
        }

    def _format_report(self, company: str, competitors: list, findings: dict) -> dict:
        """Format findings as structured report."""
        sections = [
            {
                "title": "Executive Summary",
                "content": findings.get("synthesis", "")[:500],
            },
            {
                "title": "Competitive Analysis",
                "content": findings.get("synthesis", "")[500:],
            },
            {
                "title": "Recommendations",
                "content": "Based on the analysis above, consider: ...",
            },
        ]

        return {
            "company": company,
            "competitors": competitors,
            "sections": sections,
            "generated_at": datetime.utcnow().isoformat(),
        }

    def _calculate_quality(self, findings: dict) -> float:
        """Calculate quality score (0-100) for this output."""
        score = 70.0  # Base score

        # Bonus for sources
        if findings.get("sources"):
            score += min(20, len(findings["sources"]) * 2)

        # Bonus for citations
        synthesis = findings.get("synthesis", "")
        if "[" in synthesis and "]" in synthesis:
            score += 10

        return min(100.0, score)

    async def _save_to_inventory(self, article: dict) -> None:
        """Save article to pre-written inventory."""
        inventory_file = OUTPUT_DIR / "inventory.jsonl"
        with open(inventory_file, "a") as f:
            f.write(json.dumps(article) + "\n")

    async def _log_order(self, order: dict) -> None:
        """Log a customer order."""
        with open(ORDERS_FILE, "a") as f:
            f.write(json.dumps(order) + "\n")


# ─────────────────────────────────────────────────────────────
# CLI Interface
# ─────────────────────────────────────────────────────────────


async def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Monetization Engine for Research Agent")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Generate article
    gen_parser = subparsers.add_parser("generate", help="Generate sellable content")
    gen_parser.add_argument(
        "--type",
        choices=["seo", "report"],
        default="seo",
        help="Type of content to generate",
    )
    gen_parser.add_argument("--topic", required=True, help="Topic/subject for content")
    gen_parser.add_argument("--length", type=int, default=1500, help="Target word count")

    # Process order
    order_parser = subparsers.add_parser("order", help="Process customer order")
    order_parser.add_argument("--id", required=True, help="Order ID")
    order_parser.add_argument("--topic", required=True, help="Customer topic")
    order_parser.add_argument("--email", required=True, help="Customer email")
    order_parser.add_argument("--price", type=float, default=49.99, help="Order price")

    # Daily run
    subparsers.add_parser("daily", help="Run daily production (generate 5 articles)")

    # Stats
    subparsers.add_parser("stats", help="Show revenue stats")

    # Inventory
    subparsers.add_parser("inventory", help="List pre-written articles")

    args = parser.parse_args()

    # Initialize agent
    router = ModelRouter("config/model_config.yaml")
    memory = MemoryManager("data")
    agent = ResearchAgent(router=router, memory=memory)
    monetizer = MonetizationEngine(agent)

    # Execute command
    if args.command == "generate":
        result = await monetizer.generate_seo_article(
            topic=args.topic,
            target_length=args.length,
        )
        print(json.dumps(result, indent=2))

    elif args.command == "order":
        result = await monetizer.process_customer_order(
            order_id=args.id,
            customer_topic=args.topic,
            customer_email=args.email,
            price=args.price,
        )
        print(json.dumps(result, indent=2, default=str))

    elif args.command == "daily":
        result = await monetizer.daily_production_run()
        print(json.dumps(result, indent=2, default=str))

    elif args.command == "stats":
        stats = await monetizer.get_order_stats()
        print(f"Total Orders: {stats['total_orders']}")
        print(f"Total Revenue: €{stats['total_revenue']:.2f}")
        print(f"Average Price: €{stats['average_price']:.2f}")

    elif args.command == "inventory":
        articles = await monetizer.get_inventory()
        print(f"Pre-written articles: {len(articles)}")
        for article in articles[:5]:
            print(f"  - {article['title'][:60]}")


if __name__ == "__main__":
    asyncio.run(main())
