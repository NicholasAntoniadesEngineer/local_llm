#!/bin/bash
# Single entry point: Run improvement cycles continuously

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

source mlx_agent_env/bin/activate

echo "🚀 Self-Improving Agent System"
echo "================================"
echo ""

CYCLE=1

while true; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Cycle #$CYCLE - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    python3 improve.py $CYCLE
    RESULT=$?

    if [ $RESULT -eq 0 ]; then
        echo "✅ Cycle #$CYCLE completed"
    else
        echo "⚠️  Cycle #$CYCLE encountered issues"
    fi

    echo ""
    echo "Latest commit:"
    git log --oneline | head -1

    echo ""
    CYCLE=$((CYCLE + 1))

    # Show summary every 5 cycles
    if [ $((CYCLE % 5)) -eq 1 ]; then
        echo ""
        echo "📈 Progress:"
        echo "   Cycles completed: $((CYCLE - 1))"
        echo "   Features added:"
        git log --oneline | grep "feat(cycle" | wc -l
        echo ""
    fi

    sleep 3
done
