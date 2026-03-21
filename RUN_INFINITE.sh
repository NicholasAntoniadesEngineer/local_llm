#!/bin/bash
# Better way to run infinite self-improvement with visible output

cd /Users/nicholasantoniades/Documents/GitHub/local_llm

source mlx_agent_env/bin/activate

echo "🚀 Starting infinite self-improvement..."
echo ""

# Counter for cycles
CYCLE=1

while true; do
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "CYCLE #$CYCLE - $(date '+%H:%M:%S')"
    echo "════════════════════════════════════════════════════════════════"
    echo ""

    # Run one improvement cycle with full output
    python3 self_improve.py $CYCLE

    # Get git status
    echo ""
    echo "📊 Git status:"
    git log --oneline | head -1

    # Wait a moment before next cycle
    echo ""
    echo "⏳ Waiting 5 seconds before next cycle..."
    sleep 5

    CYCLE=$((CYCLE + 1))

    # Show progress every 10 cycles
    if [ $((CYCLE % 10)) -eq 0 ]; then
        echo ""
        echo "🎉 Completed $((CYCLE-1)) improvement cycles!"
        echo "📈 All improvements committed to git"
        echo ""
    fi
done
