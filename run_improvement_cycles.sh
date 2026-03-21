#!/bin/bash
# Self-Improvement Cycle Launcher
# Run this to start the agent improving itself

set -e

PROJECT_DIR="/Users/nicholasantoniades/Documents/GitHub/local_llm"
cd "$PROJECT_DIR"

echo "=========================================="
echo "🤖 SELF-IMPROVING AGENT SYSTEM"
echo "=========================================="
echo ""

# Check venv
if [ ! -d "mlx_agent_env" ]; then
    echo "❌ Virtual environment not found"
    exit 1
fi

source mlx_agent_env/bin/activate

# Get number of cycles from argument or default to 1
CYCLES=${1:-1}

echo "📊 Starting $CYCLES improvement cycle(s)..."
echo ""

for i in $(seq 1 $CYCLES); do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "CYCLE $i / $CYCLES"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # Run self-improvement
    python3 self_improve.py

    # Check for improvements created
    IMPROVEMENTS=$(ls -1 agent_outputs/improvement_v*.py 2>/dev/null | wc -l)
    echo ""
    echo "✅ Cycle $i complete. Improvements created: $IMPROVEMENTS"

    if [ $i -lt $CYCLES ]; then
        echo ""
        echo "Waiting before next cycle..."
        sleep 3
    fi
done

echo ""
echo "=========================================="
echo "🎉 Self-improvement cycles complete!"
echo "=========================================="
echo ""
echo "📁 Check improvements in: agent_outputs/"
echo "📊 Review changes:        git diff agent.py reflection.py memory.py"
echo "🔄 Run more cycles:       bash run_improvement_cycles.sh 3"
echo ""
