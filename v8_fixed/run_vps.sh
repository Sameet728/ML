#!/bin/bash
# =============================================================================
# run_vps.sh — Start or resume the pipeline on VPS inside a screen session
# Usage: bash run_vps.sh
# =============================================================================

cd ~/BTCML
source venv/bin/activate

TODAY=$(date +%Y-%m-%d)
LOG="logs/platform_${TODAY}.log"

echo "Starting pipeline... Log: $LOG"
echo "Detach with Ctrl+A then D"
echo "Reattach with: screen -r btcml"

# Kill old session if exists
screen -S btcml -X quit 2>/dev/null || true

# Start new session
screen -dmS btcml bash -c "
    cd ~/BTCML
    source venv/bin/activate
    python run_pipeline.py 2>&1 | tee -a $LOG
    echo 'Pipeline finished!' | tee -a $LOG
"

echo "Pipeline running in background (screen session: btcml)"
echo ""
echo "Monitor: tail -f $LOG"
echo "Attach:  screen -r btcml"
