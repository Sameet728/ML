#!/bin/bash
# =============================================================================
# setup_vps.sh — One-shot VPS setup for BTCML AI Quant Research Platform
# Run this ONCE after cloning the repo on your Oracle VPS.
# =============================================================================
set -e

echo "============================================================"
echo "  BTCML VPS Setup"
echo "============================================================"

# ── 1. System packages ────────────────────────────────────────────────────────
echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y python3.11 python3.11-venv python3-pip screen git curl wget htop

# ── 2. Virtual environment ────────────────────────────────────────────────────
echo "[2/6] Creating virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# ── 3. Upgrade pip ────────────────────────────────────────────────────────────
echo "[3/6] Upgrading pip..."
pip install --upgrade pip setuptools wheel

# ── 4. Install dependencies ───────────────────────────────────────────────────
echo "[4/6] Installing Python dependencies..."
pip install -r requirements.txt

# ── 5. Create required directories ────────────────────────────────────────────
echo "[5/6] Creating directories..."
mkdir -p data/raw data/processed models/saved reports logs .cache

# ── 6. Done ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  To run the pipeline:"
echo "    screen -S btcml"
echo "    source venv/bin/activate"
echo "    python run_pipeline.py"
echo "    (Ctrl+A then D to detach)"
echo ""
echo "  To check progress:"
echo "    screen -r btcml"
echo "    tail -f logs/\$(date +%Y-%m-%d | xargs -I{} echo 'platform_{}.log')"
echo "============================================================"
