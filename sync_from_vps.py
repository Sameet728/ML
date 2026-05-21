#!/usr/bin/env python3
"""
sync_from_vps.py
================
Downloads all pipeline output files from VPS to local machine.
Run this on your LAPTOP after the VPS pipeline completes.

Usage:
    python sync_from_vps.py --host YOUR_VPS_IP --key path/to/key.pem
    python sync_from_vps.py --host YOUR_VPS_IP          # password auth
"""
import argparse
import subprocess
import sys
from pathlib import Path

LOCAL_BASE = Path(__file__).parent

# Files/folders to download from VPS
SYNC_TARGETS = [
    "reports/",                        # All backtest results, HTML report, charts
    "models/saved/",                   # Trained model files
    "data/processed/",                 # Processed feature matrix
    "logs/",                           # Run logs
    ".cache/optuna_studies.db",        # Optuna SQLite DB (resume support)
    "reports/optuna_best_params.json", # Optuna JSON cache
]


def rsync(src: str, dst: str, key: str | None):
    """Run rsync with optional SSH key."""
    ssh_opt = f"-e 'ssh -i {key} -o StrictHostKeyChecking=no'" if key else \
              "-e 'ssh -o StrictHostKeyChecking=no'"
    cmd = f'rsync -avz --progress {ssh_opt} {src} {dst}'
    print(f"\n  Syncing: {src}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"  WARNING: rsync failed for {src} (may not exist yet)")


def main():
    parser = argparse.ArgumentParser(description="Sync VPS results to local machine")
    parser.add_argument("--host",  required=True, help="VPS IP address")
    parser.add_argument("--user",  default="ubuntu", help="SSH username (default: ubuntu)")
    parser.add_argument("--key",   default=None, help="Path to SSH key .pem file")
    parser.add_argument("--remote-dir", default="~/BTCML", help="Remote project dir")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Syncing from {args.user}@{args.host}:{args.remote_dir}")
    print("=" * 60)

    for target in SYNC_TARGETS:
        remote = f"{args.user}@{args.host}:{args.remote_dir}/{target}"
        # Mirror directory structure locally
        local_rel = target.rstrip("/")
        local_dst  = LOCAL_BASE / Path(local_rel).parent
        local_dst.mkdir(parents=True, exist_ok=True)
        rsync(remote, str(local_dst) + "/", args.key)

    print("\n" + "=" * 60)
    print("  Sync complete!")
    print(f"  Reports:   {LOCAL_BASE / 'reports'}")
    print(f"  Open HTML: reports/backtest_report.html")
    print(f"  Dashboard: python run_pipeline.py --mode=dashboard")
    print("=" * 60)


if __name__ == "__main__":
    main()
