"""
save_optuna_cache.py
====================
Parses the current run log to extract the best Optuna params for each
completed fold and saves them to reports/optuna_best_params.json.

On the next run, walk_forward.py will load this cache and skip Optuna
for any fold that already has saved params.
"""
import re
import ast
import json
from pathlib import Path

LOG_PATH  = Path(r"c:\Users\samee\Desktop\BTCML\logs\platform_2026-05-21.log")
OUT_PATH  = Path(r"c:\Users\samee\Desktop\BTCML\reports\optuna_best_params.json")

fold_params = {}
current_fold = None

with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        # Detect fold start
        m = re.search(r"Fold (\d+)/17", line)
        if m:
            current_fold = int(m.group(1))

        # Capture best params line (logged after Optuna finishes each fold)
        m2 = re.search(r"Best params: (\{.+\})", line)
        if m2 and current_fold is not None:
            try:
                params = ast.literal_eval(m2.group(1))
                fold_params[str(current_fold)] = params
                print(f"  OK Fold {current_fold:2d}: {len(params)} params captured")
            except Exception as e:
                print(f"  ERR Fold {current_fold:2d}: parse error - {e}")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(fold_params, f, indent=2)

print(f"\nSaved {len(fold_params)} fold(s) -> {OUT_PATH}")
print(f"Folds captured: {sorted(int(k) for k in fold_params)}")
