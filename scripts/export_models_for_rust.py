#!/usr/bin/env python3
"""Export Flask xG XGBoost models to compact JSON for the pure-Rust tree scorer.

Run once (and on any retrain) from anywhere:
    python scripts/export_models_for_rust.py

Output: `Model/rust/{model_basename}.json` for every `Model/xg*_*.pkl`.

Why this format:
  - Uses `Booster.save_raw(raw_format='json')`, the canonical serialized
    representation that `predict` actually consumes, so the Rust tree walk is
    structurally identical to XGBoost's (no re-indexing or dump-text parsing).
  - The `base_score` in the model config is NOT what old pickled models use at
    predict time (the reloaded booster applies a different offset). We
    calibrate `base_score_effective` empirically: predict an all-zero
    reference row (output_margin=True) and subtract our tree sum.

JSON schema:
  {
    "features": [one-hot column names...],
    "base_score": <calibrated effective base>,
    "objective": "binary:logistic",
    "trees": [ {
        "split_indices": [...],   // feature index per node (-1 => leaf)
        "split_conditions": [...],// threshold per node
        "left": [...], "right": [...],
        "default_left": [...],    // 1 => missing goes left
        "leaf": [...]             // leaf weight per node (from base_weights)
    }, ...]
  }

Rust scorer (`nhl_rust/src/models/xg.rs`): walk each tree from node 0;
leaf if left==-1 (value = leaf[i]); else value = row[split_indices[i]],
NaN -> default_left, value < split_condition -> left, else right.
margin = base_score + sum(leaf values); probability = sigmoid(margin).
"""
import glob
import json
import math
import os

import joblib
import numpy as np
import xgboost as xgb

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "Model")
OUT_DIR = os.path.join(MODEL_DIR, "rust")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    files = sorted(glob.glob(os.path.join(MODEL_DIR, "xg*_*.pkl")))
    if not files:
        raise SystemExit("No xg*_*.pkl models found in Model/")
    converted = 0
    for path in files:
        model = joblib.load(path)
        booster = model.get_booster()
        fnames_attr = getattr(model, "feature_names_in_", None)
        features = list(fnames_attr) if fnames_attr is not None else []
        raw = json.loads(booster.save_raw(raw_format='json'))
        learner = raw['learner']
        trees_raw = learner['gradient_booster']['model']['trees']

        trees = []
        for t in trees_raw:
            trees.append({
                "split_indices": t["split_indices"],
                "split_conditions": t["split_conditions"],
                "left": t["left_children"],
                "right": t["right_children"],
                "default_left": t["default_left"],
                "leaf": t["base_weights"],
            })

        def score_tree(row, t):
            i = 0
            while True:
                lc = t["left"][i]
                if lc == -1:
                    return t["leaf"][i]
                f = t["split_indices"][i]
                v = row[f]
                if math.isnan(v):
                    i = t["left"][i] if t["default_left"][i] == 1 else t["right"][i]
                elif v < t["split_conditions"][i]:
                    i = lc
                else:
                    i = t["right"][i]

        def tree_sum(row):
            return sum(score_tree(row, t) for t in trees)

        zero = [0.0] * len(features)
        d0 = xgb.DMatrix(np.asarray([zero], dtype=np.float32), feature_names=features)
        base_score = float(booster.predict(d0, output_margin=True)[0]) - tree_sum(zero)

        out = {
            "features": features,
            "base_score": base_score,
            "objective": "binary:logistic",
            "trees": trees,
        }
        fname = os.path.basename(path)
        out_path = os.path.join(OUT_DIR, fname + ".json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f)
        converted += 1
        print(f"converted {fname} -> {os.path.basename(out_path)} "
              f"({len(trees)} trees, {len(features)} feats, base={base_score:.6f})")
    print(f"Done: {converted} models -> {OUT_DIR}")


if __name__ == "__main__":
    main()
