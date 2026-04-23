from __future__ import annotations

import argparse
from dataclasses import dataclass
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.Game_Projection_Model import (
    _load_csvs,
    add_b2b,
    add_situation,
    build_fixed_preseason_model,
    build_player_game_box_stats,
    build_preseason_updating_features,
)


BASE_FEATURE_COLS = [
    "poss_value",
    "off_the_puck",
    "gax",
    "goalie_gsax",
    "rookie_F",
    "rookie_D",
    "rookie_G",
]


@dataclass
class SplitData:
    train_logits: np.ndarray
    test_logits: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def summarize_logits(name: str, train_logits: np.ndarray, test_logits: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> dict[str, float | str]:
    train_prob = expit(train_logits)
    test_prob = expit(test_logits)
    train_fav = np.maximum(train_prob, 1.0 - train_prob)
    test_fav = np.maximum(test_prob, 1.0 - test_prob)
    return {
        "name": name,
        "train_logloss": float(log_loss(y_train, train_prob)),
        "test_logloss": float(log_loss(y_test, test_prob)),
        "train_brier": float(brier_score_loss(y_train, train_prob)),
        "test_brier": float(brier_score_loss(y_test, test_prob)),
        "train_auc": float(roc_auc_score(y_train, train_prob)),
        "test_auc": float(roc_auc_score(y_test, test_prob)),
        "train_mean_fav_prob": float(train_fav.mean()),
        "test_mean_fav_prob": float(test_fav.mean()),
        "test_pct_favorites_ge_55": float((test_fav >= 0.55).mean()),
        "test_pct_favorites_ge_60": float((test_fav >= 0.60).mean()),
    }


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    def objective(theta: np.ndarray) -> float:
        temp = np.exp(theta[0])
        prob = expit(logits / temp)
        eps = 1e-12
        return float(-np.sum(y_true * np.log(prob + eps) + (1.0 - y_true) * np.log(1.0 - prob + eps)))

    result = minimize(objective, x0=np.array([0.0]), method="L-BFGS-B", bounds=[(-3.0, 3.0)])
    return float(np.exp(result.x[0]))


def fit_affine(logits: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    logits = np.asarray(logits, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    def objective(theta: np.ndarray) -> float:
        a, b = theta
        prob = expit(a * logits + b)
        eps = 1e-12
        return float(-np.sum(y_true * np.log(prob + eps) + (1.0 - y_true) * np.log(1.0 - prob + eps)))

    result = minimize(objective, x0=np.array([1.0, 0.0]), method="L-BFGS-B")
    return float(result.x[0]), float(result.x[1])


def fit_favorite_shift(logits: np.ndarray, y_true: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    sign = np.sign(logits)

    def objective(alpha_arr: np.ndarray) -> float:
        alpha = alpha_arr[0]
        prob = expit(logits + alpha * sign)
        eps = 1e-12
        return float(-np.sum(y_true * np.log(prob + eps) + (1.0 - y_true) * np.log(1.0 - prob + eps)))

    result = minimize(objective, x0=np.array([0.0]), method="L-BFGS-B", bounds=[(0.0, 1.0)])
    return float(result.x[0])


def build_feature_table() -> pd.DataFrame:
    games, pvm, skaters, goalies, _gamescore = _load_csvs(include_gamescore=True)
    games = add_situation(add_b2b(games.copy()))
    player_game_box_stats = build_player_game_box_stats(pvm, skaters, goalies, games)
    features = build_preseason_updating_features(games, player_game_box_stats)
    seasons = sorted(features["season"].astype(str).unique())
    train_seasons = seasons[1:]
    features = features[features["season"].astype(str).isin(train_seasons)].copy()

    model, feature_cols, _coef_df = build_fixed_preseason_model()
    for col in BASE_FEATURE_COLS:
        features[f"diff_{col}"] = features[f"home_{col}"] - features[f"away_{col}"]

    features["base_logit"] = model.decision_function(features[feature_cols]) + features["situation_offset"].to_numpy(dtype=float)
    return features


def evaluate_random_split(features: pd.DataFrame, seed: int) -> pd.DataFrame:
    X = features[["base_logit"]].copy()
    y = features["home_win"].astype(int).copy()
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    train_logits = X_train["base_logit"].to_numpy(dtype=float)
    test_logits = X_test["base_logit"].to_numpy(dtype=float)
    y_train_arr = y_train.to_numpy(dtype=int)
    y_test_arr = y_test.to_numpy(dtype=int)

    results = [summarize_logits("baseline", train_logits, test_logits, y_train_arr, y_test_arr)]

    for alpha in (0.05, 0.10, 0.15, 0.20, 0.25):
        shifted_train = train_logits + alpha * np.sign(train_logits)
        shifted_test = test_logits + alpha * np.sign(test_logits)
        results.append(summarize_logits(f"favorite_shift_{alpha:.2f}", shifted_train, shifted_test, y_train_arr, y_test_arr))

    fit_logits, cal_logits, y_fit, y_cal = train_test_split(train_logits, y_train_arr, test_size=0.25, random_state=seed + 1)
    temp = fit_temperature(cal_logits, y_cal)
    results.append(summarize_logits(f"temperature_T={temp:.3f}", train_logits / temp, test_logits / temp, y_train_arr, y_test_arr))

    affine_a, affine_b = fit_affine(cal_logits, y_cal)
    results.append(summarize_logits(f"affine_a={affine_a:.3f}_b={affine_b:.3f}", affine_a * train_logits + affine_b, affine_a * test_logits + affine_b, y_train_arr, y_test_arr))

    tuned_alpha = fit_favorite_shift(cal_logits, y_cal)
    results.append(summarize_logits(f"tuned_favorite_shift_{tuned_alpha:.3f}", train_logits + tuned_alpha * np.sign(train_logits), test_logits + tuned_alpha * np.sign(test_logits), y_train_arr, y_test_arr))

    out = pd.DataFrame(results)
    out.insert(0, "split", f"random_seed_{seed}")
    return out


def evaluate_season_holdout(features: pd.DataFrame, test_season: str) -> pd.DataFrame:
    features = features.copy()
    train_mask = features["season"].astype(str) < str(test_season)
    test_mask = features["season"].astype(str) == str(test_season)
    train_logits = features.loc[train_mask, "base_logit"].to_numpy(dtype=float)
    test_logits = features.loc[test_mask, "base_logit"].to_numpy(dtype=float)
    y_train = features.loc[train_mask, "home_win"].to_numpy(dtype=int)
    y_test = features.loc[test_mask, "home_win"].to_numpy(dtype=int)

    results = [summarize_logits("baseline", train_logits, test_logits, y_train, y_test)]

    for alpha in (0.05, 0.10, 0.15, 0.20, 0.25):
        results.append(
            summarize_logits(
                f"favorite_shift_{alpha:.2f}",
                train_logits + alpha * np.sign(train_logits),
                test_logits + alpha * np.sign(test_logits),
                y_train,
                y_test,
            )
        )

    temp = fit_temperature(train_logits, y_train)
    results.append(summarize_logits(f"temperature_T={temp:.3f}", train_logits / temp, test_logits / temp, y_train, y_test))

    affine_a, affine_b = fit_affine(train_logits, y_train)
    results.append(summarize_logits(f"affine_a={affine_a:.3f}_b={affine_b:.3f}", affine_a * train_logits + affine_b, affine_a * test_logits + affine_b, y_train, y_test))

    tuned_alpha = fit_favorite_shift(train_logits, y_train)
    results.append(summarize_logits(f"tuned_favorite_shift_{tuned_alpha:.3f}", train_logits + tuned_alpha * np.sign(train_logits), test_logits + tuned_alpha * np.sign(test_logits), y_train, y_test))

    out = pd.DataFrame(results)
    out.insert(0, "split", f"season_holdout_{test_season}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate favorite-boost adjustments on preseason-updating game probabilities")
    parser.add_argument("--random-seed", type=int, default=0, help="Random split seed")
    parser.add_argument("--season-holdout", default="20252026", help="Season to evaluate as strict holdout")
    args = parser.parse_args()

    features = build_feature_table()
    random_df = evaluate_random_split(features, seed=args.random_seed)
    season_df = evaluate_season_holdout(features, test_season=str(args.season_holdout))
    results = pd.concat([random_df, season_df], ignore_index=True)

    for split_name, split_df in results.groupby("split", sort=False):
        baseline = split_df.loc[split_df["name"] == "baseline"].iloc[0]
        ordered = split_df.copy()
        ordered["delta_test_logloss"] = ordered["test_logloss"] - float(baseline["test_logloss"])
        ordered["delta_test_brier"] = ordered["test_brier"] - float(baseline["test_brier"])
        ordered = ordered.sort_values(["test_logloss", "test_brier", "test_auc"], ascending=[True, True, False])
        print(f"\n=== {split_name} ===")
        print(ordered.to_string(index=False))


if __name__ == "__main__":
    main()