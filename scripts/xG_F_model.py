import os
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text
import requests as req
import pandas as pd

username = 'root'
password = 'Sunesen1' # MySQL password
host = 'localhost'  # or your remote host
port = '3306'       # default MySQL port
database = 'public'

# Allow environment variable overrides
username = os.getenv('DB_USER', username)
password = os.getenv('DB_PASSWORD', password)
host = os.getenv('DB_HOST', host)
port = os.getenv('DB_PORT', port)
database = os.getenv('DB_NAME', database)

# Create the engine
engine = create_engine(f'mysql+mysqlconnector://{username}:{password}@{host}:{port}/{database}')

query = text('SELECT * FROM model_fenwick')

# Execute the query and load into a DataFrame
df = pd.read_sql(query, con=engine)

import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, log_loss
import joblib

# Save directory for models
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'Model')
os.makedirs(MODEL_DIR, exist_ok=True)

# Define features and target
feature_cols = ["Venue", "shotType2", "ScoreState2", "RinkVenue", "StrengthState2", "BoxID2", "LastEvent"]
target_col = "Goal"

# Fill missing values and convert to string
df[feature_cols] = df[feature_cols].fillna("missing").astype(str)
df["Season"] = df["Season"].astype(str).str.strip()

# Sort seasons
seasons = sorted(df["Season"].unique())
window_size = 3
models = {}
results = []

# Replace RinkVenue in test seasons with synthetic weighted labels
def build_weighted_rinkvenue_map(df, train_seasons):
    weights = [1/6, 1/3, 1/2]
    rink_map = {}
    for season, weight in zip(train_seasons, weights):
        season_df = df[df["Season"] == season]
        for rink in season_df["RinkVenue"].unique():
            if "-" not in rink:
                continue
            team = rink.split("-")[1]
            rink_map.setdefault(team, []).append((season, weight))
    return rink_map

for i in range(len(seasons) - window_size + 1):
    train_seasons = seasons[i:i+window_size]
    test_season_index = i + window_size
    test_season = seasons[test_season_index] if test_season_index < len(seasons) else None
    model_name = f"{train_seasons[0]}_{train_seasons[-1]}"
    
    # Replace RinkVenue values in test season
    if test_season:
        rink_map = build_weighted_rinkvenue_map(df, train_seasons)
        test_idx = df["Season"] == test_season
        for idx in df[test_idx].index:
            original_rink = df.at[idx, "RinkVenue"]
            if "-" in original_rink:
                team = original_rink.split("-")[1]
                df.at[idx, "RinkVenue"] = f"weighted_{team}"

# Create dummy variables
X_full = pd.get_dummies(df[feature_cols]).astype(float)
y_full = df[target_col]

# Main loop
for i in range(len(seasons) - window_size + 1):
    train_seasons = seasons[i:i+window_size]
    test_season_index = i + window_size
    test_season = seasons[test_season_index] if test_season_index < len(seasons) else None
    model_name = f"{train_seasons[0]}_{train_seasons[-1]}"
    
    # Split training data
    train_idx = df["Season"].isin(train_seasons)
    X_train = X_full[train_idx].copy()
    y_train = y_full[train_idx]
    X_train = X_train.astype(float)
    
    # Train model
    model = XGBClassifier(eval_metric="logloss")
    model.fit(X_train, y_train)
    models[model_name] = model
    joblib.dump(model, os.path.join(MODEL_DIR, f"xgb_{model_name}.pkl"))
    
    # Predict for training seasons
    for season in train_seasons:
        season_idx = df["Season"] == season
        X_season = X_full[season_idx].copy()
        missing_cols = set(X_train.columns) - set(X_season.columns)
        for col in missing_cols:
            X_season[col] = 0
        extra_cols = set(X_season.columns) - set(X_train.columns)
        X_season.drop(columns=extra_cols, inplace=True)
        X_season = X_season[X_train.columns].astype(float)
        df.loc[season_idx, f"xG_{model_name}"] = model.predict_proba(X_season)[:, 1]
    
    # Predict for test season if available
    if test_season:
        test_idx = df["Season"] == test_season
        X_test = X_full[test_idx].copy()
        
        # Reconstruct weighted RinkVenue values
        for idx in X_test.index:
            rink_label = df.at[idx, "RinkVenue"]
            if rink_label.startswith("weighted_"):
                team = rink_label.split("_")[1]
                for season, weight in zip(train_seasons, [1/6, 1/3, 1/2]):
                    col_name = f"RinkVenue_{season}-{team}"
                    if col_name in X_test.columns:
                        X_test.at[idx, col_name] = weight
        
        # Drop synthetic RinkVenue columns
        drop_cols = [col for col in X_test.columns if col.startswith("RinkVenue_weighted_")]
        X_test.drop(columns=drop_cols, inplace=True, errors='ignore')
        
        # Align columns
        missing_cols = set(X_train.columns) - set(X_test.columns)
        for col in missing_cols:
            X_test[col] = 0
        extra_cols = set(X_test.columns) - set(X_train.columns)
        X_test.drop(columns=extra_cols, inplace=True)
        X_test = X_test[X_train.columns].astype(float)
        
        # Predict and evaluate
        df.loc[test_idx, f"xG_test_{model_name}"] = model.predict_proba(X_test)[:, 1]
        y_pred = df.loc[test_idx, f"xG_test_{model_name}"]
        y_true = y_full[test_idx]
        mask = (~y_pred.isna()) & (~y_true.isna())
        y_pred_clean = y_pred[mask]
        y_true_clean = y_true[mask]
        
        if len(y_true_clean) > 0:
            auc = roc_auc_score(y_true_clean, y_pred_clean)
            logloss = log_loss(y_true_clean, y_pred_clean)
            results.append({
                "Model": model_name,
                "TestSeason": test_season,
                "AUC": round(auc, 4),
                "LogLoss": round(logloss, 4),
                "Samples": len(y_true_clean)
            })
        else:
            print(f"Skipping {model_name} → {test_season}: no valid samples.")
    else:
        print(f"{model_name}: no test season available, model trained for future use.")

# Final xG average
xg_cols = [col for col in df.columns if col.startswith("xG_")]
df["xG_avg"] = df[xg_cols].mean(axis=1)

# Summary table
results_df = pd.DataFrame(results)
print(results_df)