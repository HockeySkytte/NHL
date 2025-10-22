import os
import argparse
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text
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

def load_dataframe(sql: str) -> pd.DataFrame:
	try:
		q = text(sql)
		return pd.read_sql(q, con=engine)
	except SQLAlchemyError as e:
		raise SystemExit(f"Failed to run SQL: {e}")

import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib

# Save directory for models
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'Model')
os.makedirs(MODEL_DIR, exist_ok=True)

def build_and_train(df: pd.DataFrame,
					feature_cols: list[str],
					target_col: str,
					test_size: float = 0.2,
					random_state: int = 42,
					model_params: dict | None = None):
	if model_params is None:
		model_params = dict(
			n_estimators=800,
			learning_rate=0.03,
			subsample=0.9,
			colsample_bytree=0.8,
			max_depth=5,
			reg_lambda=1.0,
			reg_alpha=0.0,
			random_state=random_state,
			tree_method='hist',
			eval_metric='logloss',
			n_jobs=0,
		)

	# Split features: categorical vs numeric
	cat_cols = [c for c in feature_cols if c.lower() == 'situation']
	num_cols = [c for c in feature_cols if c not in cat_cols]

	df = df.copy()
	# Ensure target is binary 0/1
	if df[target_col].dtype != np.int64 and df[target_col].dtype != np.int32:
		df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
	# Replace infs
	df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

	X = df[feature_cols]
	y = df[target_col].astype(int)

	# Preprocess: impute numeric; one-hot encode Situation
	numeric_tf = Pipeline(steps=[
		('imputer', SimpleImputer(strategy='median')),
	])
	categorical_tf = Pipeline(steps=[
		('imputer', SimpleImputer(strategy='most_frequent')),
		('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False)),
	])

	pre = ColumnTransformer(
		transformers=[
			('num', numeric_tf, num_cols),
			('cat', categorical_tf, cat_cols),
		]
	)

	model = XGBClassifier(**model_params)

	pipe = Pipeline(steps=[
		('pre', pre),
		('model', model),
	])

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=test_size, random_state=random_state, stratify=y
	)

	pipe.fit(X_train, y_train)
	# Predictions for test and train
	proba_test = pipe.predict_proba(X_test)[:, 1]
	proba_train = pipe.predict_proba(X_train)[:, 1]

	# Metrics
	metrics = {
		'test_auc': roc_auc_score(y_test, proba_test),
		'test_logloss': log_loss(y_test, proba_test),
		'train_auc': roc_auc_score(y_train, proba_train),
		'train_logloss': log_loss(y_train, proba_train),
		'test_size': test_size,
		'features': feature_cols,
		'cat_cols': cat_cols,
		'num_cols': num_cols,
	}

	# Feature importances (expanded and grouped)
	pre = pipe.named_steps['pre']
	model = pipe.named_steps['model']
	try:
		out_names = list(pre.get_feature_names_out())
	except Exception:
		# Fallback: unknown names, index by position
		out_names = [f'f{i}' for i in range(len(model.feature_importances_))]
	importances = model.feature_importances_.tolist()

	# Expanded importances
	expanded = sorted([
		{'feature': n, 'importance': float(v)} for n, v in zip(out_names, importances)
	], key=lambda x: x['importance'], reverse=True)

	# Grouped by original column
	grouped: dict[str, float] = {}
	for n, v in zip(out_names, importances):
		if n.startswith('num__'):
			base = n.split('__', 1)[1]
		elif n.startswith('cat__'):
			tail = n.split('__', 1)[1]
			base = tail.split('_', 1)[0]  # e.g., Situation_<cat>
		else:
			base = n
		grouped[base] = grouped.get(base, 0.0) + float(v)
	grouped_sorted = sorted(
		[{'feature': k, 'importance': v} for k, v in grouped.items()],
		key=lambda x: x['importance'], reverse=True
	)

	metrics['feature_importances_expanded'] = expanded
	metrics['feature_importances_grouped'] = grouped_sorted

	return pipe, metrics


def main():
	parser = argparse.ArgumentParser(description='Train XGBoost game projection model.')
	parser.add_argument('--sql', default='SELECT * FROM Game_Model_Preseason_Away', help='SQL query to load training data')
	parser.add_argument('--target', default='Win', help='Target column name')
	parser.add_argument('--test-size', type=float, default=0.2, help='Holdout test size')
	parser.add_argument('--seed', type=int, default=42, help='Random seed')
	parser.add_argument('--model-out', default=None, help='Output path for joblib model')
	args = parser.parse_args()

	df = load_dataframe(args.sql)

	# Feature set: "Situation" (categorical) + all remaining numerics specified here
	feature_cols = [
		"Situation", "Age_F", "Age_D", "pEV_TOI_F", "pPP_TOI_F", "pSH_TOI_F",
		"pEV_TOI_D", "pPP_TOI_D", "pSH_TOI_D", "pGF", "pGA", "pxGF", "pxGA",
		"pPP_GF", "pPP_xGF", "pSH_GA", "pSH_xGA", "pPEN", "pEV_iG", "pEV_A1",
		"pEV_A2", "pPP_iG", "pPP_A1", "pPP_A2", "pSH_iG", "pSH_A1", "pSH_A2",
		"pEV_QoT", "pEV_QoC", "pEV_ZS", "pPP_QoT", "pPP_QoC", "pPP_ZS",
		"pSH_QoT", "pSH_QoC", "pSH_ZS", "pDist_Total", "pSpeed_Bursts", "pZone_Time"
	]

	missing = [c for c in feature_cols + [args.target] if c not in df.columns]
	if missing:
		raise SystemExit(f"Missing columns in data: {missing}")

	model, metrics = build_and_train(
		df, feature_cols=feature_cols, target_col=args.target,
		test_size=args.test_size, random_state=args.seed
	)

	# Save model pipeline
	ts = datetime.now().strftime('%Y%m%d_%H%M%S')
	default_name = f"xgb_game_projection_{ts}.joblib"
	out_path = args.model_out or os.path.join(MODEL_DIR, default_name)
	joblib.dump(dict(pipeline=model, metrics=metrics), out_path)
	print(f"Saved model to: {out_path}")
	print(
		"Test:  AUC={:.4f}  LogLoss={:.4f}\nTrain: AUC={:.4f}  LogLoss={:.4f}".format(
			metrics['test_auc'], metrics['test_logloss'], metrics['train_auc'], metrics['train_logloss']
		)
	)
	# Show top 25 grouped importances
	top_grouped = metrics['feature_importances_grouped'][:25]
	print("Top grouped importances:")
	for item in top_grouped:
		print(f"  {item['feature']:<20} {item['importance']:.6f}")


if __name__ == '__main__':
	main()
