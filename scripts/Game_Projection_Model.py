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
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
		# Defaults for Logistic Regression to encourage generalization
		model_params = dict(
			penalty='l2',
			C=1.0,
			solver='liblinear',  # robust for smaller datasets
			max_iter=5000,
			random_state=random_state,
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
	# sklearn >=1.2 switched from `sparse` to `sparse_output`; support both
	def make_ohe():
		try:
			return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
		except TypeError:
			return OneHotEncoder(handle_unknown='ignore', sparse=False)
	numeric_tf = Pipeline(steps=[
		('imputer', SimpleImputer(strategy='median')),
		('scaler', StandardScaler()),
	])
	categorical_tf = Pipeline(steps=[
		('imputer', SimpleImputer(strategy='most_frequent')),
		('onehot', make_ohe()),
	])

	pre = ColumnTransformer(
		transformers=[
			('num', numeric_tf, num_cols),
			('cat', categorical_tf, cat_cols),
		]
	)

	# Switch to Logistic Regression model as requested
	model = LogisticRegression(**model_params)

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
		# We'll infer size from coefficients or set empty list
		out_names = None

	importances: list[float]
	if hasattr(model, 'feature_importances_'):
		importances = model.feature_importances_.tolist()
		if out_names is None:
			out_names = [f'f{i}' for i in range(len(importances))]
	elif hasattr(model, 'coef_'):
		# Use absolute value of coefficients as importance proxy for LR
		coef = model.coef_[0]
		importances = [float(abs(c)) for c in coef]
		if out_names is None:
			out_names = [f'f{i}' for i in range(len(importances))]
	else:
		importances = []
		out_names = []

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

	# Coefficients (for Logistic Regression)
	coefficients_expanded = []
	coefficients_grouped: dict[str, list[dict]] = {}
	intercept = None
	if hasattr(model, 'coef_') and model.coef_.ndim == 2:
		coef = model.coef_[0]
		intercept = float(model.intercept_[0]) if hasattr(model, 'intercept_') else None
		if out_names is None:
			out_names = [f'f{i}' for i in range(len(coef))]
		for name, c in zip(out_names, coef):
			# Prettify names back to original columns
			if name.startswith('num__'):
				base = name.split('__', 1)[1]
				pretty = base
			elif name.startswith('cat__'):
				tail = name.split('__', 1)[1]  # e.g., Situation_value
				if '_' in tail:
					base, level = tail.split('_', 1)
					pretty = f"{base}[{level}]"
				else:
					base, pretty = tail, tail
			else:
				base = name
				pretty = name
			coefficients_expanded.append({'feature': pretty, 'coef': float(c)})
			coefficients_grouped.setdefault(base, []).append({'feature': pretty, 'coef': float(c)})

	# Sort expanded coefficients by absolute value descending
	coefficients_expanded = sorted(coefficients_expanded, key=lambda x: abs(x['coef']), reverse=True)
	metrics['coefficients_expanded'] = coefficients_expanded
	metrics['coefficients_grouped'] = coefficients_grouped
	if intercept is not None:
		metrics['intercept'] = intercept

	return pipe, metrics


def main():
	parser = argparse.ArgumentParser(description='Train Logistic Regression game projection model.')
	parser.add_argument('--sql', default='SELECT * FROM Game_Model_Preseason_Away', help='SQL query to load training data')
	parser.add_argument('--target', default='Win', help='Target column name')
	parser.add_argument('--test-size', type=float, default=0.2, help='Holdout test size')
	parser.add_argument('--seed', type=int, default=42, help='Random seed')
	parser.add_argument('--model-out', default=None, help='Output path for joblib model')
	args = parser.parse_args()

	df = load_dataframe(args.sql)

	# Build derived features requested: pTOI_F, pTOI_D, piG, pA1, pA2
	required_for_sums = {
		'pTOI_F': ["pEV_TOI_F", "pPP_TOI_F", "pSH_TOI_F"],
		'pTOI_D': ["pEV_TOI_D", "pPP_TOI_D", "pSH_TOI_D"],
		#'piG':    ["pEV_iG", "pPP_iG", "pSH_iG"],
		#'pA1':    ["pEV_A1", "pPP_A1", "pSH_A1"],
		#'pA2':    ["pEV_A2", "pPP_A2", "pSH_A2"],
		'piP':    ["pEV_iG", "pPP_iG", "pSH_iG", "pEV_A1", "pPP_A1", "pSH_A1", "pEV_A2", "pPP_A2", "pSH_A2"],
	}
	# Validate presence of columns needed to compute the derived features
	missing_sum_inputs = [c for cols in required_for_sums.values() for c in cols if c not in df.columns]
	if missing_sum_inputs:
		raise SystemExit(f"Missing columns required to compute derived features: {sorted(set(missing_sum_inputs))}")

	# Compute sums with NaNs treated as 0
	for out_col, cols in required_for_sums.items():
		df[out_col] = df[cols].fillna(0).sum(axis=1)

	# Requested full feature set with derived columns
	feature_cols = [
		"Situation", 
		"Age_F", "Age_D", 
		#"pTOI_F", "pTOI_D",
		#"pGF", "pGA", 
		"pxGF", "pxGA", "pPP_GF", 
		#"pPP_xGF", "pSH_GA", 
		"pSH_GA", "pPEN",
		#"piG", "pA1", "pA2", 
        #"pEV_QoT",
		"piP"
		#,"pEV_ZS"
	]

	missing = [c for c in feature_cols + [args.target] if c not in df.columns]
	if missing:
		raise SystemExit(f"Missing columns in data: {missing}")

	model, metrics = build_and_train(
		df, feature_cols=feature_cols, target_col=args.target,
		test_size=args.test_size, random_state=args.seed, model_params=None
	)

	# Save model pipeline
	ts = datetime.now().strftime('%Y%m%d_%H%M%S')
	default_name = f"logreg_game_projection_{ts}.joblib"
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

	# Print coefficients
	print("\nCoefficients (sorted by |coef|):")
	if 'intercept' in metrics:
		print(f"  Intercept: {metrics['intercept']:.6f}")
	for item in metrics.get('coefficients_expanded', [])[:100]:  # limit to first 100 for readability
		print(f"  {item['feature']:<30} {item['coef']:.6f}")


if __name__ == '__main__':
	main()
