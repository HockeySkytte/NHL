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

# Create the # Create the engine
engine = create_engine(f'mysql+mysqlconnector://{username}:{password}@{host}:{port}/{database}')

query = text('SELECT * FROM game_model_preseason_away')

# Execute the query and load into a DataFrame
df = pd.read_sql(query, con=engine)


import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import joblib

# Preseason calculation using provided coefficients (excluding Situation dummies)
preseason_coeffs = {
	"Rookie_G": -0.12,
    "pxGA": -0.2631652307425146,
    "pxGF": 0.2236978723207229,
    "pGA": -0.10641066618060169,
    "pGF": 0.1459154931787248,
    #"Rookie_D": -0.046580253137838985,
    #"Rookie_F": -0.0408702240948283,
    "Age_F": -0.03309704025575828,
    "Age_G": -0.02311629852461659,
    "pPP_GF": 0.025038290070174687,
    "pPEN": -0.0006673090871371258,
    "pGSAx_S": 0.0026361609743112096,
    "piP": 0.0013821749901097836,
}

# Compute the Preseason score as the weighted sum of the raw variables
missing_cols = [c for c in preseason_coeffs if c not in df.columns]
if missing_cols:
	print(f"\nWarning: Missing columns for Preseason calculation: {missing_cols}")
else:
	df["Preseason"] = sum(df[col] * w for col, w in preseason_coeffs.items())
	print("\nPreseason summary (from provided coefficients):")
	print(df["Preseason"].describe())

player_coeffs = {
	"Rookie_G": -0.12,
    "pxGA": -0.21980,
    "pxGF": 0.18687,
    "pGA": -0.08900,
    "pGF": 0.12180,
    "Rookie_D": -0.03890,
    "Rookie_F": -0.03413,
    "Age_F": -0.02764,
    "Age_G": -0.01932,
    "pPP_GF": 0.02091,
    "pPEN": -0.00056,
    "pGSAx_S": 0.00220,
    "piP": 0.00115,
	"iiP": 0.00091,
    "iEV_xGF": 0.00333,
    "iEV_xGA": -0.00298,
    "iPP_GF": 0.00090,
	"iSH_GA": -0.00058,
	"iGSAx_S": 0.00557
}

# Compute the Player score as the weighted sum of the raw variables
missing_cols = [c for c in player_coeffs if c not in df.columns]
if missing_cols:
	print(f"\nWarning: Missing columns for Player calculation: {missing_cols}")
else:
	df["Player"] = sum(df[col] * w for col, w in player_coeffs.items())
	print("\nPlayer summary (from provided coefficients):")
	print(df["Player"].describe())

y = df["Win"]
X1 = df[["Situation"]]
#X2 = df[["Rookie_G", "pxGA", "pxGF", "pGA", "pGF", "Rookie_D", "Rookie_F", "Age_F", "Age_G", "pPP_GF", "pPEN", "pGSAx_S", "piP"]]
X2 = df[[
         #"Player",
		 #"Rookie_G", "pxGA", "pxGF", "pGA", "pGF", "Rookie_D", "Rookie_F", "Age_F", "Age_G", "pPP_GF", "pPEN", "pGSAx_S", "piP",
		 "Preseason", "Rookie_D", "Rookie_F",
		 "iiP",
		 #"iiG", "iA1", "iA2", 
		 #"iEV_GF", "iEV_GA", 
		 "iEV_xGF", "iEV_xGA", "iPP_GF", 
		 #"iPP_xGF", 
		 #"iSH_GA",
		 "iSH_xGA", 
		 "iGSAx_S"
		 ]]

X1 = pd.get_dummies(X1)
X1 = X1.astype(int)

X = X1.join(X2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)

log_reg = LogisticRegression(random_state = 0, max_iter=5000, fit_intercept=False).fit(X_train, y_train)

coef = pd.DataFrame(zip(X_train.columns, np.transpose(log_reg.coef_)), columns=['features', 'coef'])
print(coef)

# Evaluate logloss (and AUC) on train and test
y_proba_test = log_reg.predict_proba(X_test)[:, 1]
y_proba_train = log_reg.predict_proba(X_train)[:, 1]
print("\nMetrics:")
print(f"Test  LogLoss: {log_loss(y_test, y_proba_test):.4f}  AUC: {roc_auc_score(y_test, y_proba_test):.4f}")
print(f"Train LogLoss: {log_loss(y_train, y_proba_train):.4f}  AUC: {roc_auc_score(y_train, y_proba_train):.4f}")
