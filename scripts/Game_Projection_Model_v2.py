"""
Game Projection Model v2 — MySQL-backed
Fits models one at a time from model_data table.
"""

import os, argparse, sys, math
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, log_loss, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

load_dotenv()

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(REPO_ROOT, 'Model')
os.makedirs(MODEL_DIR, exist_ok=True)

N_STATES = 10  # random states for coefficient averaging

# League-average goals per team per game (used to un-center gf predictions)
LG_AVG = {
    '20192020': 2.9505,
    '20202021': 2.8834,
    '20212022': 3.1096,
    '20222023': 3.1422,
    '20232024': 3.0679,
    '20242025': 3.0168,
    '20252026': 3.0724,
}


def _get_engine(db_url: str = ''):
    url = db_url or os.getenv('DATABASE_URL', '')
    if not url:
        raise RuntimeError('No DB URL. Set DATABASE_URL or use --db-url.')
    return create_engine(url, connect_args={'connect_timeout': 60})


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_model_data(eng) -> pd.DataFrame:
    """Load home games from model_data."""
    q = text("""
        SELECT *
        FROM model_data
        WHERE venue = 'home'
    """)
    print('  Loading model_data (home games) …', end=' ', flush=True)
    df = pd.read_sql(q, eng)
    # Rename awkward rookie column names
    df.rename(columns={
        'p.rookie_f-o.rookie_f': 'rookie_f',
        'p.rookie_d-o.rookie_d': 'rookie_d',
        'p.rookie_g-o.rookie_g': 'rookie_g',
    }, inplace=True)
    print(f'{len(df):,} rows')
    return df


# ═══════════════════════════════════════════════════════════════════════
# MODEL 1: WIN PROBABILITY (LOGISTIC REGRESSION)
# ═══════════════════════════════════════════════════════════════════════

def fit_logistic(df: pd.DataFrame):
    """
    Logistic regression: win ~ situation + poss_value + off_the_puck
                         + xga_5v5_4v4 + gax + gsax + rookies
    No intercept.  Coefficients averaged over {N_STATES} random splits.
    """
    print('\n' + '=' * 60)
    print(f'Logistic Regression — Win Probability (avg over {N_STATES} splits)')
    print('=' * 60)

    situ_dummies = pd.get_dummies(df['situation'], prefix='sit',
                                  drop_first=False).astype(float)
    numeric_cols = ['poss_value', 'xga_5v5_4v4',
                    'gax', 'gsax', 'rookie_f', 'rookie_d', 'rookie_g']
    X = pd.concat([situ_dummies, df[numeric_cols].astype(float)], axis=1)
    y = df['win'].astype(int)

    blobs = []
    tr_auc, te_auc, tr_ll, te_ll = [], [], [], []
    for rs in range(N_STATES):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=rs)
        m = LogisticRegression(fit_intercept=False, max_iter=5000).fit(Xtr, ytr)
        blobs.append({'model': m, 'features': list(X.columns)})
        tr_auc.append(roc_auc_score(ytr, m.predict_proba(Xtr)[:, 1]))
        te_auc.append(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
        tr_ll.append(log_loss(ytr, m.predict_proba(Xtr)[:, 1]))
        te_ll.append(log_loss(yte, m.predict_proba(Xte)[:, 1]))

    result = _average_logistic(blobs)

    print(f'  Features ({len(X.columns)}): {list(X.columns)}')
    print(f'\n  Train — AUC={np.mean(tr_auc):.4f}±{np.std(tr_auc):.4f}  '
          f'LogLoss={np.mean(tr_ll):.4f}±{np.std(tr_ll):.4f}')
    print(f'  Test  — AUC={np.mean(te_auc):.4f}±{np.std(te_auc):.4f}  '
          f'LogLoss={np.mean(te_ll):.4f}±{np.std(te_ll):.4f}')
    _print_coefs(result['model'], result['features'])

    return result


# ═══════════════════════════════════════════════════════════════════════
# MODEL 2: xG DIFFERENTIAL (LINEAR REGRESSION)
# ═══════════════════════════════════════════════════════════════════════

def fit_xg_diff(df: pd.DataFrame):
    """
    Linear regression: xg_diff ~ situation + poss_value + off_the_puck
                         + xga_5v5_4v4 + rookies
    Also runs a strength-state variant for comparison.
    No intercept.  Random 80/20 train/test split.
    """
    situ_dummies = pd.get_dummies(df['situation'], prefix='sit',
                                  drop_first=False).astype(float)

    # ── Variant A: aggregate poss_value / off_the_puck ──
    print('\n' + '=' * 60)
    print(f'Linear Regression — xG Differential  [A] aggregate (avg over {N_STATES} splits)')
    print('=' * 60)

    numeric_cols_a = ['poss_value', 'xga_5v5_4v4',
                      'rookie_f', 'rookie_d', 'rookie_g']
    X_a = pd.concat([situ_dummies, df[numeric_cols_a].astype(float)], axis=1)
    y = df['xg_diff'].astype(float)

    blobs_a = []
    tr_r2, te_r2, tr_rmse, te_rmse = [], [], [], []
    for rs in range(N_STATES):
        Xtr, Xte, ytr, yte = train_test_split(X_a, y, test_size=0.2, random_state=rs)
        m = LinearRegression(fit_intercept=False).fit(Xtr, ytr)
        blobs_a.append({'model': m, 'features': list(X_a.columns)})
        tr_r2.append(r2_score(ytr, m.predict(Xtr)))
        te_r2.append(r2_score(yte, m.predict(Xte)))
        tr_rmse.append(np.sqrt(mean_squared_error(ytr, m.predict(Xtr))))
        te_rmse.append(np.sqrt(mean_squared_error(yte, m.predict(Xte))))

    result_a = _average_linear(blobs_a)

    print(f'  Features ({len(X_a.columns)}): {list(X_a.columns)}')
    print(f'\n  Train — R²={np.mean(tr_r2):.4f}±{np.std(tr_r2):.4f}  '
          f'RMSE={np.mean(tr_rmse):.4f}±{np.std(tr_rmse):.4f}')
    print(f'  Test  — R²={np.mean(te_r2):.4f}±{np.std(te_r2):.4f}  '
          f'RMSE={np.mean(te_rmse):.4f}±{np.std(te_rmse):.4f}')
    _print_coefs(result_a['model'], result_a['features'])

    # ── Variant B: strength-state poss_value (single-split comparison) ──
    print('\n' + '=' * 60)
    print('Linear Regression — xG Differential  [B] strength-state (comparison)')
    print('=' * 60)

    numeric_cols_b = [
        'poss_value_5v5_4v4', 'poss_value_5v4', 'poss_value_4v5_3v5_3v4',
        'poss_value_3v3', 'poss_value_5v3_4v3',
        'xga_5v5_4v4',
        'rookie_f', 'rookie_d', 'rookie_g',
    ]
    X_b = pd.concat([situ_dummies, df[numeric_cols_b].astype(float)], axis=1)
    X_tr_b, X_te_b, y_tr_b, y_te_b = train_test_split(
        X_b, y, test_size=0.2, random_state=42)

    print(f'  Train: {len(X_tr_b):,}  Test: {len(X_te_b):,}')
    print(f'  Features ({len(X_b.columns)}): {list(X_b.columns)}')

    m_b = LinearRegression(fit_intercept=False).fit(X_tr_b, y_tr_b)
    yp_tr = m_b.predict(X_tr_b); yp_te = m_b.predict(X_te_b)
    print(f'\n  Train — R²={r2_score(y_tr_b, yp_tr):.4f}  '
          f'RMSE={np.sqrt(mean_squared_error(y_tr_b, yp_tr)):.4f}')
    print(f'  Test  — R²={r2_score(y_te_b, yp_te):.4f}  '
          f'RMSE={np.sqrt(mean_squared_error(y_te_b, yp_te)):.4f}')
    _print_coefs(m_b, X_b.columns)

    return result_a


def _print_coefs(m, cols):
    print('\n  Coefficients:')
    for name, coef in zip(cols, m.coef_.flatten()):
        print(f'    {name:30s}  {coef:+.6f}')


def _average_linear(blobs):
    """Given list of {'model':m,'features':[...]}, return one with averaged coef_."""
    coefs = np.mean([b['model'].coef_ for b in blobs], axis=0)
    avg = LinearRegression(fit_intercept=False)
    avg.coef_ = coefs
    avg.intercept_ = 0.0
    return {'model': avg, 'features': blobs[0]['features']}


def _average_logistic(blobs):
    """Given list of {'model':m,'features':[...]}, return one with averaged coef_."""
    coefs = np.mean([b['model'].coef_ for b in blobs], axis=0)
    avg = LogisticRegression(fit_intercept=False, max_iter=5000)
    avg.coef_ = coefs
    avg.intercept_ = 0.0
    avg.classes_ = np.array([0, 1])
    return {'model': avg, 'features': blobs[0]['features']}


# ═══════════════════════════════════════════════════════════════════════
# MODEL 3: GOAL DIFFERENTIAL (LINEAR REGRESSION)
# ═══════════════════════════════════════════════════════════════════════

def fit_g_diff(df: pd.DataFrame, xg_diff_mdata: dict):
    """
    Linear regression: g_diff ~ situation + playdriving + gax + gsax
    playdriving = predicted xg_diff from model 2 averaged coefficients.
    No intercept.  Coefficients averaged over {N_STATES} random splits.
    """
    print('\n' + '=' * 60)
    print(f'Linear Regression — Goal Differential (cascading, avg over {N_STATES} splits)')
    print('=' * 60)

    m2 = xg_diff_mdata['model']
    m2_feats = xg_diff_mdata['features']
    situ_all = pd.get_dummies(df['situation'], prefix='sit',
                              drop_first=False).astype(float)
    X_m2 = pd.concat([situ_all, df[['poss_value', 'xga_5v5_4v4',
                                     'rookie_f', 'rookie_d', 'rookie_g']].astype(float)], axis=1)
    X_m2 = X_m2[m2_feats]
    playdriving = m2.predict(X_m2)

    X_num = pd.DataFrame({'playdriving': playdriving,
                          'gax': df['gax'].astype(float),
                          'gsax': df['gsax'].astype(float)})
    X = pd.concat([situ_all, X_num], axis=1)
    y = df['g_diff'].astype(float)

    blobs = []
    tr_r2, te_r2, tr_rmse, te_rmse = [], [], [], []
    for rs in range(N_STATES):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=rs)
        m = LinearRegression(fit_intercept=False).fit(Xtr, ytr)
        blobs.append({'model': m, 'features': list(X.columns)})
        tr_r2.append(r2_score(ytr, m.predict(Xtr)))
        te_r2.append(r2_score(yte, m.predict(Xte)))
        tr_rmse.append(np.sqrt(mean_squared_error(ytr, m.predict(Xtr))))
        te_rmse.append(np.sqrt(mean_squared_error(yte, m.predict(Xte))))

    result = _average_linear(blobs)

    print(f'  Features ({len(X.columns)}): {list(X.columns)}')
    print(f'\n  Train — R²={np.mean(tr_r2):.4f}±{np.std(tr_r2):.4f}  '
          f'RMSE={np.mean(tr_rmse):.4f}±{np.std(tr_rmse):.4f}')
    print(f'  Test  — R²={np.mean(te_r2):.4f}±{np.std(te_r2):.4f}  '
          f'RMSE={np.mean(te_rmse):.4f}±{np.std(te_rmse):.4f}')
    _print_coefs(result['model'], result['features'])

    return result


# ═══════════════════════════════════════════════════════════════════════
# MODEL 4: GOAL DIFFERENTIAL — EV / PP / SH (LINEAR REGRESSION)
# ═══════════════════════════════════════════════════════════════════════

def fit_g_diff_evppsh(df: pd.DataFrame):
    """
    Linear regression: g_diff ~ situation
       + poss_value_ev + xga_ev
       + poss_value_st + xgf_pp + xga_sh + off_the_puck_sh
       + gax + gsax
       + rookie_f + rookie_d + rookie_g
    No intercept.  Coefficients averaged over {N_STATES} random splits.
    """
    print('\n' + '=' * 60)
    print(f'Linear Regression — Goal Differential (EV/PP/SH, avg over {N_STATES} splits)')
    print('=' * 60)

    situ_dummies = pd.get_dummies(df['situation'], prefix='sit',
                                  drop_first=False).astype(float)
    numeric_cols = [
        'poss_value_ev', 'xga_ev',
        'poss_value_st', 'xgf_pp', 'xga_sh', 'off_the_puck_sh',
        'gax', 'gsax',
        'rookie_f', 'rookie_d', 'rookie_g',
    ]
    X = pd.concat([situ_dummies, df[numeric_cols].astype(float)], axis=1)
    y = df['g_diff'].astype(float)

    blobs = []
    tr_r2, te_r2, tr_rmse, te_rmse = [], [], [], []
    for rs in range(N_STATES):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=rs)
        m = LinearRegression(fit_intercept=False).fit(Xtr, ytr)
        blobs.append({'model': m, 'features': list(X.columns)})
        tr_r2.append(r2_score(ytr, m.predict(Xtr)))
        te_r2.append(r2_score(yte, m.predict(Xte)))
        tr_rmse.append(np.sqrt(mean_squared_error(ytr, m.predict(Xtr))))
        te_rmse.append(np.sqrt(mean_squared_error(yte, m.predict(Xte))))

    result = _average_linear(blobs)

    print(f'  Features ({len(X.columns)}): {list(X.columns)}')
    print(f'\n  Train — R²={np.mean(tr_r2):.4f}±{np.std(tr_r2):.4f}  '
          f'RMSE={np.mean(tr_rmse):.4f}±{np.std(tr_rmse):.4f}')
    print(f'  Test  — R²={np.mean(te_r2):.4f}±{np.std(te_r2):.4f}  '
          f'RMSE={np.mean(te_rmse):.4f}±{np.std(te_rmse):.4f}')
    _print_coefs(result['model'], result['features'])

    return result


# ═══════════════════════════════════════════════════════════════════════
# MODEL 4b: GOAL DIFFERENTIAL — SINGLE STEP (LINEAR REGRESSION)
# ═══════════════════════════════════════════════════════════════════════

def fit_g_diff_single(df: pd.DataFrame):
    """
    Linear regression: g_diff ~ situation + poss_value + off_the_puck
                         + xga_5v5_4v4 + gax + gsax + rookies
    No intercept.  Coefficients averaged over {N_STATES} random splits.
    """
    print('\n' + '=' * 60)
    print(f'Linear Regression — Goal Differential (single-step, avg over {N_STATES} splits)')
    print('=' * 60)

    situ_dummies = pd.get_dummies(df['situation'], prefix='sit',
                                  drop_first=False).astype(float)
    numeric_cols = ['poss_value', 'xga_5v5_4v4', 'gax', 'gsax',
                    'rookie_f', 'rookie_d', 'rookie_g']
    X = pd.concat([situ_dummies, df[numeric_cols].astype(float)], axis=1)
    y = df['g_diff'].astype(float)

    blobs = []
    tr_r2, te_r2, tr_rmse, te_rmse = [], [], [], []
    for rs in range(N_STATES):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=rs)
        m = LinearRegression(fit_intercept=False).fit(Xtr, ytr)
        blobs.append({'model': m, 'features': list(X.columns)})
        tr_r2.append(r2_score(ytr, m.predict(Xtr)))
        te_r2.append(r2_score(yte, m.predict(Xte)))
        tr_rmse.append(np.sqrt(mean_squared_error(ytr, m.predict(Xtr))))
        te_rmse.append(np.sqrt(mean_squared_error(yte, m.predict(Xte))))

    result = _average_linear(blobs)

    print(f'  Features ({len(X.columns)}): {list(X.columns)}')
    print(f'\n  Train — R²={np.mean(tr_r2):.4f}±{np.std(tr_r2):.4f}  '
          f'RMSE={np.mean(tr_rmse):.4f}±{np.std(tr_rmse):.4f}')
    print(f'  Test  — R²={np.mean(te_r2):.4f}±{np.std(te_r2):.4f}  '
          f'RMSE={np.mean(te_rmse):.4f}±{np.std(te_rmse):.4f}')
    _print_coefs(result['model'], result['features'])

    return result


# ═══════════════════════════════════════════════════════════════════════
# MODEL 5: xGF METRIC (LINEAR REGRESSION)
# ═══════════════════════════════════════════════════════════════════════

def fit_xgf_metric(df: pd.DataFrame):
    """
    Linear regression: xgf_metric ~ situation + poss_value_team
                         + off_the_puck_team + xga_opp + rookies
    No intercept.  Coefficients averaged over {N_STATES} random splits.
    """
    print('\n' + '=' * 60)
    print(f'Linear Regression — xGF Metric (avg over {N_STATES} splits)')
    print('=' * 60)

    situ_dummies = pd.get_dummies(df['situation'], prefix='sit',
                                  drop_first=False).astype(float)
    numeric_cols = ['poss_value_team', 'xga_opp',
                    'rookie_f', 'rookie_d', 'rookie_g']
    X = pd.concat([situ_dummies, df[numeric_cols].astype(float)], axis=1)
    y = df['xgf_metric'].astype(float)

    blobs = []
    tr_r2, te_r2, tr_rmse, te_rmse = [], [], [], []
    for rs in range(N_STATES):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=rs)
        m = LinearRegression(fit_intercept=False).fit(Xtr, ytr)
        blobs.append({'model': m, 'features': list(X.columns)})
        tr_r2.append(r2_score(ytr, m.predict(Xtr)))
        te_r2.append(r2_score(yte, m.predict(Xte)))
        tr_rmse.append(np.sqrt(mean_squared_error(ytr, m.predict(Xtr))))
        te_rmse.append(np.sqrt(mean_squared_error(yte, m.predict(Xte))))

    result = _average_linear(blobs)

    print(f'  Features ({len(X.columns)}): {list(X.columns)}')
    print(f'\n  Train — R²={np.mean(tr_r2):.4f}±{np.std(tr_r2):.4f}  '
          f'RMSE={np.mean(tr_rmse):.4f}±{np.std(tr_rmse):.4f}')
    print(f'  Test  — R²={np.mean(te_r2):.4f}±{np.std(te_r2):.4f}  '
          f'RMSE={np.mean(te_rmse):.4f}±{np.std(te_rmse):.4f}')
    _print_coefs(result['model'], result['features'])

    return result


# ═══════════════════════════════════════════════════════════════════════
# MODEL 6: GOALS FOR (LINEAR REGRESSION)
# ═══════════════════════════════════════════════════════════════════════

def fit_gf(df: pd.DataFrame):
    """
    Linear regression: gf ~ situation + poss_value_team + off_the_puck_team
                        + xga_opp + gax_team + gsax_opp + rookies
    No intercept.  Coefficients averaged over {N_STATES} random splits.
    """
    print('\n' + '=' * 60)
    print(f'Linear Regression — Goals For (avg over {N_STATES} splits)')
    print('=' * 60)

    situ_dummies = pd.get_dummies(df['situation'], prefix='sit',
                                  drop_first=False).astype(float)
    numeric_cols = ['poss_value_team', 'xga_opp',
                    'gax_team', 'gsax_opp',
                    'rookie_f', 'rookie_d', 'rookie_g']
    X = pd.concat([situ_dummies, df[numeric_cols].astype(float)], axis=1)
    y = df['gf'].astype(float)

    blobs = []
    tr_r2, te_r2, tr_rmse, te_rmse = [], [], [], []
    for rs in range(N_STATES):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=rs)
        m = LinearRegression(fit_intercept=False).fit(Xtr, ytr)
        blobs.append({'model': m, 'features': list(X.columns)})
        tr_r2.append(r2_score(ytr, m.predict(Xtr)))
        te_r2.append(r2_score(yte, m.predict(Xte)))
        tr_rmse.append(np.sqrt(mean_squared_error(ytr, m.predict(Xtr))))
        te_rmse.append(np.sqrt(mean_squared_error(yte, m.predict(Xte))))

    result = _average_linear(blobs)

    print(f'  Features ({len(X.columns)}): {list(X.columns)}')
    print(f'\n  Train — R²={np.mean(tr_r2):.4f}±{np.std(tr_r2):.4f}  '
          f'RMSE={np.mean(tr_rmse):.4f}±{np.std(tr_rmse):.4f}')
    print(f'  Test  — R²={np.mean(te_r2):.4f}±{np.std(te_r2):.4f}  '
          f'RMSE={np.mean(te_rmse):.4f}±{np.std(te_rmse):.4f}')
    _print_coefs(result['model'], result['features'])

    return result


# ═══════════════════════════════════════════════════════════════════════
# COMBINED WIN PROBABILITY (Normal CDF via g_diff + gf models)
# ═══════════════════════════════════════════════════════════════════════

def _norm_cdf(x):
    """Standard normal CDF using math.erf (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def evaluate_combined_winprob(df, xg_diff_mdata, g_diff_mdata, gf_mdata):
    """
    Compute win probability by combining g_diff and gf models.

    Approach:
      μ  = predicted g_diff (model 3 cascading)
      λ_home = gf_centered_pred + lg_avg        (actual GF home)
      λ_away = gf_centered_pred - μ + lg_avg    (actual GF away, via identity)
      σ  = sqrt(λ_home + λ_away)                (Poisson-difference variance)
      P(home win) = Φ(μ / σ)

    Uses model 2 → playdriving → model 3 → g_diff, and model 6 → gf.
    """
    print('\n' + '=' * 60)
    print('Combined Win Probability (Normal CDF)')
    print('=' * 60)

    # ── Build playdriving (model 2 features → predict) ──
    m2 = xg_diff_mdata['model']
    m2_feats = xg_diff_mdata['features']
    situ_all = pd.get_dummies(df['situation'], prefix='sit',
                              drop_first=False).astype(float)
    X_m2 = pd.concat([situ_all,
                      df[['poss_value', 'xga_5v5_4v4',
                          'rookie_f', 'rookie_d', 'rookie_g']].astype(float)],
                     axis=1)
    X_m2 = X_m2[m2_feats]
    playdriving = m2.predict(X_m2)

    # ── Build g_diff features (model 3: situation + playdriving + gax + gsax) ──
    m3 = g_diff_mdata['model']
    m3_feats = g_diff_mdata['features']
    X_m3 = pd.concat([situ_all,
                      pd.DataFrame({'playdriving': playdriving,
                                    'gax': df['gax'].astype(float),
                                    'gsax': df['gsax'].astype(float)})],
                     axis=1)
    X_m3 = X_m3[m3_feats]
    g_diff_pred = m3.predict(X_m3)

    # ── Build gf features (model 6) ──
    m6 = gf_mdata['model']
    m6_feats = gf_mdata['features']
    X_m6 = pd.concat([situ_all,
                      df[['poss_value_team', 'xga_opp',
                          'gax_team', 'gsax_opp',
                          'rookie_f', 'rookie_d', 'rookie_g']].astype(float)],
                     axis=1)
    X_m6 = X_m6[m6_feats]
    gf_centered_pred = m6.predict(X_m6)

    # ── Compute win probabilities ──
    probs = []
    for i in range(len(df)):
        season = df['season'].iloc[i]
        lg = LG_AVG.get(str(season), 3.0)

        lam_home = max(0.5, gf_centered_pred[i] + lg)
        lam_away = max(0.5, gf_centered_pred[i] - g_diff_pred[i] + lg)
        sigma = math.sqrt(lam_home + lam_away)
        mu = g_diff_pred[i]

        probs.append(_norm_cdf(mu / sigma))

    probs = np.array(probs)
    y_true = df['win'].astype(int).values

    # ── Evaluate ──
    auc = roc_auc_score(y_true, probs)
    ll = log_loss(y_true, np.clip(probs, 1e-15, 1 - 1e-15))
    acc = (y_true == (probs >= 0.5).astype(int)).mean()

    print(f'  AUC={auc:.4f}  LogLoss={ll:.4f}  Acc={acc:.3f}')

    # Distribution stats
    sigmas = np.array([math.sqrt(max(0.5, gf_centered_pred[j] + LG_AVG.get(str(s), 3.0))
                                 + max(0.5, gf_centered_pred[j] - g_diff_pred[j]
                                       + LG_AVG.get(str(s), 3.0)))
                       for j, s in enumerate(df['season'])])
    print(f'\n  g_diff_pred:  mean={g_diff_pred.mean():.3f}  '
          f'std={g_diff_pred.std():.3f}')
    print(f'  gf_centered_pred: mean={gf_centered_pred.mean():.3f}  '
          f'std={gf_centered_pred.std():.3f}')
    print(f'  σ (sqrt total goals): mean={sigmas.mean():.3f}  '
          f'std={sigmas.std():.3f}')
    print(f'  P(home win):  mean={probs.mean():.3f}  std={probs.std():.3f}')

    return {'probs': probs, 'auc': auc, 'logloss': ll}


def evaluate_combined_winprob_evppsh(df, g_diff_evppsh_mdata, gf_mdata):
    """
    Compute win probability by combining g_diff_evppsh and gf models.

    Approach:
      μ  = predicted g_diff (EV/PP/SH model)
      λ_home = gf_centered_pred + lg_avg        (actual GF home)
      λ_away = gf_centered_pred - μ + lg_avg    (actual GF away, via identity)
      σ  = sqrt(λ_home + λ_away)                (Poisson-difference variance)
      P(home win) = Φ(μ / σ)

    Uses EV/PP/SH g_diff model + gf model.
    """
    print('\n' + '=' * 60)
    print('Combined Win Probability — EV/PP/SH + GF (Normal CDF)')
    print('=' * 60)

    situ_all = pd.get_dummies(df['situation'], prefix='sit',
                              drop_first=False).astype(float)

    # ── Build g_diff_evppsh features ──
    m_gd = g_diff_evppsh_mdata['model']
    gd_feats = g_diff_evppsh_mdata['features']
    X_gd = pd.concat([situ_all,
                      df[['poss_value_ev', 'xga_ev',
                          'poss_value_st', 'xgf_pp', 'xga_sh',
                          'off_the_puck_sh',
                          'gax', 'gsax',
                          'rookie_f', 'rookie_d', 'rookie_g']].astype(float)],
                     axis=1)
    X_gd = X_gd[gd_feats]
    g_diff_pred = m_gd.predict(X_gd)

    # ── Build gf features (same as model 6) ──
    m_gf = gf_mdata['model']
    gf_feats = gf_mdata['features']
    X_gf = pd.concat([situ_all,
                      df[['poss_value_team', 'xga_opp',
                          'gax_team', 'gsax_opp',
                          'rookie_f', 'rookie_d', 'rookie_g']].astype(float)],
                     axis=1)
    X_gf = X_gf[gf_feats]
    gf_centered_pred = m_gf.predict(X_gf)

    # ── Compute win probabilities ──
    probs = []
    for i in range(len(df)):
        season = df['season'].iloc[i]
        lg = LG_AVG.get(str(season), 3.0)

        lam_home = max(0.5, gf_centered_pred[i] + lg)
        lam_away = max(0.5, gf_centered_pred[i] - g_diff_pred[i] + lg)
        sigma = math.sqrt(lam_home + lam_away)
        mu = g_diff_pred[i]

        probs.append(_norm_cdf(mu / sigma))

    probs = np.array(probs)
    y_true = df['win'].astype(int).values

    # ── Evaluate ──
    auc = roc_auc_score(y_true, probs)
    ll = log_loss(y_true, np.clip(probs, 1e-15, 1 - 1e-15))
    acc = (y_true == (probs >= 0.5).astype(int)).mean()

    print(f'  AUC={auc:.4f}  LogLoss={ll:.4f}  Acc={acc:.3f}')

    # Distribution stats
    sigmas = np.array([math.sqrt(max(0.5, gf_centered_pred[j] + LG_AVG.get(str(s), 3.0))
                                 + max(0.5, gf_centered_pred[j] - g_diff_pred[j]
                                       + LG_AVG.get(str(s), 3.0)))
                       for j, s in enumerate(df['season'])])
    print(f'\n  g_diff_pred:  mean={g_diff_pred.mean():.3f}  '
          f'std={g_diff_pred.std():.3f}')
    print(f'  gf_centered_pred: mean={gf_centered_pred.mean():.3f}  '
          f'std={gf_centered_pred.std():.3f}')
    print(f'  σ (sqrt total goals): mean={sigmas.mean():.3f}  '
          f'std={sigmas.std():.3f}')
    print(f'  P(home win):  mean={probs.mean():.3f}  std={probs.std():.3f}')

    # ── Calibration table ──
    buckets = [
        ('>75%',      0.75, 1.01),
        ('70-75%',    0.70, 0.75),
        ('65-70%',    0.65, 0.70),
        ('60-65%',    0.60, 0.65),
        ('55-60%',    0.55, 0.60),
        ('50-55%',    0.50, 0.55),
        ('45-50%',    0.45, 0.50),
        ('40-45%',    0.40, 0.45),
        ('35-40%',    0.35, 0.40),
        ('30-35%',    0.30, 0.35),
        ('<30%',     -0.01, 0.30),
    ]
    print(f'\n  {"Bucket":>10s}  {"Games":>6s}  {"Wins":>6s}  {"Win%":>7s}')
    print(f'  {"─"*10}  {"─"*6}  {"─"*6}  {"─"*7}')
    for label, lo, hi in buckets:
        mask = (probs >= lo) & (probs < hi)
        n = mask.sum()
        w = y_true[mask].sum() if n > 0 else 0
        pct = w / n * 100 if n > 0 else 0
        print(f'  {label:>10s}  {n:>6d}  {w:>6d}  {pct:>6.1f}%')

    return {'probs': probs, 'auc': auc, 'logloss': ll}


# ═══════════════════════════════════════════════════════════════════════
# SAVE COEFFICIENTS TO MySQL
# ═══════════════════════════════════════════════════════════════════════

def save_coefficients_to_mysql(eng, model_name: str, mdata: dict):
    """Upsert model coefficients into model_coefficients table."""
    features = mdata['features']
    coefs = mdata['model'].coef_.flatten()
    rows = [{'model_name': model_name, 'feature': f, 'coefficient': float(c)}
            for f, c in zip(features, coefs)]
    df = pd.DataFrame(rows)

    with eng.connect() as conn:
        # Ensure table exists
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS model_coefficients (
                model_name VARCHAR(50),
                feature VARCHAR(100),
                coefficient DOUBLE,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                PRIMARY KEY (model_name, feature)
            )
        """))
        conn.commit()

        # Delete existing rows for this model, then insert
        conn.execute(text("DELETE FROM model_coefficients WHERE model_name = :m"),
                     {'m': model_name})
        conn.commit()

    df.to_sql('model_coefficients', eng, if_exists='append', index=False)
    print(f'  [MySQL] saved {len(df)} coefficients for {model_name}')


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Game Projection Model v2 (MySQL)')
    parser.add_argument('--save', action='store_true',
                        help='Save models to Model/')
    parser.add_argument('--db-url', default='', help='Explicit DB URL')
    args = parser.parse_args()

    eng = _get_engine(args.db_url)
    print('Loading data …')

    df = load_model_data(eng)

    win_prob = fit_logistic(df)
    xg_diff = fit_xg_diff(df)
    g_diff = fit_g_diff(df, xg_diff)
    g_diff_evppsh = fit_g_diff_evppsh(df)
    g_diff_single = fit_g_diff_single(df)
    xgf_metric = fit_xgf_metric(df)
    gf = fit_gf(df)
    evaluate_combined_winprob(df, xg_diff, g_diff, gf)
    evaluate_combined_winprob_evppsh(df, g_diff_evppsh, gf)

    # Save production model coefficients to MySQL
    save_coefficients_to_mysql(eng, 'g_diff_evppsh', g_diff_evppsh)
    save_coefficients_to_mysql(eng, 'g_diff_single', g_diff_single)
    save_coefficients_to_mysql(eng, 'gf', gf)

    if args.save:
        print('\nSaving models …')
        for name, mdata in [('win_prob', win_prob), ('xg_diff', xg_diff),
                            ('g_diff', g_diff), ('g_diff_single', g_diff_single),
                            ('xgf_metric', xgf_metric), ('gf', gf)]:
            path = os.path.join(MODEL_DIR, f'game_projection_v2_{name}.pkl')
            joblib.dump(mdata, path)
            print(f'  {path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
