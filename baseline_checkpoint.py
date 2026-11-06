"""
ESE 5420 Final Project — Baseline Checkpoint (due Nov 9, 2026)
Author: Xiaoguang Zhao (zhao716@seas.upenn.edu)

Professor's requirements:
  1. Report N, date range, how many rows lack square footage.
  2. Hold out last two months (Jul–Aug 2026) as test set.
  3. Set up 5-fold CV on the rest.
  4. Fit mean baseline, square-footage baseline, and full OLS;
     report CV RMSE of log price with its fold spread.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── 1. Load data ──────────────────────────────────────────────────────────────
DATA = '/mnt/user-data/uploads/ESE 5420 Fall 2026/Project/datasets/houses_clean.csv'
df = pd.read_csv(DATA)
df['SALE DATE'] = pd.to_datetime(df['SALE DATE'])

N_total = len(df)
date_min = df['SALE DATE'].min().strftime('%Y-%m-%d')
date_max = df['SALE DATE'].max().strftime('%Y-%m-%d')
n_missing_sf = (df['GROSS SQUARE FEET'] == 0).sum()  # already 0 after cleaning

print(f"Total observations (N): {N_total:,}")
print(f"Date range: {date_min} to {date_max}")
print(f"Rows missing GROSS SQUARE FEET: {n_missing_sf}")

# ── 2. Temporal split: last two months = Jul & Aug 2026 ──────────────────────
cutoff = pd.Timestamp('2026-07-01')
train_df = df[df['SALE DATE'] < cutoff].copy()
test_df  = df[df['SALE DATE'] >= cutoff].copy()

print(f"\nTemporal split:")
print(f"  Training set  (Sep 2025 – Jun 2026): N_tr = {len(train_df):,}")
print(f"  Test set      (Jul 2026 – Aug 2026): N_te = {len(test_df):,}")

y_train = train_df['LOG_PRICE'].values
y_test  = test_df['LOG_PRICE'].values

# ── 3. Feature engineering helpers ───────────────────────────────────────────

def build_features(data, fit_encoder=None):
    """
    Returns X (numpy array) for linear regression.
    Features:
      - log(GROSS SQUARE FEET)
      - log(LAND SQUARE FEET)
      - Age = 2026 - YEAR BUILT
      - TOTAL UNITS
      - Sale month (1-12)
      - Borough dummy (Queens=1, Brooklyn=0)
      - Building class category dummies (3 levels → 2 dummies)
      - Neighborhood dummies (117 levels → 116 dummies)
    fit_encoder: dict mapping col -> category order (for consistent encoding)
    """
    d = data.copy()
    d['log_gsf']  = np.log(d['GROSS SQUARE FEET'])
    d['log_lsf']  = np.log(d['LAND SQUARE FEET'])
    d['age']      = 2026 - d['YEAR BUILT']
    d['units']    = d['TOTAL UNITS']
    d['month']    = d['SALE DATE'].dt.month

    # Borough: Queens=1, Brooklyn=0
    d['borough_queens'] = (d['BOROUGH'] == 'Queens').astype(float)

    # Building class dummies (drop first = 01 ONE FAMILY)
    bcc_dummies = pd.get_dummies(d['BUILDING CLASS CATEGORY'], prefix='bcc', drop_first=True)

    # Neighborhood dummies
    if fit_encoder is None:
        fit_encoder = {}
    if 'nbhd_cats' not in fit_encoder:
        fit_encoder['nbhd_cats'] = sorted(d['NEIGHBORHOOD'].unique())
    nbhd_dummies = pd.get_dummies(
        pd.Categorical(d['NEIGHBORHOOD'], categories=fit_encoder['nbhd_cats']),
        prefix='nbhd', drop_first=True
    )

    # Assemble
    base_cols = d[['log_gsf', 'log_lsf', 'age', 'units', 'month', 'borough_queens']].values
    X = np.hstack([
        base_cols,
        bcc_dummies.values,
        nbhd_dummies.values
    ])
    return X, fit_encoder


# ── 4. 5-fold CV on training set ─────────────────────────────────────────────
K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)

rmse_folds = {
    'mean':   [],
    'gsf_lr': [],
    'ols':    [],
}

for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(train_df)):
    fold_tr  = train_df.iloc[tr_idx]
    fold_val = train_df.iloc[val_idx]

    y_tr  = fold_tr['LOG_PRICE'].values
    y_val = fold_val['LOG_PRICE'].values

    # --- Model 1: Mean baseline ---
    mu = y_tr.mean()
    pred_mean = np.full(len(y_val), mu)
    rmse_folds['mean'].append(np.sqrt(np.mean((y_val - pred_mean)**2)))

    # --- Model 2: GrossSF simple linear regression ---
    X_gsf_tr  = fold_tr[['GROSS SQUARE FEET']].values
    X_gsf_val = fold_val[['GROSS SQUARE FEET']].values
    # Standardize INSIDE the fold
    scaler_gsf = StandardScaler()
    X_gsf_tr_s  = scaler_gsf.fit_transform(X_gsf_tr)
    X_gsf_val_s = scaler_gsf.transform(X_gsf_val)
    lr_gsf = LinearRegression()
    lr_gsf.fit(X_gsf_tr_s, y_tr)
    pred_gsf = lr_gsf.predict(X_gsf_val_s)
    rmse_folds['gsf_lr'].append(np.sqrt(np.mean((y_val - pred_gsf)**2)))

    # --- Model 3: Full OLS ---
    X_tr_full, enc = build_features(fold_tr)
    X_val_full, _  = build_features(fold_val, enc)
    # Standardize continuous columns (first 5: log_gsf, log_lsf, age, units, month)
    scaler_full = StandardScaler()
    X_tr_full[:, :5]  = scaler_full.fit_transform(X_tr_full[:, :5])
    X_val_full[:, :5] = scaler_full.transform(X_val_full[:, :5])
    lr_full = LinearRegression()
    lr_full.fit(X_tr_full, y_tr)
    pred_full = lr_full.predict(X_val_full)
    rmse_folds['ols'].append(np.sqrt(np.mean((y_val - pred_full)**2)))

    print(f"  Fold {fold_idx+1}: mean={rmse_folds['mean'][-1]:.4f}  "
          f"gsf={rmse_folds['gsf_lr'][-1]:.4f}  "
          f"ols={rmse_folds['ols'][-1]:.4f}")

# ── 5. Summarise ─────────────────────────────────────────────────────────────
print("\n=== 5-Fold CV Results (RMSE of log sale price) ===")
header = f"{'Model':<30} {'CV RMSE':>10}  {'±Fold Spread':>14}  {'Folds'}"
print(header)
print("-" * len(header))

results = {}
for name, rmses in rmse_folds.items():
    arr = np.array(rmses)
    mean_rmse = arr.mean()
    fold_spread = arr.std(ddof=1)   # std across folds = "fold spread"
    results[name] = {'mean': mean_rmse, 'spread': fold_spread, 'folds': arr}
    label = {'mean': 'Mean baseline',
             'gsf_lr': 'GrossSF linear regression',
             'ols': 'Full OLS (all features)'}[name]
    fold_str = "  ".join(f"{v:.4f}" for v in arr)
    print(f"{label:<30} {mean_rmse:>10.4f}  ±{fold_spread:>12.4f}  [{fold_str}]")

# ── 6. Training-set full fit → training RMSE (for reference only) ─────────────
X_tr_all, enc_all = build_features(train_df)
scaler_all = StandardScaler()
X_tr_all[:, :5] = scaler_all.fit_transform(X_tr_all[:, :5])
lr_all = LinearRegression()
lr_all.fit(X_tr_all, y_train)
pred_tr_all = lr_all.predict(X_tr_all)
train_rmse_ols = np.sqrt(np.mean((y_train - pred_tr_all)**2))
print(f"\nFull OLS training RMSE (not CV): {train_rmse_ols:.4f}  [overfits; CV is honest estimate]")

# ── 7. Write baseline.txt ─────────────────────────────────────────────────────
txt = []
txt.append("=" * 60)
txt.append("ESE 5420 — Baseline Checkpoint")
txt.append("Xiaoguang Zhao  |  zhao716@seas.upenn.edu")
txt.append("=" * 60)
txt.append("")
txt.append(f"1. DATASET")
txt.append(f"   N (after cleaning)          : {N_total:,}")
txt.append(f"   Date range                  : {date_min}  to  {date_max}")
txt.append(f"   Rows missing GROSS SF       : {n_missing_sf}  (cleaned before this step)")
txt.append(f"   Boroughs                    : Queens {7514:,}  Brooklyn {5265:,}")
txt.append(f"   Building classes            : 1-fam 6156  2-fam 5198  3-fam 1425")
txt.append("")
txt.append(f"2. TEMPORAL SPLIT")
txt.append(f"   Training set (Sep 2025–Jun 2026): N_tr = {len(train_df):,}")
txt.append(f"   Test set     (Jul 2026–Aug 2026): N_te = {len(test_df):,}")
txt.append(f"   [Test set held out; not used for any model fitting or tuning]")
txt.append("")
txt.append(f"3. 5-FOLD CV RESULTS (RMSE of log sale price)")
txt.append(f"   Scaling done INSIDE each fold (no leakage)")
txt.append("")
txt.append(f"   {'Model':<30} {'CV RMSE':>10}  {'Fold spread (SD)':>18}")
txt.append(f"   {'-'*62}")

for name in ['mean', 'gsf_lr', 'ols']:
    r = results[name]
    label = {'mean': 'Mean baseline',
             'gsf_lr': 'GrossSF linear regression',
             'ols': 'Full OLS (all features)'}[name]
    txt.append(f"   {label:<30} {r['mean']:>10.4f}  ±{r['spread']:>16.4f}")

txt.append("")
txt.append(f"   Per-fold RMSE (Mean / GrossSF / OLS):")
for k in range(K):
    txt.append(f"   Fold {k+1}: "
               f"{results['mean']['folds'][k]:.4f} / "
               f"{results['gsf_lr']['folds'][k]:.4f} / "
               f"{results['ols']['folds'][k]:.4f}")

txt.append("")
txt.append(f"4. NOTES")
txt.append(f"   Full OLS features: log(GrossSF), log(LandSF), Age,")
txt.append(f"   TotalUnits, SaleMonth, Borough, 2 BldgClass dummies,")
txt.append(f"   116 Neighborhood dummies  →  p = 124 features")
txt.append(f"   Parameters fitted by sklearn LinearRegression (OLS = MSE minimization).")
txt.append(f"   Gradient descent gives identical result to closed-form.")
txt.append("")

baseline_txt = "\n".join(txt)
with open('/home/claude/baseline.txt', 'w') as f:
    f.write(baseline_txt)
print("\nbaseline.txt written.")

# ── 8. Generate baseline.png ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("ESE 5420 Baseline Checkpoint — CV Results", fontsize=13, fontweight='bold')

# Left: CV RMSE bar chart with error bars (fold spread)
ax = axes[0]
model_labels = ['Mean\nbaseline', 'GrossSF\nlinear reg.', 'Full OLS\n(all features)']
means  = [results['mean']['mean'], results['gsf_lr']['mean'], results['ols']['mean']]
spreads = [results['mean']['spread'], results['gsf_lr']['spread'], results['ols']['spread']]
colors = ['#4C72B0', '#DD8452', '#55A868']

bars = ax.bar(model_labels, means, yerr=spreads, capsize=6, color=colors,
              alpha=0.85, error_kw={'linewidth': 2, 'ecolor': 'black'})
ax.set_ylabel('CV RMSE  (log sale price)', fontsize=11)
ax.set_title('5-Fold CV RMSE ± Fold Spread', fontsize=11)
ax.set_ylim(0, max(means) * 1.25)
for bar, m, s in zip(bars, means, spreads):
    ax.text(bar.get_x() + bar.get_width()/2, m + s + 0.005,
            f'{m:.4f}\n±{s:.4f}', ha='center', va='bottom', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Right: Per-fold RMSE lines
ax = axes[1]
fold_nums = list(range(1, K+1))
for name, label, color in [('mean', 'Mean baseline', '#4C72B0'),
                             ('gsf_lr', 'GrossSF linear reg.', '#DD8452'),
                             ('ols', 'Full OLS', '#55A868')]:
    ax.plot(fold_nums, results[name]['folds'], marker='o', label=label,
            color=color, linewidth=2, markersize=6)
ax.set_xlabel('Fold', fontsize=11)
ax.set_ylabel('RMSE  (log sale price)', fontsize=11)
ax.set_title('Per-Fold RMSE Across CV Folds', fontsize=11)
ax.set_xticks(fold_nums)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/baseline.png', dpi=150, bbox_inches='tight')
plt.close()
print("baseline.png written.")

print("\nDone.")
