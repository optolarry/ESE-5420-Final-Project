"""Sign-flip tests (course Section 4D) on the July-August 2026 test predictions.

For each pair (A, B) the hypothesis H1 "A has lower expected squared error than B"
was fixed as the direction of the research question. Per test sample i,
d_i = e_i^B - e_i^A, where e_i is the squared error of log price.
The statistic is the mean of d_i. Under H0 the sign of each d_i is a coin flip:
flip signs at random 10,000 times and report the share of flipped means that are
at least as large as the observed mean (one-sided p-value).

Reads temporal_predictions.csv written by run_analysis.py. No model is refitted.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "temporal_predictions.csv"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent / "signflip_tests.csv"
FLIPS, SEED = 10_000, 42
# (A, B): H1 says A is better than B
PAIRS = [("add_OLS", "struct_OLS"), ("add_Ridge", "add_OLS"), ("add_Lasso", "add_OLS"),
         ("interaction_OLS", "add_OLS"), ("interaction_Ridge", "interaction_OLS"),
         ("interaction_Lasso", "interaction_OLS")]

d = pd.read_csv(SRC)
y = d["y"].to_numpy()
rng = np.random.default_rng(SEED)
rows = []
for a, b in PAIRS:
    ea, eb = (y - d[a].to_numpy()) ** 2, (y - d[b].to_numpy()) ** 2
    diff = eb - ea
    obs = diff.mean()
    count = 0
    for start in range(0, FLIPS, 500):
        signs = rng.choice([-1.0, 1.0], size=(500, len(diff)))
        count += int(((signs * diff).mean(axis=1) >= obs).sum())
    rows.append({"model_A": a, "model_B": b, "n_test": len(diff),
                 "mse_A": ea.mean(), "mse_B": eb.mean(), "mean_d": obs,
                 "rmse_A": np.sqrt(ea.mean()), "rmse_B": np.sqrt(eb.mean()),
                 "flips": FLIPS, "flips_at_least_observed": count, "p_value": count / FLIPS})
out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print(out.to_string(index=False))
