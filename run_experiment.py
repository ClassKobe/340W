import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

import generate_synthetic_data as g

SYM = [f"sym_{s}" for s in g.SYMPTOMS]
KAPPA, LAM = 2.0, 0.01          # class weight on negative weeks, L2 strength
K_WEEKS = 2                      # consecutive good weeks before suggesting a step down


def make_data(seed, users=300, scale=1.6, theta=1.2, noise=0.05):
    rng = np.random.default_rng(seed)
    rows = []
    for uid in range(1, users + 1):
        rows += g.simulate_user(uid, rng, scale, theta, noise)
    df = pd.DataFrame(rows)
    ids = df.user_id.unique()
    test = set(rng.choice(ids, size=int(0.2 * len(ids)), replace=False))
    df["split"] = np.where(df.user_id.isin(test), "test", "train")
    return df


def features(df):
    df = df.sort_values(["user_id", "week"]).reset_index(drop=True)
    X = df[SYM].to_numpy(float)
    prev = df.groupby("user_id")[SYM].shift(1).to_numpy(float)
    prev = np.where(np.isnan(prev), X, prev)            # week 1: no change
    D = np.maximum(0.0, X - prev)
    return df, X, D


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fit_weighted(X, D, y):
    """S = sum_s w_s (x_s + beta * max(0, delta_s)); p = sigmoid(S - tau); class-weighted logistic loss."""
    n_s = X.shape[1]

    def unpack(v):
        return v[:n_s], v[n_s], v[n_s + 1]

    def loss(v):
        w, beta, tau = unpack(v)
        S = X @ w + beta * (D @ w)
        p = np.clip(sigmoid(S - tau), 1e-9, 1 - 1e-9)
        ce = -(KAPPA * y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
        return ce + LAM * (w ** 2).sum()

    v0 = np.concatenate([np.full(n_s, 0.5), [0.5], [4.0]])   # uniform start, no peeking at true loadings
    bounds = [(0, None)] * n_s + [(0, None), (None, None)]
    res = minimize(loss, v0, method="L-BFGS-B", bounds=bounds)
    return unpack(res.x)


def score_weighted(params, X, D):
    w, beta, tau = params
    return sigmoid(X @ w + beta * (D @ w) - tau)


def best_f1_threshold(score, y):
    cands = np.unique(np.quantile(score, np.linspace(0.05, 0.99, 120)))
    f1s = [precision_recall_fscore_support(y, score >= c, average="binary", zero_division=0)[2] for c in cands]
    return cands[int(np.argmax(f1s))]


def week_metrics(y, pred, score=None):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    out = dict(precision=p, recall=r, f1=f, accuracy=(y == pred).mean())
    if score is not None:
        out["pr_auc"] = average_precision_score(y, score)
    return out


def policy_metrics(df, pred_neg):
    """Step down is suggested when the last K weeks were all predicted good (needs >= K weeks of history)."""
    d = df.assign(pred=pred_neg.astype(int))
    roll = d.groupby("user_id")["pred"].transform(lambda s: s.rolling(K_WEEKS).sum())
    step = (roll == 0)
    y = d.label_negative.to_numpy() == 1
    elig = d.week.to_numpy() >= K_WEEKS
    unsafe = (step.to_numpy() & y & elig).sum() / max((y & elig).sum(), 1)
    offered_good = (step.to_numpy() & ~y & elig).sum() / max((~y & elig).sum(), 1)
    return dict(unsafe_stepdown_rate=unsafe, stepdown_offered_in_good_weeks=offered_good)


def run_seed(seed):
    df = make_data(seed)
    df, X, D = features(df)
    y = df.label_negative.to_numpy()
    tr, te = (df.split == "train").to_numpy(), (df.split == "test").to_numpy()
    out = {}

    # Baseline A: unweighted symptom sum, threshold tuned on train for best F1
    sA = X.sum(1)
    tA = best_f1_threshold(sA[tr], y[tr])
    pa = sA >= tA
    out["A_unweighted_sum"] = {**week_metrics(y[te], pa[te], sA[te]), **policy_metrics(df[te], pa[te])}

    # Baseline B: any symptom of moderate severity (>=2) means negative
    sB = X.max(1)
    pb = sB >= 2
    out["B_any_symptom>=2"] = {**week_metrics(y[te], pb[te], sB[te]), **policy_metrics(df[te], pb[te])}

    # Proposed: weight-based classifier
    params = fit_weighted(X[tr], D[tr], y[tr])
    pw = score_weighted(params, X, D)
    out["W_weighted_p>=0.5"] = {**week_metrics(y[te], (pw >= 0.5)[te], pw[te]), **policy_metrics(df[te], (pw >= 0.5)[te])}
    tW = best_f1_threshold(pw[tr], y[tr])
    out["W_weighted_f1tuned"] = {**week_metrics(y[te], (pw >= tW)[te], pw[te]), **policy_metrics(df[te], (pw >= tW)[te])}
    return out, params


if __name__ == "__main__":
    seeds = [1, 2, 3, 4, 5]
    allres = {}
    for sd in seeds:
        res, params = run_seed(sd)
        for k, v in res.items():
            allres.setdefault(k, []).append(v)
        if sd == seeds[0]:
            first_params = params
    lines = [f"Synthetic experiment: {len(seeds)} independent datasets (seeds {seeds}), 300 users each, 80/20 split by user.",
             "Mean (std) over datasets, test users only.\n"]
    metrics = ["pr_auc", "recall", "precision", "f1", "unsafe_stepdown_rate", "stepdown_offered_in_good_weeks"]
    for k, runs in allres.items():
        df_r = pd.DataFrame(runs)
        lines.append(k)
        for m in metrics:
            lines.append(f"  {m:34s} {df_r[m].mean():.3f} ({df_r[m].std():.3f})")
        lines.append("")
    w, beta, tau = first_params
    lines.append("Learned weights (seed 1): " + ", ".join(f"{s}={x:.2f}" for s, x in zip(g.SYMPTOMS, w)) + f"; beta={beta:.2f}; tau={tau:.2f}")
    text = "\n".join(lines)
    print(text)
    open("results.txt", "w").write(text + "\n")
