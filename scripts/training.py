import os, json, math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss, confusion_matrix

SENSITIVE_ATTR = "city_tier"
APPROVE_RATE = 0.40

def group_metrics(y_true, y_pred, y_prob, groups):
    out = {}
    for g in sorted(np.unique(groups)):
        mask = (groups == g)
        yt, yp = y_true[mask], y_pred[mask]
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0,1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) else float("nan")
        fpr = fp / (fp + tn) if (fp + tn) else float("nan")
        ppv = tp / (tp + fp) if (tp + fp) else float("nan")
        out[int(g)] = {"TPR": float(tpr), "FPR": float(fpr), "PPV": float(ppv), "PositiveRate": float(yp.mean()), "Count": int(mask.sum())}
    return out

def disparity(metric_name, table):
    vals = [v[metric_name] for v in table.values() if not np.isnan(v[metric_name])]
    if len(vals) <= 1: return 0.0
    return float(max(vals) - min(vals))

def proba_to_score(p):
    return np.clip(300 + (p * 600), 300, 900)

def main():
    df = pd.read_csv("data/partners.csv")
    X = df.drop(columns=["good_repayment", "partner_id"])
    y = df["good_repayment"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    num_cols = ["tenure_months","days_active","trips_per_week","earnings_avg","earnings_var","on_time_rate","cancel_rate",
                "customer_rating","complaints","accidents","night_shift_pct","cashless_ratio","wallet_txn_volume","vehicle_age"]
    cat_cols = ["role","gender","age_group","city_tier"]

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    # Baseline
    base = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=200, solver="lbfgs"))])
    base.fit(X_tr, y_tr)
    pb = base.predict_proba(X_te)[:,1]
    thr = np.quantile(pb, 1-APPROVE_RATE)
    yhb = (pb >= thr).astype(int)

    auc_b = roc_auc_score(y_te, pb)
    f1_b = f1_score(y_te, yhb)
    brier_b = brier_score_loss(y_te, pb)

    gm_b = group_metrics(y_te.values, yhb, pb, X_te[SENSITIVE_ATTR].values)
    dpd_b = disparity("PositiveRate", gm_b)
    eod_b = disparity("TPR", gm_b)
    ppd_b = disparity("PPV", gm_b)

    # Reweighing
    tr = X_tr.copy(); tr["y"] = y_tr.values
    A = SENSITIVE_ATTR
    weights = np.ones(len(tr))
    global_pos = tr["y"].mean()
    for a_val in tr[A].unique():
        mask_a = tr[A] == a_val
        pos_a = tr.loc[mask_a, "y"].mean()
        w_pos = (global_pos / pos_a) if pos_a > 0 else 1.0
        w_neg = ((1 - global_pos) / (1 - pos_a)) if pos_a < 1 else 1.0
        idx_pos = mask_a & (tr["y"] == 1)
        idx_neg = mask_a & (tr["y"] == 0)
        weights[idx_pos.values] = w_pos
        weights[idx_neg.values] = w_neg

    deb = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=300, solver="lbfgs"))])
    deb.fit(X_tr, y_tr, clf__sample_weight=weights)
    pd_ = deb.predict_proba(X_te)[:,1]
    thr_d = np.quantile(pd_, 1-APPROVE_RATE)
    yhd = (pd_ >= thr_d).astype(int)

    auc_d = roc_auc_score(y_te, pd_)
    f1_d = f1_score(y_te, yhd)
    brier_d = brier_score_loss(y_te, pd_)

    gm_d = group_metrics(y_te.values, yhd, pd_, X_te[SENSITIVE_ATTR].values)
    dpd_d = disparity("PositiveRate", gm_d)
    eod_d = disparity("TPR", gm_d)
    ppd_d = disparity("PPV", gm_d)

    scores = pd.DataFrame({
        "partner_id": df.loc[X_te.index, "partner_id"].values,
        "score_300_900": proba_to_score(pd_),
        "prob_good": pd_,
        "approved_at_policy": yhd,
        SENSITIVE_ATTR: X_te[SENSITIVE_ATTR].values
    })
    os.makedirs("outputs", exist_ok=True)
    scores.to_csv("outputs/credit_scores_test.csv", index=False)

    metrics = {
        "policy": {"approve_rate": APPROVE_RATE, "threshold_base": float(thr), "threshold_deb": float(thr_d)},
        "performance": {"baseline": {"AUC": float(auc_b), "F1": float(f1_b), "Brier": float(brier_b)},
                        "debiased": {"AUC": float(auc_d), "F1": float(f1_d), "Brier": float(brier_d)}},
        "fairness_city_tier": {"baseline": {"DPD": float(dpd_b), "EOD": float(eod_b), "PPD": float(ppd_b), "by_group": gm_b},
                               "debiased": {"DPD": float(dpd_d), "EOD": float(eod_d), "PPD": float(ppd_d), "by_group": gm_d}}
    }
    with open("outputs/metrics.json","w") as f:
        json.dump(metrics, f, indent=2)

