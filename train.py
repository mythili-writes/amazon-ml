"""
LightGBM training + threshold tuning + matching_results.tsv.
Owner: Mythili (Person 3)

Input: a pairs table with columns s1_id, cand_id, label + feature columns.
      (column names s1_id, cand_id, label are fixed by team agreement)
Output: /kaggle/working/matching_results.tsv
"""

import glob
import numpy as np
import pandas as pd
import lightgbm as lgb

# ---------- fast macro F0.5 ----------

def macro_f05(s1_codes, n_s1, correct_mask, keep_mask, n_true):
    """
    Score all S1 entities at once, using numpy (fast, no python loop).

    s1_codes    : int code 0..n_s1-1 for each candidate row
    n_s1        : total number of S1 entities being scored
    correct_mask: True where label == 1
    keep_mask   : True where we predict "match"
    n_true      : array, true number of matches per S1 entity (from ground truth)
    """
    pred_n = np.bincount(s1_codes[keep_mask], minlength=n_s1).astype(np.float64)
    hit_n = np.bincount(s1_codes[keep_mask & correct_mask], minlength=n_s1).astype(np.float64)
    true_n = n_true.astype(np.float64)

    scores = np.zeros(n_s1)
    # Singletons: 1.0 only if we predicted nothing
    singles = true_n == 0
    scores[singles] = (pred_n[singles] == 0).astype(np.float64)
    # Non-singletons with at least one correct hit
    ok = (~singles) & (hit_n > 0)
    p = hit_n[ok] / pred_n[ok]
    r = hit_n[ok] / true_n[ok]
    scores[ok] = 1.25 * p * r / (0.25 * p + r)
    # everything else stays 0.0
    return scores.mean()


# ---------- load ----------

PAIRS_PATH = "/kaggle/working/candidates_with_features.parquet"  # from Soumitha

# TRUE only for a smoke test on a SAMPLE of entities.
# MUST be False for the real run, or the score will be fake and too high.
SMOKE_TEST = False
GT_PATH = glob.glob("/kaggle/input/**/train_ground_truth.tsv", recursive=True)[0]

pairs = pd.read_parquet(PAIRS_PATH)
val_ids = set(pd.read_csv("/kaggle/working/val_s1_ids.csv")["s1_id"])

gt = pd.read_csv(GT_PATH, sep="\t", dtype=str, keep_default_na=False)
gt.columns = ["s1_id", "matched"]
gt["n_true"] = gt["matched"].apply(lambda x: 0 if x.strip() == "" else len(x.split(",")))

FEATURES = [c for c in pairs.columns if c not in ("s1_id", "cand_id", "label")]
print("Features:", len(FEATURES))

tr = pairs[~pairs["s1_id"].isin(val_ids)]
va = pairs[pairs["s1_id"].isin(val_ids)].copy()
print("Train rows:", len(tr), "| Val rows:", len(va))
print("Positive rate:", round(tr["label"].mean(), 4))

# ---------- train ----------

model = lgb.LGBMClassifier(
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=50,
    subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
model.fit(
    tr[FEATURES], tr["label"],
    eval_set=[(va[FEATURES], va["label"])],
    eval_metric="average_precision",
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
)

va["proba"] = model.predict_proba(va[FEATURES])[:, 1]

# ---------- set up scoring over ALL val entities ----------
# IMPORTANT: every validation S1 entity is scored, even ones whose true
# matches blocking never produced. Those misses must count against us.

val_gt = gt[gt["s1_id"].isin(val_ids)]
if SMOKE_TEST:
    val_gt = val_gt[val_gt["s1_id"].isin(set(va["s1_id"]))]
    print("\n*** SMOKE TEST: scoring only", len(val_gt), "sampled entities. Not a real score. ***")
val_gt = val_gt.reset_index(drop=True)
code_of = {s: i for i, s in enumerate(val_gt["s1_id"])}
n_s1 = len(val_gt)
n_true = val_gt["n_true"].to_numpy()

codes = va["s1_id"].map(code_of).to_numpy()
correct = (va["label"] == 1).to_numpy()
proba = va["proba"].to_numpy()

# rank of each candidate within its S1 entity (1 = best)
va["rank"] = va.groupby("s1_id")["proba"].rank(ascending=False, method="first")
rank = va["rank"].to_numpy()

# ---------- tune threshold (and top-k) ----------

print("\nthresh  topk   F0.5")
best = (0.0, 0.5, 99)
for topk in (99, 1, 2, 3, 5):
    for t in np.arange(0.05, 0.96, 0.05):
        keep = (proba >= t) & (rank <= topk)
        s = macro_f05(codes, n_s1, correct, keep, n_true)
        if s > best[0]:
            best = (s, float(t), topk)
        if topk == 99:
            print(f"{t:.2f}    all   {s:.4f}")

print(f"\nBEST: F0.5 = {best[0]:.4f} at threshold {best[1]:.2f}, top-k {best[2]}")
print("Baseline to beat (predict nothing): 0.0558")

# ---------- write matching_results.tsv ----------

def write_results(df, s1_all, thresh, topk, path):
    """One row per S1 entity. Empty string when no match is predicted."""
    hits = df[(df["proba"] >= thresh) & (df["rank"] <= topk)]
    grouped = hits.groupby("s1_id")["cand_id"].apply(lambda x: ",".join(dict.fromkeys(x)))
    out = pd.DataFrame({"source1_entity_id": s1_all})
    out["matched_entity_ids"] = out["source1_entity_id"].map(grouped).fillna("")
    out.to_csv(path, sep="\t", index=False)
    return out

out = write_results(va, val_gt["s1_id"], best[1], best[2], "/kaggle/working/val_matching_results.tsv")
print("\nRows written:", len(out), "| rows with a match:", (out["matched_entity_ids"] != "").sum())

# ---------- save model + chosen settings for predict.py ----------
import joblib, json
if not SMOKE_TEST:
    joblib.dump(model, "/kaggle/working/lgbm_model.joblib")
    json.dump({"threshold": best[1], "topk": best[2], "val_f05": best[0],
               "features": FEATURES},
              open("/kaggle/working/model_settings.json", "w"), indent=2)
    print("Saved model + settings to /kaggle/working/")
else:
    print("SMOKE_TEST is on, so model was NOT saved.")
