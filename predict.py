"""
Make the two submission files from the trained model.
Owner: Mythili (Person 3)

Writes:
  /kaggle/working/output/matching_results.tsv   <- scored on the leaderboard
  /kaggle/working/output/candidate_pairs.tsv    <- blocking set, not scored
"""

import glob, json, os
import joblib
import pandas as pd

TEST_PAIRS = "/kaggle/working/test_candidates_with_features.parquet"  # from the team
OUT_DIR = "/kaggle/working/output"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load model + the threshold tuned on validation
model = joblib.load("/kaggle/working/lgbm_model.joblib")
cfg = json.load(open("/kaggle/working/model_settings.json"))
THRESH, TOPK, FEATURES = cfg["threshold"], cfg["topk"], cfg["features"]
print(f"Using threshold {THRESH}, top-k {TOPK} (val F0.5 was {cfg['val_f05']:.4f})")

# 2. EVERY test S1 entity must appear, even with no candidates
s1_path = glob.glob("/kaggle/input/**/test_source1.tsv", recursive=True)[0]
all_s1 = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)["entity_id"]
print("Test S1 entities:", len(all_s1))

# 3. Valid target IDs (Source 2 and 3 only — self-matches are rejected)
valid_ids = set()
for name in ("test_source2.tsv", "test_source3.tsv"):
    path = glob.glob(f"/kaggle/input/**/{name}", recursive=True)[0]
    valid_ids |= set(pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)["entity_id"])
print("Valid S2/S3 IDs:", len(valid_ids))

# 4. Score the candidates
pairs = pd.read_parquet(TEST_PAIRS)
pairs = pairs[pairs["cand_id"].isin(valid_ids)]           # drop anything not in the test set
pairs = pairs.drop_duplicates(["s1_id", "cand_id"])       # no duplicate IDs in a list
pairs["proba"] = model.predict_proba(pairs[FEATURES])[:, 1]
pairs["rank"] = pairs.groupby("s1_id")["proba"].rank(ascending=False, method="first")


def write(df, path, col):
    """One row per test S1 entity, empty string when there is nothing."""
    joined = df.groupby("s1_id")["cand_id"].apply(",".join)
    out = pd.DataFrame({"source1_entity_id": all_s1})
    out[col] = out["source1_entity_id"].map(joined).fillna("")
    out.to_csv(path, sep="\t", index=False)
    return out


# 5. candidate_pairs.tsv = everything the model looked at
cands = write(pairs, f"{OUT_DIR}/candidate_pairs.tsv", "candidate_entity_ids")

# 6. matching_results.tsv = what passed the threshold
hits = pairs[(pairs["proba"] >= THRESH) & (pairs["rank"] <= TOPK)]
res = write(hits, f"{OUT_DIR}/matching_results.tsv", "matched_entity_ids")

# 7. Self-checks (all must print True)
print("\n--- checks ---")
print("One row per test S1:", len(res) == len(all_s1) == res["source1_entity_id"].nunique())
print("Columns correct:", list(res.columns) == ["source1_entity_id", "matched_entity_ids"])
sets = res["matched_entity_ids"].apply(lambda x: [i for i in x.split(",") if i])
print("No duplicates in a list:", all(len(v) == len(set(v)) for v in sets))
print("No S1 IDs matched to themselves:", not any(i.startswith("S1-") for v in sets for i in v))
print("All matched IDs exist in test:", all(i in valid_ids for v in sets for i in v))
csets = cands["candidate_entity_ids"].apply(lambda x: {i for i in x.split(",") if i})
print("Matches are a subset of candidates:", all(set(m) <= c for m, c in zip(sets, csets)))
print("\nEntities with a match:", (res["matched_entity_ids"] != "").sum(),
      f"({(res['matched_entity_ids'] != '').mean():.1%})")
print("Predicted singletons:", (res["matched_entity_ids"] == "").sum())
