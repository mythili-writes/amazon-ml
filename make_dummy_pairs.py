"""
Make a FAKE pairs file so train.py can be tested before real features arrive.
Owner: Mythili (Person 3)

It uses the REAL ground truth, so the S1 IDs and match counts are real.
The features are invented. DELETE this file before the final submission.
"""

import glob
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
SAMPLE_S1 = 30000     # small sample = fast test
BLOCK_RECALL = 0.90   # pretend blocking finds 90% of true matches
DECOYS = 6            # wrong candidates per entity

GT_PATH = glob.glob("/kaggle/input/**/train_ground_truth.tsv", recursive=True)[0]
gt = pd.read_csv(GT_PATH, sep="\t", dtype=str, keep_default_na=False)
gt.columns = ["s1_id", "matched"]

gt = gt.sample(min(SAMPLE_S1, len(gt)), random_state=42)

rows = []
for s1, matched in zip(gt["s1_id"], gt["matched"]):
    truth = [x.strip() for x in matched.split(",") if x.strip()]
    for t in truth:                                   # positives blocking found
        if rng.random() < BLOCK_RECALL:
            rows.append((s1, t, 1))
    for k in range(DECOYS):                           # wrong candidates
        rows.append((s1, f"S2-FAKE{rng.integers(1, 999999)}", 0))

p = pd.DataFrame(rows, columns=["s1_id", "cand_id", "label"])

# Invented features that carry some signal, like real similarity scores would
for col, hi, lo, sd in [("name_sim", 0.80, 0.35, 0.15),
                        ("addr_sim", 0.70, 0.40, 0.20),
                        ("token_overlap", 0.75, 0.30, 0.25)]:
    centre = np.where(p["label"] == 1, hi, lo)
    p[col] = np.clip(rng.normal(centre, sd), 0, 1)

p.to_parquet("/kaggle/working/candidates_with_features.parquet")
print("Rows:", len(p), "| S1 entities:", p["s1_id"].nunique())
print("Positive rate:", round(p["label"].mean(), 4))
print("Saved to /kaggle/working/candidates_with_features.parquet")
