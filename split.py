"""
Validation split grouped by S1 ID.
Owner: Mythili (Person 3)

Idea: split the S1 entities (not the pairs). Every pair row for one S1 ID
then lands in the same side, so the model never sees a validation entity
during training.
"""

import glob
import pandas as pd
from sklearn.model_selection import train_test_split

VAL_SIZE = 0.2   # 20% of S1 entities go to validation
SEED = 42        # fixed seed = same split every run, for everyone

# 1. Find the files
GT_PATH = glob.glob("/kaggle/input/**/train_ground_truth.tsv", recursive=True)[0]
S1_PATH = glob.glob("/kaggle/input/**/train_source1.tsv", recursive=True)[0]

# 2. Load ground truth (fast version, no row-by-row loop)
gt = pd.read_csv(GT_PATH, sep="\t", dtype=str, keep_default_na=False)
gt.columns = ["s1_id", "matched"]
gt["n_matches"] = gt["matched"].apply(lambda x: 0 if x.strip() == "" else len(x.split(",")))

# 3. Add country from Source 1 (to keep the country mix the same in both sides)
s1 = pd.read_csv(S1_PATH, sep="\t", dtype=str, keep_default_na=False, usecols=["entity_id", "country"])
gt = gt.merge(s1.rename(columns={"entity_id": "s1_id"}), on="s1_id", how="left")
gt["country"] = gt["country"].fillna("UNKNOWN")

# 4. Stratify by country + singleton-or-not
gt["strata"] = gt["country"] + "_" + (gt["n_matches"] == 0).astype(str)

train_ids, val_ids = train_test_split(
    gt["s1_id"], test_size=VAL_SIZE, random_state=SEED, stratify=gt["strata"]
)

# 5. Checks (all must print True)
print("No overlap:", len(set(train_ids) & set(val_ids)) == 0)
print("Nothing lost:", len(train_ids) + len(val_ids) == len(gt))
print("Train S1:", len(train_ids), "| Val S1:", len(val_ids))
for name, ids in [("train", train_ids), ("val", val_ids)]:
    part = gt[gt["s1_id"].isin(set(ids))]
    print(f"{name}: singleton share = {(part['n_matches'] == 0).mean():.4f}, "
          f"countries = {part['country'].value_counts(normalize=True).round(3).to_dict()}")

# 6. Save the ID lists so the whole team uses the SAME split
pd.Series(sorted(train_ids), name="s1_id").to_csv("/kaggle/working/train_s1_ids.csv", index=False)
pd.Series(sorted(val_ids), name="s1_id").to_csv("/kaggle/working/val_s1_ids.csv", index=False)
print("Saved to /kaggle/working/")
