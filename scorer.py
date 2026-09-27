"""
F0.5 scorer for the Business Entity Resolution Challenge.
Owner: Mythili (Person 3)

Rules (from the problem statement):
  - F0.5 is computed for EACH Source 1 entity, then averaged (macro-average).
  - ALL Source 1 entities count, including singletons (no true matches).
  - Singleton: predict empty -> 1.0, predict anything -> 0.0.
  - F0.5 = 1.25 * P * R / (0.25 * P + R)
"""

import pandas as pd


def parse_ids(text):
    """Turn 'S2-1,S3-4' into a set {'S2-1', 'S3-4'}. Empty text -> empty set."""
    if text is None:
        return set()
    text = str(text).strip()
    if text == "" or text.lower() == "nan":
        return set()
    return {x.strip() for x in text.split(",") if x.strip()}


def f05_one(pred, truth):
    """F0.5 for ONE Source 1 entity. pred and truth are sets of IDs."""
    # Singleton case: no true matches
    if len(truth) == 0:
        return 1.0 if len(pred) == 0 else 0.0
    # Has true matches, but we predicted nothing -> score 0
    if len(pred) == 0:
        return 0.0
    correct = len(pred & truth)
    if correct == 0:
        return 0.0
    precision = correct / len(pred)
    recall = correct / len(truth)
    return 1.25 * precision * recall / (0.25 * precision + recall)


def f05_macro(pred_dict, truth_dict):
    """
    Average F0.5 over every S1 entity in truth_dict.
    pred_dict / truth_dict: {s1_id: set_of_matched_ids}
    An S1 ID missing from pred_dict is treated as "predicted empty".
    """
    scores = [f05_one(pred_dict.get(s1, set()), truth) for s1, truth in truth_dict.items()]
    return sum(scores) / len(scores)


def load_tsv(path):
    """Read a ground-truth or matching_results TSV into {s1_id: set}."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {row.iloc[0]: parse_ids(row.iloc[1]) for _, row in df.iterrows()}


def pairs_to_dict(pairs_df, s1_ids, pred_col="pred"):
    """
    Turn a pairs table (columns s1_id, cand_id, pred) into {s1_id: set}.
    pred = 1 means "match". s1_ids = every S1 entity we must score,
    so entities with no predicted match still get an empty set.
    """
    out = {s1: set() for s1 in s1_ids}
    hits = pairs_df[pairs_df[pred_col] == 1]
    for s1, cand in zip(hits["s1_id"], hits["cand_id"]):
        out[s1].add(cand)
    return out


if __name__ == "__main__":
    # ---- Self-tests. All must print True. ----
    # 1. Example from the problem statement (expected 0.714)
    p = {"S2-00047", "S2-00193", "S3-00812"}
    t = {"S2-00047", "S3-00812"}
    print("PDF example 0.714:", round(f05_one(p, t), 3) == 0.714)
    # 2. Singletons
    print("Singleton, predict empty = 1.0:", f05_one(set(), set()) == 1.0)
    print("Singleton, predict a match = 0.0:", f05_one({"S2-1"}, set()) == 0.0)
    # 3. Has matches but predicted nothing
    print("Missed all = 0.0:", f05_one(set(), {"S2-1"}) == 0.0)
    # 4. Perfect match
    print("Perfect = 1.0:", f05_one({"S2-1", "S3-2"}, {"S2-1", "S3-2"}) == 1.0)
    # 5. Macro-average: (0.714 + 1.0 + 0.0) / 3, missing S1 treated as empty
    truth = {"S1-1": t, "S1-2": set(), "S1-3": {"S3-9"}}
    pred = {"S1-1": p}  # S1-2 and S1-3 missing -> empty
    print("Macro average:", round(f05_macro(pred, truth), 4) == round((0.7143 + 1.0 + 0.0) / 3, 4))
    # 6. Parsing
    print("Parse empty:", parse_ids("") == set() and parse_ids(float("nan")) == set())
    print("Parse list:", parse_ids("S2-1, S3-2") == {"S2-1", "S3-2"})
