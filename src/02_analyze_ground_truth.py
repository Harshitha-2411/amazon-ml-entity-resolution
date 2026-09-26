import pandas as pd


# ============================================
# LOAD GROUND TRUTH
# ============================================

gt = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)


# ============================================
# CHECK EMPTY MATCHES
# ============================================

gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")


# Number of matches for each Source 1 entity
gt["match_count"] = gt["matched_entity_ids"].apply(
    lambda x: 0 if x.strip() == "" else len(x.split(","))
)


# ============================================
# BASIC STATISTICS
# ============================================

print("\n========== GROUND TRUTH STATISTICS ==========")

print("Total Source 1 entities:", len(gt))

print(
    "Entities with no matches:",
    (gt["match_count"] == 0).sum()
)

print(
    "Entities with at least one match:",
    (gt["match_count"] > 0).sum()
)

print(
    "Total matched IDs:",
    gt["match_count"].sum()
)


# ============================================
# DISTRIBUTION
# ============================================

print("\n========== MATCH COUNT DISTRIBUTION ==========")

print(
    gt["match_count"]
    .value_counts()
    .sort_index()
)


# ============================================
# SUMMARY STATISTICS
# ============================================

print("\n========== SUMMARY ==========")

print(
    gt["match_count"].describe()
)


# ============================================
# EXAMPLES OF NO MATCH
# ============================================

print("\n========== EXAMPLES: NO MATCH ==========")

print(
    gt[gt["match_count"] == 0]
    .head(10)
)


# ============================================
# EXAMPLES WITH MANY MATCHES
# ============================================

print("\n========== EXAMPLES: MANY MATCHES ==========")

print(
    gt[gt["match_count"] >= 5]
    .sort_values("match_count", ascending=False)
    .head(10)
)