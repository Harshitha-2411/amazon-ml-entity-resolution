import pandas as pd
import re


# ============================================
# LOAD DATA
# ============================================

s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
)

s2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t"
)

s3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t"
)

gt = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)

gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")


# ============================================
# NORMALIZATION
# ============================================

def normalize(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = text.replace("&", " and ")

    text = re.sub(r"[^\w\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


for df in [s1, s2, s3]:

    df["name_norm"] = (
        df["business_name"]
        .fillna("")
        .apply(normalize)
    )

    df["address_norm"] = (
        df["business_address"]
        .fillna("")
        .apply(normalize)
    )


# ============================================
# CREATE SIMPLE BLOCKING KEYS
# ============================================

def name_tokens(text):

    return set(
        token
        for token in text.split()
        if len(token) >= 3
    )


def address_tokens(text):

    return set(
        token
        for token in text.split()
        if len(token) >= 3
    )


# ============================================
# BUILD INDICES
# ============================================

print("Building blocking indices...")


name_index_s2 = {}

for idx, row in s2.iterrows():

    for token in name_tokens(row["name_norm"]):

        key = (row["country"], token)

        name_index_s2.setdefault(key, set()).add(idx)


name_index_s3 = {}

for idx, row in s3.iterrows():

    for token in name_tokens(row["name_norm"]):

        key = (row["country"], token)

        name_index_s3.setdefault(key, set()).add(idx)


address_index_s2 = {}

for idx, row in s2.iterrows():

    for token in address_tokens(row["address_norm"]):

        key = (row["country"], token)

        address_index_s2.setdefault(key, set()).add(idx)


address_index_s3 = {}

for idx, row in s3.iterrows():

    for token in address_tokens(row["address_norm"]):

        key = (row["country"], token)

        address_index_s3.setdefault(key, set()).add(idx)


print("Indices created.")


# ============================================
# TEST BLOCKING
# ============================================

def get_candidates(row, source, name_index, address_index):

    candidates = set()

    country = row["country"]

    # Name-token blocking
    for token in name_tokens(row["name_norm"]):

        key = (country, token)

        candidates.update(
            name_index.get(key, set())
        )

    # Address-token blocking
    for token in address_tokens(row["address_norm"]):

        key = (country, token)

        candidates.update(
            address_index.get(key, set())
        )

    return candidates


# ============================================
# TEST FIRST 100 MATCHED ENTITIES
# ============================================

matched_gt = gt[
    gt["matched_entity_ids"].str.strip() != ""
].head(100)


total_true_matches = 0
retrieved_matches = 0

candidate_counts = []


for _, gt_row in matched_gt.iterrows():

    s1_id = gt_row["source1_entity_id"]

    true_matches = set(
        x.strip()
        for x in gt_row["matched_entity_ids"].split(",")
    )

    source1_row = s1[
        s1["entity_id"] == s1_id
    ].iloc[0]


    # S2 candidates
    s2_candidates = get_candidates(
        source1_row,
        "S2",
        name_index_s2,
        address_index_s2
    )

    # S3 candidates
    s3_candidates = get_candidates(
        source1_row,
        "S3",
        name_index_s3,
        address_index_s3
    )


    candidate_ids = set(
        s2.loc[list(s2_candidates), "entity_id"]
    )

    candidate_ids.update(
        s3.loc[list(s3_candidates), "entity_id"]
    )


    found = true_matches.intersection(candidate_ids)


    total_true_matches += len(true_matches)
    retrieved_matches += len(found)

    candidate_counts.append(len(candidate_ids))


# ============================================
# RESULTS
# ============================================

recall = (
    retrieved_matches / total_true_matches
    if total_true_matches > 0
    else 0
)

print("\n========== BLOCKING RESULTS ==========")

print(
    "True matches:",
    total_true_matches
)

print(
    "Matches retrieved:",
    retrieved_matches
)

print(
    "Blocking recall:",
    round(recall * 100, 2),
    "%"
)

print(
    "Average candidates per Source 1:",
    round(
        sum(candidate_counts) / len(candidate_counts),
        2
    )
)

print(
    "Maximum candidates:",
    max(candidate_counts)
)

print(
    "Minimum candidates:",
    min(candidate_counts)
)