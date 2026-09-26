import os
import time
import pandas as pd

from blocking import (
    normalize_text,
    build_exact_index,
    build_token_index,
    build_number_index,
    calculate_token_frequency,
    exact_name_block,
    rare_name_block,
    rare_address_block,
    address_number_block,
    combine_candidates
)


# =========================================================
# CONFIGURATION
# =========================================================

DATASET_DIR = "dataset/train"

S1_FILE = os.path.join(DATASET_DIR, "train_source1.tsv")
S2_FILE = os.path.join(DATASET_DIR, "train_source2.tsv")
S3_FILE = os.path.join(DATASET_DIR, "train_source3.tsv")
GT_FILE = os.path.join(DATASET_DIR, "train_ground_truth.tsv")

SAMPLE_SIZE = 1000

# We will test several rarity thresholds.
RARE_THRESHOLDS = [10, 50, 100]


# =========================================================
# LOAD DATA
# =========================================================

print("=" * 70)
print("BLOCKING BENCHMARK")
print("=" * 70)

print("\nLoading data...")

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

s2 = pd.read_csv(
    S2_FILE,
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

s3 = pd.read_csv(
    S3_FILE,
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

gt = pd.read_csv(
    GT_FILE,
    sep="\t"
)


print(f"S1 rows: {len(s1):,}")
print(f"S2 rows: {len(s2):,}")
print(f"S3 rows: {len(s3):,}")


# =========================================================
# NORMALIZE
# =========================================================

print("\nNormalizing text...")

for df in [s1, s2, s3]:

    df["name_norm"] = (
        df["business_name"]
        .fillna("")
        .map(normalize_text)
    )

    df["address_norm"] = (
        df["business_address"]
        .fillna("")
        .map(normalize_text)
    )


# =========================================================
# CREATE ENTITY-ID ARRAYS
# =========================================================

s2_ids = s2["entity_id"].values
s3_ids = s3["entity_id"].values


# =========================================================
# BUILD BLOCKING INDEXES
# =========================================================

print("\nBuilding indexes...")

index_start = time.perf_counter()

s2_name_index = build_exact_index(
    s2,
    "name_norm"
)

s3_name_index = build_exact_index(
    s3,
    "name_norm"
)

s2_name_token_index = build_token_index(
    s2,
    "name_norm"
)

s3_name_token_index = build_token_index(
    s3,
    "name_norm"
)

s2_address_token_index = build_token_index(
    s2,
    "address_norm"
)

s3_address_token_index = build_token_index(
    s3,
    "address_norm"
)

s2_number_index = build_number_index(
    s2,
    "address_norm"
)

s3_number_index = build_number_index(
    s3,
    "address_norm"
)

print(
    f"Index building time: "
    f"{time.perf_counter() - index_start:.2f} seconds"
)


# =========================================================
# TOKEN FREQUENCIES
# =========================================================

print("\nCalculating token frequencies...")

frequency_start = time.perf_counter()

s2_name_frequency = calculate_token_frequency(
    s2,
    "name_norm"
)

s3_name_frequency = calculate_token_frequency(
    s3,
    "name_norm"
)

s2_address_frequency = calculate_token_frequency(
    s2,
    "address_norm"
)

s3_address_frequency = calculate_token_frequency(
    s3,
    "address_norm"
)

print(
    f"Frequency calculation time: "
    f"{time.perf_counter() - frequency_start:.2f} seconds"
)


# =========================================================
# GROUND TRUTH
# =========================================================

gt["matched_entity_ids"] = (
    gt["matched_entity_ids"]
    .fillna("")
)


def parse_matches(value):

    if not value:
        return set()

    return {
        x.strip()
        for x in value.split(",")
        if x.strip()
    }


gt["true_matches"] = gt["matched_entity_ids"].map(
    parse_matches
)


# Keep only entities that actually have matches.
gt_with_matches = gt[
    gt["true_matches"].map(len) > 0
].copy()


# =========================================================
# SELECT SAMPLE
# =========================================================

sample_gt = gt_with_matches.head(SAMPLE_SIZE)

sample_ids = set(
    sample_gt["source1_entity_id"]
)


sample_s1 = s1[
    s1["entity_id"].isin(sample_ids)
].copy()


# Preserve GT order
sample_s1 = (
    sample_s1
    .set_index("entity_id")
    .loc[sample_gt["source1_entity_id"]]
    .reset_index()
)


print("\nBenchmark sample:")
print(f"S1 entities: {len(sample_s1):,}")


# =========================================================
# RESULT STORAGE
# =========================================================

results = []


# =========================================================
# BENCHMARK FUNCTION
# =========================================================

def evaluate_blocker(name, candidate_function):

    print("\n" + "-" * 70)
    print(f"TESTING: {name}")
    print("-" * 70)

    total_candidates = 0
    max_candidates = 0

    matched_entities = 0
    total_true_matches = 0
    retrieved_true_matches = 0

    start_time = time.perf_counter()

    for _, row in sample_s1.iterrows():

        true_matches = set(
            sample_gt.loc[
                sample_gt["source1_entity_id"]
                == row["entity_id"],
                "true_matches"
            ].iloc[0]
        )

        candidates = candidate_function(row)

        candidate_count = len(candidates)

        total_candidates += candidate_count
        max_candidates = max(
            max_candidates,
            candidate_count
        )

        total_true_matches += len(true_matches)

        retrieved = true_matches.intersection(
            candidates
        )

        retrieved_true_matches += len(retrieved)

        if retrieved:
            matched_entities += 1

    runtime = time.perf_counter() - start_time

    recall = (
        retrieved_true_matches / total_true_matches
        if total_true_matches > 0
        else 0
    )

    entity_recall = (
        matched_entities / len(sample_s1)
        if len(sample_s1) > 0
        else 0
    )

    average_candidates = (
        total_candidates / len(sample_s1)
        if len(sample_s1) > 0
        else 0
    )

    result = {
        "blocker": name,
        "recall": recall,
        "entity_recall": entity_recall,
        "average_candidates": average_candidates,
        "max_candidates": max_candidates,
        "runtime_seconds": runtime
    }

    results.append(result)

    print(f"Match recall       : {recall:.4%}")
    print(f"Entity recall      : {entity_recall:.4%}")
    print(f"Average candidates : {average_candidates:,.2f}")
    print(f"Maximum candidates : {max_candidates:,}")
    print(f"Runtime            : {runtime:.2f} seconds")


# =========================================================
# 1. EXACT NAME BLOCK
# =========================================================

def exact_name_candidates(row):

    candidates = set()

    s2_rows = exact_name_block(
        row["name_norm"],
        s2_name_index
    )

    for idx in s2_rows:
        candidates.add(
            s2_ids[idx]
        )

    s3_rows = exact_name_block(
        row["name_norm"],
        s3_name_index
    )

    for idx in s3_rows:
        candidates.add(
            s3_ids[idx]
        )

    return candidates


evaluate_blocker(
    "Exact normalized name",
    exact_name_candidates
)


# =========================================================
# 2. RARE NAME TOKEN
# =========================================================

for threshold in RARE_THRESHOLDS:

    def rare_name_candidates(
        row,
        threshold=threshold
    ):

        candidates = set()

        s2_rows = rare_name_block(
            row["name_norm"],
            s2_name_token_index,
            s2_name_frequency,
            threshold
        )

        s3_rows = rare_name_block(
            row["name_norm"],
            s3_name_token_index,
            s3_name_frequency,
            threshold
        )

        for idx in s2_rows:
            candidates.add(
                s2_ids[idx]
            )

        for idx in s3_rows:
            candidates.add(
                s3_ids[idx]
            )

        return candidates

    evaluate_blocker(
        f"Rare name token <= {threshold}",
        rare_name_candidates
    )


# =========================================================
# 3. RARE ADDRESS TOKEN
# =========================================================

for threshold in RARE_THRESHOLDS:

    def rare_address_candidates(
        row,
        threshold=threshold
    ):

        candidates = set()

        s2_rows = rare_address_block(
            row["address_norm"],
            s2_address_token_index,
            s2_address_frequency,
            threshold
        )

        s3_rows = rare_address_block(
            row["address_norm"],
            s3_address_token_index,
            s3_address_frequency,
            threshold
        )

        for idx in s2_rows:
            candidates.add(
                s2_ids[idx]
            )

        for idx in s3_rows:
            candidates.add(
                s3_ids[idx]
            )

        return candidates

    evaluate_blocker(
        f"Rare address token <= {threshold}",
        rare_address_candidates
    )


# =========================================================
# 4. ADDRESS NUMBER
# =========================================================

def number_candidates(row):

    candidates = set()

    s2_rows = address_number_block(
        row["address_norm"],
        s2_number_index
    )

    s3_rows = address_number_block(
        row["address_norm"],
        s3_number_index
    )

    for idx in s2_rows:
        candidates.add(
            s2_ids[idx]
        )

    for idx in s3_rows:
        candidates.add(
            s3_ids[idx]
        )

    return candidates


evaluate_blocker(
    "Address number",
    number_candidates
)


# =========================================================
# 5. COMBINED BLOCKING
# =========================================================

def combined_candidates(row):

    candidates = set()

    # Exact name
    candidates.update(
        exact_name_candidates(row)
    )

    # Rare name token
    candidates.update(
        rare_name_candidates(row, 100)
    )

    # Rare address token
    candidates.update(
        rare_address_candidates(row, 100)
    )

    # Address number
    candidates.update(
        number_candidates(row)
    )

    return candidates


evaluate_blocker(
    "Combined blocking",
    combined_candidates
)


# =========================================================
# FINAL RESULTS
# =========================================================

print("\n")
print("=" * 70)
print("BLOCKING BENCHMARK RESULTS")
print("=" * 70)

results_df = pd.DataFrame(results)

print(
    results_df.to_string(
        index=False,
        formatters={
            "recall": "{:.4%}".format,
            "entity_recall": "{:.4%}".format,
            "average_candidates": "{:,.2f}".format,
            "runtime_seconds": "{:.2f}".format
        }
    )
)

print("\nBenchmark complete.")