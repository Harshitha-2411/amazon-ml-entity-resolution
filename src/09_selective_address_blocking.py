import pandas as pd
import re
import time
from collections import defaultdict, Counter

from blocking import normalize_text, get_tokens, get_numbers


# ============================================================
# SETTINGS
# ============================================================

RARE_TOKEN_LIMIT = 100
SAMPLE_SIZE = 1000


# ============================================================
# LOAD DATA
# ============================================================

print("Loading datasets...")

s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t")
s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t")
gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t")

print("S1:", s1.shape)
print("S2:", s2.shape)
print("S3:", s3.shape)


# ============================================================
# NORMALIZE
# ============================================================

print("\nNormalizing addresses...")

start = time.time()

for df in [s1, s2, s3]:
    df["norm_address"] = df["business_address"].fillna("").map(normalize_text)
    df["norm_name"] = df["business_name"].fillna("").map(normalize_text)

print(f"Normalization time: {time.time() - start:.2f}s")


# ============================================================
# BUILD ADDRESS TOKEN FREQUENCY
# ============================================================

print("\nCalculating address token frequencies...")

start = time.time()

address_freq = Counter()

for df in [s2, s3]:
    for address in df["norm_address"]:
        tokens = set(get_tokens(address))

        for token in tokens:
            # Ignore very short / numeric-only tokens
            if len(token) >= 2 and not token.isdigit():
                address_freq[token] += 1

print(f"Frequency calculation time: {time.time() - start:.2f}s")

rare_tokens = {
    token
    for token, freq in address_freq.items()
    if freq <= RARE_TOKEN_LIMIT
}

print("Total address tokens:", len(address_freq))
print(
    f"Rare address tokens <= {RARE_TOKEN_LIMIT}:",
    len(rare_tokens)
)


# ============================================================
# BUILD LIGHTWEIGHT INDEX
#
# KEY:
# (country, address_number, rare_address_token)
#
# IMPORTANT:
# We ONLY create keys for rare address tokens.
# ============================================================

print("\nBuilding selective address-number index...")

start = time.time()

index = defaultdict(set)

for df in [s2, s3]:

    for row in df.itertuples(index=False):

        country = normalize_text(row.country)
        address = normalize_text(row.business_address)

        numbers = set(get_numbers(address))
        tokens = set(get_tokens(address))

        useful_tokens = [
            token
            for token in tokens
            if token in rare_tokens
        ]

        if not numbers or not useful_tokens:
            continue

        for number in numbers:
            for token in useful_tokens:

                key = (country, number, token)

                index[key].add(row.entity_id)

print(f"Index building time: {time.time() - start:.2f}s")
print("Index keys:", len(index))


# ============================================================
# GROUND TRUTH
# ============================================================

gt_dict = dict(
    zip(
        gt["source1_entity_id"],
        gt["matched_entity_ids"]
    )
)


def get_true_matches(s1_id):

    value = gt_dict.get(s1_id, "")

    if pd.isna(value) or value == "":
        return set()

    return set(
        x.strip()
        for x in str(value).split(",")
        if x.strip()
    )


# ============================================================
# TEST SAMPLE
# ============================================================

matched_s1 = [
    x
    for x in s1["entity_id"]
    if len(get_true_matches(x)) > 0
]

test_ids = matched_s1[:SAMPLE_SIZE]

s1_lookup = s1.set_index("entity_id")


# ============================================================
# EVALUATION
# ============================================================

print("\nEvaluating selective address-number blocking...")

start = time.time()

total_true = 0
total_retrieved = 0
entities_with_all_matches = 0

candidate_counts = []

for s1_id in test_ids:

    row = s1_lookup.loc[s1_id]

    country = normalize_text(row["country"])
    address = normalize_text(row["business_address"])

    numbers = set(get_numbers(address))
    tokens = set(get_tokens(address))

    candidates = set()

    useful_tokens = [
        token
        for token in tokens
        if token in rare_tokens
    ]

    for number in numbers:
        for token in useful_tokens:

            key = (country, number, token)

            if key in index:
                candidates.update(index[key])

    true_matches = get_true_matches(s1_id)

    retrieved_true = candidates.intersection(true_matches)

    total_true += len(true_matches)
    total_retrieved += len(retrieved_true)

    if true_matches.issubset(candidates):
        entities_with_all_matches += 1

    candidate_counts.append(len(candidates))


# ============================================================
# RESULTS
# ============================================================

runtime = time.time() - start

recall = (
    total_retrieved / total_true
    if total_true > 0
    else 0
)

entity_recall = (
    entities_with_all_matches / len(test_ids)
    if test_ids
    else 0
)

avg_candidates = (
    sum(candidate_counts) / len(candidate_counts)
    if candidate_counts
    else 0
)

max_candidates = max(candidate_counts) if candidate_counts else 0


print("\n" + "=" * 60)
print("SELECTIVE ADDRESS-NUMBER BLOCKING RESULTS")
print("=" * 60)

print(f"Sample S1 entities       : {len(test_ids)}")
print(f"Rare token threshold     : {RARE_TOKEN_LIMIT}")

print(f"\nTrue matches             : {total_true}")
print(f"Retrieved true matches   : {total_retrieved}")

print(f"\nMatch recall             : {recall:.4%}")
print(f"Entity recall            : {entity_recall:.4%}")

print(f"\nAverage candidates       : {avg_candidates:.2f}")
print(f"Maximum candidates       : {max_candidates}")

print(f"\nEvaluation runtime       : {runtime:.2f}s")

print("=" * 60)