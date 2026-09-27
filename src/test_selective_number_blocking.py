import pandas as pd
import re
import time
from collections import defaultdict

from blocking import normalize_text, get_tokens, get_numbers


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_SIZE = 10_000
RARE_TOKEN_MAX_FREQ = 100

S1_PATH = "dataset/train/train_source1.tsv"
S2_PATH = "dataset/train/train_source2.tsv"
S3_PATH = "dataset/train/train_source3.tsv"
GT_PATH = "dataset/train/train_ground_truth.tsv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading datasets...")

start_total = time.time()

s1 = pd.read_csv(S1_PATH, sep="\t", dtype=str).head(SAMPLE_SIZE)
s2 = pd.read_csv(S2_PATH, sep="\t", dtype=str)
s3 = pd.read_csv(S3_PATH, sep="\t", dtype=str)
gt = pd.read_csv(GT_PATH, sep="\t", dtype=str)

print(f"S1 sample: {len(s1):,}")
print(f"S2: {len(s2):,}")
print(f"S3: {len(s3):,}")


# ============================================================
# COMBINE SOURCE 2 + SOURCE 3
# ============================================================

targets = pd.concat([s2, s3], ignore_index=True)

print(f"Combined target records: {len(targets):,}")


# ============================================================
# NORMALIZE NAME + ADDRESS
# ============================================================

print("\nNormalizing target data...")

targets["name_norm"] = targets["business_name"].fillna("").map(normalize_text)
targets["address_norm"] = targets["business_address"].fillna("").map(normalize_text)
targets["country_norm"] = targets["country"].fillna("").map(normalize_text)

print("Normalization complete.")


# ============================================================
# STEP 1: CALCULATE NAME TOKEN FREQUENCIES
# ============================================================

print("\nCalculating name-token frequencies...")

token_frequency = defaultdict(int)

for name in targets["name_norm"]:
    tokens = set(get_tokens(name))

    for token in tokens:
        if token:
            token_frequency[token] += 1

print(f"Unique name tokens: {len(token_frequency):,}")


# ============================================================
# STEP 2: BUILD SELECTIVE INDEX
#
# KEY:
#     (country, address_number, rare_name_token)
#
# Only name tokens with frequency <= 100 are used.
# ============================================================

print("\nBuilding selective number + name-token index...")

index = defaultdict(list)

records_indexed = 0
keys_created = 0

for row in targets.itertuples(index=False):

    country = row.country_norm
    name = row.name_norm
    address = row.address_norm
    entity_id = row.entity_id

    if not country or not entity_id:
        continue

    name_tokens = set(get_tokens(name))

    # Keep only rare name tokens
    rare_tokens = [
        token
        for token in name_tokens
        if token_frequency.get(token, 0) <= RARE_TOKEN_MAX_FREQ
    ]

    if not rare_tokens:
        continue

    numbers = set(get_numbers(address))

    if not numbers:
        continue

    records_indexed += 1

    for number in numbers:
        for token in rare_tokens:

            key = (country, number, token)

            index[key].append(entity_id)
            keys_created += 1


print(f"Records indexed: {records_indexed:,}")
print(f"Index keys created: {keys_created:,}")
print(f"Unique index keys: {len(index):,}")


# ============================================================
# STEP 3: BUILD GROUND TRUTH
# ============================================================

print("\nPreparing ground truth...")

gt_sample = gt[gt["source1_entity_id"].isin(s1["entity_id"])]

ground_truth = {}

true_match_links = 0
s1_with_matches = 0

for row in gt_sample.itertuples(index=False):

    source1_id = row.source1_entity_id
    value = row.matched_entity_ids

    if pd.isna(value) or not str(value).strip():
        ground_truth[source1_id] = set()
        continue

    matches = {
        x.strip()
        for x in str(value).split(",")
        if x.strip()
    }

    ground_truth[source1_id] = matches

    if matches:
        s1_with_matches += 1
        true_match_links += len(matches)


# ============================================================
# STEP 4: GENERATE CANDIDATES
# ============================================================

print("\nGenerating candidates...")

total_retrieved_true = 0
total_candidates = 0

entity_recall_count = 0

candidate_counts = []

for i, row in enumerate(s1.itertuples(index=False), start=1):

    country = normalize_text(row.country)
    name = normalize_text(row.business_name)
    address = normalize_text(row.business_address)

    candidates = set()

    name_tokens = set(get_tokens(name))

    rare_tokens = [
        token
        for token in name_tokens
        if token_frequency.get(token, 0) <= RARE_TOKEN_MAX_FREQ
    ]

    numbers = set(get_numbers(address))

    # --------------------------------------------------------
    # SELECTIVE BLOCK:
    # country + address number + rare name token
    # --------------------------------------------------------

    for number in numbers:
        for token in rare_tokens:

            key = (country, number, token)

            for entity_id in index.get(key, []):
                candidates.add(entity_id)

    candidate_count = len(candidates)

    candidate_counts.append(candidate_count)
    total_candidates += candidate_count

    # --------------------------------------------------------
    # RECALL EVALUATION
    # --------------------------------------------------------

    true_matches = ground_truth.get(row.entity_id, set())

    retrieved = true_matches.intersection(candidates)

    total_retrieved_true += len(retrieved)

    if true_matches and retrieved:
        entity_recall_count += 1

    if i % 1000 == 0:
        print(
            f"Processed {i:,}/{len(s1):,} | "
            f"Candidates: {total_candidates:,}"
        )


# ============================================================
# STEP 5: CALCULATE METRICS
# ============================================================

pair_recall = (
    total_retrieved_true / true_match_links
    if true_match_links > 0
    else 0
)

entity_recall = (
    entity_recall_count / s1_with_matches
    if s1_with_matches > 0
    else 0
)

avg_candidates = (
    total_candidates / len(s1)
    if len(s1) > 0
    else 0
)

max_candidates = max(candidate_counts) if candidate_counts else 0

runtime = time.time() - start_total


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("SELECTIVE NUMBER + RARE NAME TOKEN RESULTS")
print("=" * 60)

print(f"S1 sample size             : {len(s1):,}")
print(f"S1 with true matches       : {s1_with_matches:,}")
print(f"True match links           : {true_match_links:,}")
print(f"Retrieved true links       : {total_retrieved_true:,}")

print(f"\nPair recall                : {pair_recall * 100:.4f}%")
print(f"Entity recall              : {entity_recall * 100:.4f}%")

print(f"\nAverage candidates / S1    : {avg_candidates:,.2f}")
print(f"Maximum candidates         : {max_candidates:,}")

print(f"\nRuntime                    : {runtime:.2f} seconds")
print(f"Runtime                    : {runtime / 60:.2f} minutes")

print("=" * 60)