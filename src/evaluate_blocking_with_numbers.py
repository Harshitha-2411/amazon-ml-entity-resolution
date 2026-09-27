import pandas as pd
import re
import unicodedata
from collections import Counter, defaultdict
import time


S1_FILE = "dataset/train/train_source1.tsv"
S2_FILE = "dataset/train/train_source2.tsv"
S3_FILE = "dataset/train/train_source3.tsv"
GT_FILE = "dataset/train/train_ground_truth.tsv"


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = text.replace("&", " and ")

    text = "".join(
        " " if unicodedata.category(ch).startswith("P") else ch
        for ch in text
    )

    return re.sub(r"\s+", " ", text).strip()


def get_tokens(text):
    if not text:
        return set()
    return set(text.split())


def get_numbers(text):
    if not text:
        return set()

    return set(re.findall(r"\d+", text))


# ============================================================
# LOAD
# ============================================================

print("Loading training datasets...")

s1 = pd.read_csv(S1_FILE, sep="\t", dtype=str)
s2 = pd.read_csv(S2_FILE, sep="\t", dtype=str)
s3 = pd.read_csv(S3_FILE, sep="\t", dtype=str)
gt = pd.read_csv(GT_FILE, sep="\t", dtype=str)

print("Source 1:", len(s1))
print("Source 2:", len(s2))
print("Source 3:", len(s3))
print("Ground truth:", len(gt))


# ============================================================
# NORMALIZE
# ============================================================

print("\nNormalizing...")

for df in [s1, s2, s3]:
    df["norm_name"] = df["business_name"].fillna("").map(normalize_text)
    df["norm_address"] = df["business_address"].fillna("").map(normalize_text)


# ============================================================
# COMBINE TARGET SOURCES
# ============================================================

print("Combining Source 2 + Source 3...")

target = pd.concat(
    [
        s2[["entity_id", "country", "norm_name", "norm_address"]],
        s3[["entity_id", "country", "norm_name", "norm_address"]]
    ],
    ignore_index=True
)


# ============================================================
# EXACT NAME INDEX
# ============================================================

print("Building exact-name index...")

exact_name_index = defaultdict(set)

for row in target.itertuples(index=False):

    if row.norm_name:
        key = (row.country, row.norm_name)
        exact_name_index[key].add(row.entity_id)


# ============================================================
# RARE NAME TOKEN INDEX
# ============================================================

print("Calculating name token frequencies...")

name_frequency = Counter()

for name in target["norm_name"]:
    for token in get_tokens(name):
        name_frequency[token] += 1


print("Building rare-name index...")

rare_name_index = defaultdict(set)

for row in target.itertuples(index=False):

    for token in get_tokens(row.norm_name):

        if name_frequency[token] <= 100:
            key = (row.country, token)
            rare_name_index[key].add(row.entity_id)


# ============================================================
# RARE ADDRESS TOKEN INDEX
# ============================================================

print("Calculating address token frequencies...")

address_frequency = Counter()

for address in target["norm_address"]:
    for token in get_tokens(address):
        address_frequency[token] += 1


print("Building rare-address index...")

rare_address_index = defaultdict(set)

for row in target.itertuples(index=False):

    for token in get_tokens(row.norm_address):

        if address_frequency[token] <= 100:
            key = (row.country, token)
            rare_address_index[key].add(row.entity_id)


# ============================================================
# ADDRESS NUMBER INDEX
# ============================================================

print("Building address-number index...")

number_index = defaultdict(set)

for row in target.itertuples(index=False):

    numbers = get_numbers(row.norm_address)

    for number in numbers:

        key = (row.country, number)

        number_index[key].add(row.entity_id)


# ============================================================
# GROUND TRUTH
# ============================================================

print("Preparing ground truth...")

gt_dict = {}

for row in gt.itertuples(index=False):

    matches = row.matched_entity_ids

    if pd.isna(matches) or not matches:

        gt_dict[row.source1_entity_id] = set()

    else:

        gt_dict[row.source1_entity_id] = set(
            x.strip()
            for x in str(matches).split(",")
            if x.strip()
        )


# ============================================================
# EVALUATION
# ============================================================

print("\nGenerating candidates with address-number blocking...")

start = time.time()

total_true_matches = 0
total_retrieved_matches = 0

entity_recall_count = 0
evaluated_entities = 0

candidate_count = 0


for i, row in enumerate(s1.itertuples(index=False), start=1):

    candidates = set()

    country = row.country

    # --------------------------------------------------------
    # 1. EXACT NAME
    # --------------------------------------------------------

    if row.norm_name:

        key = (country, row.norm_name)

        candidates.update(
            exact_name_index.get(key, set())
        )

    # --------------------------------------------------------
    # 2. RARE NAME TOKEN
    # --------------------------------------------------------

    for token in get_tokens(row.norm_name):

        if name_frequency[token] <= 100:

            key = (country, token)

            candidates.update(
                rare_name_index.get(key, set())
            )

    # --------------------------------------------------------
    # 3. RARE ADDRESS TOKEN
    # --------------------------------------------------------

    for token in get_tokens(row.norm_address):

        if address_frequency[token] <= 100:

            key = (country, token)

            candidates.update(
                rare_address_index.get(key, set())
            )

    # --------------------------------------------------------
    # 4. ADDRESS NUMBER
    # --------------------------------------------------------

    for number in get_numbers(row.norm_address):

        key = (country, number)

        candidates.update(
            number_index.get(key, set())
        )

    # --------------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------------

    true_matches = gt_dict.get(row.entity_id, set())

    if true_matches:

        evaluated_entities += 1

        total_true_matches += len(true_matches)

        retrieved = true_matches.intersection(candidates)

        total_retrieved_matches += len(retrieved)

        if retrieved:
            entity_recall_count += 1

    candidate_count += len(candidates)

    if i % 100000 == 0:

        print(
            f"Processed {i:,} S1 records | "
            f"Candidates: {candidate_count:,}"
        )


# ============================================================
# RESULTS
# ============================================================

runtime = time.time() - start

pair_recall = (
    total_retrieved_matches / total_true_matches
    if total_true_matches
    else 0
)

entity_recall = (
    entity_recall_count / evaluated_entities
    if evaluated_entities
    else 0
)

avg_candidates = candidate_count / len(s1)


print("\n")
print("=" * 65)
print("BLOCKING RESULTS WITH ADDRESS-NUMBER BLOCKING")
print("=" * 65)

print(f"Training S1 records       : {len(s1):,}")
print(f"S1 entities with matches  : {evaluated_entities:,}")
print(f"True match links          : {total_true_matches:,}")
print(f"Retrieved true links      : {total_retrieved_matches:,}")

print()
print(f"PAIR RECALL                : {pair_recall:.4%}")
print(f"ENTITY RECALL              : {entity_recall:.4%}")

print()
print(f"Average candidates/S1      : {avg_candidates:,.2f}")
print(f"Runtime                    : {runtime:.2f} seconds")

print("=" * 65)