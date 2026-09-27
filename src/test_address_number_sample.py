import pandas as pd
import re
import unicodedata
from collections import Counter, defaultdict
import time


# ============================================================
# CONFIG
# ============================================================

SAMPLE_SIZE = 10000

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

print("Loading training data...")

s1 = pd.read_csv(S1_FILE, sep="\t", dtype=str)
s2 = pd.read_csv(S2_FILE, sep="\t", dtype=str)
s3 = pd.read_csv(S3_FILE, sep="\t", dtype=str)
gt = pd.read_csv(GT_FILE, sep="\t", dtype=str)

# Only evaluate first 10,000 S1 records
s1 = s1.iloc[:SAMPLE_SIZE].copy()

print("S1 sample:", len(s1))
print("S2:", len(s2))
print("S3:", len(s3))


# ============================================================
# NORMALIZE
# ============================================================

print("\nNormalizing S1/S2/S3...")

for df in [s1, s2, s3]:

    df["norm_name"] = (
        df["business_name"]
        .fillna("")
        .map(normalize_text)
    )

    df["norm_address"] = (
        df["business_address"]
        .fillna("")
        .map(normalize_text)
    )


# ============================================================
# COMBINE S2 + S3
# ============================================================

print("Combining S2 + S3...")

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

        exact_name_index[
            (row.country, row.norm_name)
        ].add(row.entity_id)


# ============================================================
# NAME FREQUENCY
# ============================================================

print("Calculating name frequencies...")

name_frequency = Counter()

for name in target["norm_name"]:

    for token in get_tokens(name):
        name_frequency[token] += 1


# ============================================================
# RARE NAME INDEX
# ============================================================

print("Building rare-name index...")

rare_name_index = defaultdict(set)

for row in target.itertuples(index=False):

    for token in get_tokens(row.norm_name):

        if name_frequency[token] <= 100:

            rare_name_index[
                (row.country, token)
            ].add(row.entity_id)


# ============================================================
# ADDRESS FREQUENCY
# ============================================================

print("Calculating address frequencies...")

address_frequency = Counter()

for address in target["norm_address"]:

    for token in get_tokens(address):
        address_frequency[token] += 1


# ============================================================
# RARE ADDRESS INDEX
# ============================================================

print("Building rare-address index...")

rare_address_index = defaultdict(set)

for row in target.itertuples(index=False):

    for token in get_tokens(row.norm_address):

        if address_frequency[token] <= 100:

            rare_address_index[
                (row.country, token)
            ].add(row.entity_id)


# ============================================================
# ADDRESS NUMBER INDEX
# ============================================================

print("Building address-number index...")

number_index = defaultdict(set)

for row in target.itertuples(index=False):

    for number in get_numbers(row.norm_address):

        number_index[
            (row.country, number)
        ].add(row.entity_id)


print("Indexes ready.")


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

print("\nEvaluating address-number blocking...")

start = time.time()

total_true = 0
total_found = 0
entities_with_match = 0
entities_with_found_match = 0
total_candidates = 0


for i, row in enumerate(
    s1.itertuples(index=False),
    start=1
):

    candidates = set()

    country = row.country

    # --------------------------------------------------------
    # 1. Exact name
    # --------------------------------------------------------

    if row.norm_name:

        candidates.update(
            exact_name_index.get(
                (country, row.norm_name),
                set()
            )
        )

    # --------------------------------------------------------
    # 2. Rare name token
    # --------------------------------------------------------

    for token in get_tokens(row.norm_name):

        if name_frequency[token] <= 100:

            candidates.update(
                rare_name_index.get(
                    (country, token),
                    set()
                )
            )

    # --------------------------------------------------------
    # 3. Rare address token
    # --------------------------------------------------------

    for token in get_tokens(row.norm_address):

        if address_frequency[token] <= 100:

            candidates.update(
                rare_address_index.get(
                    (country, token),
                    set()
                )
            )

    # --------------------------------------------------------
    # 4. ADDRESS NUMBER
    # --------------------------------------------------------

    for number in get_numbers(row.norm_address):

        candidates.update(
            number_index.get(
                (country, number),
                set()
            )
        )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    true_matches = gt_dict.get(
        row.entity_id,
        set()
    )

    if true_matches:

        entities_with_match += 1

        found = true_matches.intersection(candidates)

        total_true += len(true_matches)
        total_found += len(found)

        if found:
            entities_with_found_match += 1

    total_candidates += len(candidates)

    if i % 1000 == 0:

        print(
            f"Processed {i:,} / {SAMPLE_SIZE:,}"
        )


# ============================================================
# RESULTS
# ============================================================

runtime = time.time() - start

pair_recall = (
    total_found / total_true
    if total_true
    else 0
)

entity_recall = (
    entities_with_found_match / entities_with_match
    if entities_with_match
    else 0
)

avg_candidates = total_candidates / len(s1)


print("\n")
print("=" * 60)
print("10,000-S1 ADDRESS-NUMBER EXPERIMENT")
print("=" * 60)

print(f"S1 sample                 : {len(s1):,}")
print(f"S1 with true matches      : {entities_with_match:,}")
print(f"True match links          : {total_true:,}")
print(f"Retrieved true links      : {total_found:,}")

print()
print(f"PAIR RECALL               : {pair_recall:.4%}")
print(f"ENTITY RECALL             : {entity_recall:.4%}")

print()
print(f"Average candidates/S1     : {avg_candidates:,.2f}")
print(f"Runtime                   : {runtime:.2f} seconds")

print("=" * 60)