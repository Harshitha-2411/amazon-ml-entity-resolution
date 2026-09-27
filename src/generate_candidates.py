import os
import sys
import time
from collections import defaultdict, Counter

import pandas as pd

# Allow importing blocking.py from the same src folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from blocking import normalize_text


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

TEST_DIR = "dataset/test"
OUTPUT_DIR = "output"

SOURCE1_FILE = os.path.join(TEST_DIR, "test_source1.tsv")
SOURCE2_FILE = os.path.join(TEST_DIR, "test_source2.tsv")
SOURCE3_FILE = os.path.join(TEST_DIR, "test_source3.tsv")

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def get_tokens(text):
    """Return normalized word tokens."""
    text = normalize_text(text)

    if not text:
        return set()

    return set(text.split())


def build_name_index(source2, source3):
    """
    Build an index:

    (country, normalized_name) -> candidate entity IDs
    """

    index = defaultdict(set)

    for row in pd.concat([source2, source3], ignore_index=True).itertuples(index=False):
        name = normalize_text(row.business_name)
        country = normalize_text(row.country)

        if name:
            index[(country, name)].add(row.entity_id)

    return index


def build_name_token_index(source2, source3):
    """
    Build:

    (country, token) -> entity IDs

    Only relatively rare tokens are used for blocking.
    """

    records = pd.concat([source2, source3], ignore_index=True)

    token_frequency = Counter()

    for row in records.itertuples(index=False):
        country = normalize_text(row.country)
        tokens = get_tokens(row.business_name)

        for token in tokens:
            token_frequency[(country, token)] += 1

    index = defaultdict(set)

    MAX_FREQUENCY = 100

    for row in records.itertuples(index=False):
        country = normalize_text(row.country)
        tokens = get_tokens(row.business_name)

        for token in tokens:
            if token_frequency[(country, token)] <= MAX_FREQUENCY:
                index[(country, token)].add(row.entity_id)

    return index


def build_address_token_index(source2, source3):
    """
    Build:

    (country, address_token) -> entity IDs

    Only relatively rare address tokens are used.
    """

    records = pd.concat([source2, source3], ignore_index=True)

    token_frequency = Counter()

    for row in records.itertuples(index=False):
        country = normalize_text(row.country)
        tokens = get_tokens(row.business_address)

        for token in tokens:
            token_frequency[(country, token)] += 1

    index = defaultdict(set)

    MAX_FREQUENCY = 100

    for row in records.itertuples(index=False):
        country = normalize_text(row.country)
        tokens = get_tokens(row.business_address)

        for token in tokens:
            if token_frequency[(country, token)] <= MAX_FREQUENCY:
                index[(country, token)].add(row.entity_id)

    return index


# ---------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------

def generate_candidates(source1, source2, source3):

    print("Building exact-name index...")
    name_index = build_name_index(source2, source3)

    print("Building rare-name-token index...")
    name_token_index = build_name_token_index(source2, source3)

    print("Building rare-address-token index...")
    address_token_index = build_address_token_index(source2, source3)

    results = []

    total_candidates = 0

    print("Generating candidates...")

    for counter, row in enumerate(source1.itertuples(index=False), start=1):

        s1_id = row.entity_id

        country = normalize_text(row.country)

        name = normalize_text(row.business_name)

        candidates = set()

        # -------------------------------------------------
        # Block 1: Exact normalized name + country
        # -------------------------------------------------

        if name:
            candidates.update(
                name_index.get((country, name), set())
            )

        # -------------------------------------------------
        # Block 2: Rare business-name tokens
        # -------------------------------------------------

        name_tokens = get_tokens(row.business_name)

        for token in name_tokens:
            candidates.update(
                name_token_index.get((country, token), set())
            )

        # -------------------------------------------------
        # Block 3: Rare address tokens
        # -------------------------------------------------

        address_tokens = get_tokens(row.business_address)

        for token in address_tokens:
            candidates.update(
                address_token_index.get((country, token), set())
            )

        # Remove duplicates and make ordering stable
        candidates = sorted(candidates)

        total_candidates += len(candidates)

        results.append(
            {
                "source1_entity_id": s1_id,
                "candidate_entity_ids": ",".join(candidates)
            }
        )

        if counter % 100000 == 0:
            print(
                f"Processed {counter:,} Source 1 records | "
                f"Candidates so far: {total_candidates:,}"
            )

    return pd.DataFrame(results)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    start_time = time.perf_counter()

    print("Loading test datasets...")

    source1 = pd.read_csv(
        SOURCE1_FILE,
        sep="\t"
    )

    source2 = pd.read_csv(
        SOURCE2_FILE,
        sep="\t"
    )

    source3 = pd.read_csv(
        SOURCE3_FILE,
        sep="\t"
    )

    print(f"Source 1 records: {len(source1):,}")
    print(f"Source 2 records: {len(source2):,}")
    print(f"Source 3 records: {len(source3):,}")

    candidates = generate_candidates(
        source1,
        source2,
        source3
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    candidates.to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False
    )

    elapsed = time.perf_counter() - start_time

    total_candidate_links = (
        candidates["candidate_entity_ids"]
        .str.split(",")
        .apply(
            lambda x: 0
            if x == [""] else len(x)
        )
        .sum()
    )

    average_candidates = (
        total_candidate_links / len(source1)
    )

    print("\n" + "=" * 60)
    print("CANDIDATE GENERATION COMPLETE")
    print("=" * 60)

    print(f"Source 1 records       : {len(source1):,}")
    print(f"Source 2 records       : {len(source2):,}")
    print(f"Source 3 records       : {len(source3):,}")
    print(f"Candidate links        : {total_candidate_links:,}")
    print(f"Average candidates/S1  : {average_candidates:.2f}")
    print(f"Runtime                : {elapsed:.2f} seconds")
    print(f"Output                 : {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()