import os
import re
import pandas as pd
from collections import Counter


# =========================================================
# CONFIGURATION
# =========================================================

DATASET_DIR = "dataset/train"

S2_FILE = os.path.join(DATASET_DIR, "train_source2.tsv")
S3_FILE = os.path.join(DATASET_DIR, "train_source3.tsv")

SAMPLE_SIZE = 100_000


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Preserve multilingual scripts
    text = text.casefold()

    # Convert punctuation to spaces
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================================================
# TOKEN EXTRACTION
# =========================================================

def get_tokens(text):
    if not text:
        return []

    return text.split()


# =========================================================
# ANALYZE ONE SOURCE
# =========================================================

def analyze_source(file_path, source_name):

    print("\n" + "=" * 70)
    print(f"ANALYZING {source_name}")
    print("=" * 70)

    df = pd.read_csv(
        file_path,
        sep="\t",
        usecols=[
            "business_name",
            "business_address"
        ],
        nrows=SAMPLE_SIZE
    )

    print(f"Rows analyzed: {len(df):,}")

    name_counter = Counter()
    address_counter = Counter()

    # -----------------------------------------------------
    # Name tokens
    # -----------------------------------------------------

    for value in df["business_name"].fillna(""):

        normalized = normalize_text(value)

        for token in set(get_tokens(normalized)):
            name_counter[token] += 1

    # -----------------------------------------------------
    # Address tokens
    # -----------------------------------------------------

    for value in df["business_address"].fillna(""):

        normalized = normalize_text(value)

        for token in set(get_tokens(normalized)):
            address_counter[token] += 1

    # -----------------------------------------------------
    # Display common name tokens
    # -----------------------------------------------------

    print("\nTOP 30 NAME TOKENS")
    print("-" * 50)

    for token, count in name_counter.most_common(30):
        print(f"{token:<30} {count:>8,}")

    # -----------------------------------------------------
    # Display common address tokens
    # -----------------------------------------------------

    print("\nTOP 30 ADDRESS TOKENS")
    print("-" * 50)

    for token, count in address_counter.most_common(30):
        print(f"{token:<30} {count:>8,}")

    # -----------------------------------------------------
    # Frequency distribution
    # -----------------------------------------------------

    print("\nNAME TOKEN FREQUENCY")

    frequency_groups = {
        "1 occurrence": 0,
        "2-5 occurrences": 0,
        "6-10 occurrences": 0,
        "11-50 occurrences": 0,
        "51-100 occurrences": 0,
        "101-500 occurrences": 0,
        "501-1000 occurrences": 0,
        "1000+ occurrences": 0
    }

    for count in name_counter.values():

        if count == 1:
            frequency_groups["1 occurrence"] += 1
        elif count <= 5:
            frequency_groups["2-5 occurrences"] += 1
        elif count <= 10:
            frequency_groups["6-10 occurrences"] += 1
        elif count <= 50:
            frequency_groups["11-50 occurrences"] += 1
        elif count <= 100:
            frequency_groups["51-100 occurrences"] += 1
        elif count <= 500:
            frequency_groups["101-500 occurrences"] += 1
        elif count <= 1000:
            frequency_groups["501-1000 occurrences"] += 1
        else:
            frequency_groups["1000+ occurrences"] += 1

    for group, count in frequency_groups.items():
        print(f"{group:<25} {count:>10,}")

    return name_counter, address_counter


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("BLOCKING TOKEN FREQUENCY ANALYSIS")
    print("=" * 70)

    s2_name_counter, s2_address_counter = analyze_source(
        S2_FILE,
        "SOURCE 2"
    )

    s3_name_counter, s3_address_counter = analyze_source(
        S3_FILE,
        "SOURCE 3"
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)