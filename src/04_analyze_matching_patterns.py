import pandas as pd


# Load training data
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

# Replace missing values
for df in [s1, s2, s3]:
    df["business_name"] = df["business_name"].fillna("")
    df["business_address"] = df["business_address"].fillna("")

gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")


# --------------------------------------------------
# BASIC COUNTRY ANALYSIS
# --------------------------------------------------

print("\n========== COUNTRY DISTRIBUTION ==========")

print("\nSource 1:")
print(s1["country"].value_counts())

print("\nSource 2:")
print(s2["country"].value_counts())

print("\nSource 3:")
print(s3["country"].value_counts())


# --------------------------------------------------
# MATCH EXAMPLES
# --------------------------------------------------

print("\n========== MATCH EXAMPLES ==========")

# Pick Source 1 records that have matches
matched_gt = gt[gt["matched_entity_ids"].str.strip() != ""]

for _, row in matched_gt.head(10).iterrows():

    s1_id = row["source1_entity_id"]

    match_ids = [
        x.strip()
        for x in row["matched_entity_ids"].split(",")
    ]

    source1_record = s1[
        s1["entity_id"] == s1_id
    ]

    print("\n----------------------------------------")

    print("SOURCE 1")
    print(source1_record[
        ["entity_id", "business_name", "business_address", "country"]
    ].to_string(index=False))

    print("\nMATCHES")

    matches2 = s2[
        s2["entity_id"].isin(match_ids)
    ]

    matches3 = s3[
        s3["entity_id"].isin(match_ids)
    ]

    if len(matches2) > 0:
        print("\nSource 2:")
        print(matches2[
            ["entity_id", "business_name", "business_address", "country"]
        ].to_string(index=False))

    if len(matches3) > 0:
        print("\nSource 3:")
        print(matches3[
            ["entity_id", "business_name", "business_address", "country"]
        ].to_string(index=False))