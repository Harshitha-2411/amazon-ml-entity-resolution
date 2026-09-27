import pandas as pd

FILE = "output/candidate_pairs.tsv"

print("Loading candidate file...")

df = pd.read_csv(FILE, sep="\t", dtype=str)

print("\n========== BASIC CHECK ==========")

print("Rows:", len(df))
print("Columns:", list(df.columns))

# Expected columns
expected = ["source1_entity_id", "candidate_entity_ids"]

if list(df.columns) != expected:
    print("ERROR: Incorrect columns")
else:
    print("Columns: OK")


print("\n========== S1 ID CHECK ==========")

duplicate_s1 = df["source1_entity_id"].duplicated().sum()

print("Duplicate S1 IDs:", duplicate_s1)

if duplicate_s1 == 0:
    print("S1 uniqueness: OK")
else:
    print("ERROR: Duplicate S1 IDs found")


print("\n========== CANDIDATE ID CHECK ==========")

invalid_ids = 0
duplicate_candidates = 0
total_candidates = 0

for candidates in df["candidate_entity_ids"].fillna(""):
    if not candidates:
        continue

    ids = candidates.split(",")

    total_candidates += len(ids)

    # Check duplicates inside a row
    if len(ids) != len(set(ids)):
        duplicate_candidates += 1

    # Check S2/S3 format
    for entity_id in ids:
        if not (entity_id.startswith("S2-") or
                entity_id.startswith("S3-")):
            invalid_ids += 1

print("Total candidate links:", total_candidates)
print("Rows with duplicate candidates:", duplicate_candidates)
print("Invalid candidate IDs:", invalid_ids)

if duplicate_candidates == 0:
    print("Duplicate candidates: OK")

if invalid_ids == 0:
    print("Candidate ID format: OK")


print("\n========== EMPTY CANDIDATES ==========")

empty = df["candidate_entity_ids"].fillna("").eq("").sum()

print("S1 records with no candidates:", empty)


print("\n========== AVERAGE ==========")

print(
    "Average candidates/S1:",
    total_candidates / len(df)
)


print("\n========== VALIDATION COMPLETE ==========")