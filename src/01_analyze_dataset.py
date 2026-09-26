import pandas as pd

# ============================================
# 1. LOAD TRAINING DATA
# ============================================

source1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
)

source2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t"
)

source3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t"
)

ground_truth = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)


# ============================================
# 2. DATASET SIZE
# ============================================

print("\n========== DATASET SIZE ==========")

print("Source 1:", source1.shape)
print("Source 2:", source2.shape)
print("Source 3:", source3.shape)
print("Ground Truth:", ground_truth.shape)


# ============================================
# 3. COLUMN NAMES
# ============================================

print("\n========== COLUMNS ==========")

print("\nSource 1:")
print(source1.columns.tolist())

print("\nSource 2:")
print(source2.columns.tolist())

print("\nSource 3:")
print(source3.columns.tolist())

print("\nGround Truth:")
print(ground_truth.columns.tolist())


# ============================================
# 4. FIRST 5 RECORDS
# ============================================

print("\n========== SOURCE 1 SAMPLE ==========")
print(source1.head())

print("\n========== SOURCE 2 SAMPLE ==========")
print(source2.head())

print("\n========== SOURCE 3 SAMPLE ==========")
print(source3.head())

print("\n========== GROUND TRUTH SAMPLE ==========")
print(ground_truth.head())


# ============================================
# 5. MISSING VALUES
# ============================================

print("\n========== MISSING VALUES ==========")

print("\nSource 1:")
print(source1.isnull().sum())

print("\nSource 2:")
print(source2.isnull().sum())

print("\nSource 3:")
print(source3.isnull().sum())

print("\nGround Truth:")
print(ground_truth.isnull().sum())


# ============================================
# 6. DUPLICATES
# ============================================

print("\n========== DUPLICATES ==========")

print("Source 1 duplicate rows:",
      source1.duplicated().sum())

print("Source 2 duplicate rows:",
      source2.duplicated().sum())

print("Source 3 duplicate rows:",
      source3.duplicated().sum())

print("Ground Truth duplicate rows:",
      ground_truth.duplicated().sum())


# ============================================
# 7. COUNTRIES
# ============================================

print("\n========== COUNTRIES ==========")

print("\nSource 1:")
print(source1["country"].value_counts(dropna=False))

print("\nSource 2:")
print(source2["country"].value_counts(dropna=False))

print("\nSource 3:")
print(source3["country"].value_counts(dropna=False))