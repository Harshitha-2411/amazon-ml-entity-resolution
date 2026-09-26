# Business Entity Resolution — Candidate Generation

## Project Overview

This project implements the candidate-generation/blocking stage for the Amazon Business Entity Resolution Challenge.

The goal of this stage is to generate a set of plausible Source 2 and Source 3 candidates for every Source 1 entity before the final matching model performs inference.

## Candidate Generation Strategy

The blocking pipeline currently uses:

1. Exact normalized business-name matching
2. Rare business-name token matching
3. Rare business-address token matching

Country is included as part of the blocking key so that records from different countries are not mixed.

Text normalization handles:

* Case differences
* Unicode normalization
* Punctuation
* `&` versus `and`
* Whitespace differences

No external business databases, APIs, geocoding services, or translation services are used.

## Input Files

The test data should be placed in:

```text
dataset/test/
```

Required files:

```text
test_source1.tsv
test_source2.tsv
test_source3.tsv
```

## Generate Candidate Pairs

Run the following command from the project root:

```bash
python src/generate_candidates.py
```

The generated file will be:

```text
output/candidate_pairs.tsv
```

## Output Format

`candidate_pairs.tsv` contains:

```text
source1_entity_id    candidate_entity_ids
```

Each Source 1 entity has exactly one row.

Candidate IDs contain only Source 2 (`S2-`) and Source 3 (`S3-`) entity IDs.

## Benchmarking

The blocking benchmark can be run with:

```bash
python src/08_blocking_benchmark.py
```

The benchmark is used to measure candidate-generation recall, candidate-set size, and runtime on the training data.

## Scope

This code covers only:

* Candidate generation
* Blocking
* Blocking evaluation

The final matching model, probability generation, thresholding, and `matching_results.tsv` generation are handled separately.
