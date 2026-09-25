import pandas as pd


def parse_matches(value):
    """Convert comma-separated entity IDs into a set."""
    if pd.isna(value):
        return set()

    value = str(value).strip()

    if not value:
        return set()

    return {
        x.strip()
        for x in value.split(",")
        if x.strip()
    }


def precision_recall_f05(predicted, actual):
    """Calculate precision, recall and F0.5 for one entity."""

    predicted = set(predicted)
    actual = set(actual)

    # Correct no-match prediction
    if not predicted and not actual:
        return 1.0, 1.0, 1.0

    # No prediction, but matches exist
    if not predicted:
        return 0.0, 0.0, 0.0

    true_positive = len(predicted & actual)

    precision = true_positive / len(predicted)

    recall = (
        true_positive / len(actual)
        if actual
        else 0.0
    )

    beta = 0.5

    if precision == 0 and recall == 0:
        f05 = 0.0
    else:
        f05 = (
            (1 + beta**2)
            * precision
            * recall
            / ((beta**2 * precision) + recall)
        )

    return precision, recall, f05


def evaluate_predictions(
    ground_truth,
    predictions,
    source1_column="source1_entity_id",
    matches_column="matched_entity_ids",
):
    """
    Efficient evaluation of predictions.

    Returns:
        result_df:
            Per-Source-1 precision, recall and F0.5.

        metrics:
            Macro precision, recall and F0.5.
    """

    # Keep only required columns
    gt = ground_truth[
        [source1_column, matches_column]
    ].copy()

    pred = predictions[
        [source1_column, matches_column]
    ].copy()

    # Parse match lists
    gt["actual_set"] = gt[matches_column].map(parse_matches)
    pred["predicted_set"] = pred[matches_column].map(parse_matches)

    # Join predictions onto ground truth
    merged = gt[
        [source1_column, "actual_set"]
    ].merge(
        pred[
            [source1_column, "predicted_set"]
        ],
        on=source1_column,
        how="left",
    )

    # Missing predictions = empty set
    merged["predicted_set"] = merged[
        "predicted_set"
    ].apply(
        lambda x: x if isinstance(x, set) else set()
    )

    # Calculate counts using vectorized list comprehension
    merged["true_positive"] = [
        len(pred & actual)
        for pred, actual
        in zip(
            merged["predicted_set"],
            merged["actual_set"],
        )
    ]

    merged["predicted_count"] = [
        len(x)
        for x in merged["predicted_set"]
    ]

    merged["actual_count"] = [
        len(x)
        for x in merged["actual_set"]
    ]

    # Precision
    merged["precision"] = (
        merged["true_positive"]
        / merged["predicted_count"].replace(
            0,
            float("nan")
        )
    )

    # Recall
    merged["recall"] = (
        merged["true_positive"]
        / merged["actual_count"].replace(
            0,
            float("nan")
        )
    )

    # Correct no-match cases
    no_match = (
        (merged["predicted_count"] == 0)
        & (merged["actual_count"] == 0)
    )

    merged.loc[no_match, "precision"] = 1.0
    merged.loc[no_match, "recall"] = 1.0

    # F0.5
    beta = 0.5

    denominator = (
        beta**2 * merged["precision"]
        + merged["recall"]
    )

    merged["f05"] = (
        (1 + beta**2)
        * merged["precision"]
        * merged["recall"]
        / denominator
    )

    merged.loc[no_match, "f05"] = 1.0

    # Cases with zero precision/recall
    merged["precision"] = merged[
        "precision"
    ].fillna(0.0)

    merged["recall"] = merged[
        "recall"
    ].fillna(0.0)

    merged["f05"] = merged[
        "f05"
    ].fillna(0.0)

    result_df = merged[
        [
            source1_column,
            "precision",
            "recall",
            "f05",
        ]
    ]

    metrics = {
        "macro_precision": result_df[
            "precision"
        ].mean(),

        "macro_recall": result_df[
            "recall"
        ].mean(),

        "macro_f05": result_df[
            "f05"
        ].mean(),
    }

    return result_df, metrics