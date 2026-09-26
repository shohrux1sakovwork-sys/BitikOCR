"""Score OCR predictions against human-reviewed benchmark facts."""

import unicodedata
from collections import defaultdict
from typing import Any


def normalize_fact_text(text: str) -> str:
    """Normalize case, Unicode, punctuation, and whitespace for matching."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(
        "".join(
            character if character.isalnum() else " " for character in text
        ).split()
    )


def substring_similarity(value: str, prediction: str) -> float:
    """Return best normalized Levenshtein similarity to any substring."""
    value = normalize_fact_text(value)
    prediction = normalize_fact_text(prediction)
    if not value:
        return 1.0
    if value in prediction:
        return 1.0
    previous = list(range(len(value) + 1))
    best_distance = len(value)
    for predicted_character in prediction:
        current = [0]
        for index, value_character in enumerate(value, start=1):
            current.append(
                min(
                    previous[index] + 1,
                    current[index - 1] + 1,
                    previous[index - 1]
                    + (predicted_character != value_character),
                )
            )
        best_distance = min(best_distance, current[-1])
        previous = current
    return max(0.0, 1.0 - best_distance / len(value))


def score_annotated_facts(
    prediction_rows: list[dict[str, Any]],
    annotations: dict[str, list[dict[str, Any]]],
    fuzzy_threshold: float = 0.85,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    """Score required facts by category and return per-fact decisions."""
    if not prediction_rows:
        raise ValueError("Provide at least one prediction row.")
    if not 0 <= fuzzy_threshold <= 1:
        raise ValueError("fuzzy_threshold must be between zero and one.")
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    details = []
    exact_documents = 0
    for row in prediction_rows:
        document_id = row.get("id")
        if document_id not in annotations:
            raise ValueError(f"Missing fact annotations for {document_id!r}.")
        document_matches = []
        for fact in annotations[document_id]:
            similarity = substring_similarity(fact["value"], row["prediction"])
            threshold = fuzzy_threshold if fact["fuzzy"] else 1.0
            fact_matched = similarity >= threshold
            category = str(fact["category"])
            counts[category][0] += 1
            counts[category][1] += fact_matched
            counts["all"][0] += 1
            counts["all"][1] += fact_matched
            kind = "fuzzy" if fact["fuzzy"] else "exact"
            counts[kind][0] += 1
            counts[kind][1] += fact_matched
            document_matches.append(fact_matched)
            details.append(
                {
                    "id": document_id,
                    "category": category,
                    "value": fact["value"],
                    "fuzzy": bool(fact["fuzzy"]),
                    "similarity": similarity,
                    "matched": fact_matched,
                }
            )
        exact_documents += bool(document_matches) and all(document_matches)
    metrics: dict[str, float | int] = {
        "documents": len(prediction_rows),
        "documents_all_facts_correct": exact_documents,
        "document_accuracy": exact_documents / len(prediction_rows),
        "fuzzy_threshold": fuzzy_threshold,
    }
    for category in sorted(counts):
        total, matched_count = counts[category]
        metrics[f"{category}_total"] = total
        metrics[f"{category}_matched"] = matched_count
        metrics[f"{category}_accuracy"] = (
            matched_count / total if total else 0.0
        )
    return metrics, details
