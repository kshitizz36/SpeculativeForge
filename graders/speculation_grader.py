from __future__ import annotations

import re
from collections import Counter


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def token_overlap_score(reference: str, candidate: str) -> float:
    ref_tokens = normalize_text(reference).split()
    cand_tokens = normalize_text(candidate).split()
    if not ref_tokens:
        return 1.0
    ref_counter = Counter(ref_tokens)
    cand_counter = Counter(cand_tokens)
    overlap = sum((ref_counter & cand_counter).values())
    return overlap / max(len(ref_tokens), 1)


def score_pairs(
    reference_outputs: list[str],
    candidate_outputs: list[str],
) -> list[dict[str, float | bool]]:
    compared = zip(reference_outputs, candidate_outputs, strict=False)
    scores: list[dict[str, float | bool]] = []
    for reference, candidate in compared:
        normalized_reference = normalize_text(reference)
        normalized_candidate = normalize_text(candidate)
        overlap = token_overlap_score(reference, candidate)
        scores.append(
            {
                "token_overlap": overlap,
                "exact_match": normalized_reference == normalized_candidate,
            }
        )
    return scores


def quality_match_rate(reference_outputs: list[str], candidate_outputs: list[str]) -> float:
    if not reference_outputs:
        return 0.0
    scores = [
        float(item["token_overlap"])
        for item in score_pairs(reference_outputs, candidate_outputs)
    ]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
