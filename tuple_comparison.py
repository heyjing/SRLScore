"""
This is a script that implements the tuple comparison algorithm. 
"""


from typing import List, Optional
import numpy as np
from rouge_score import rouge_scorer
from jiwer import wer
import extract_tuples as et


def exact_match_string_similarity(
    source_str: Optional[str], summary_str: Optional[str]
):
    if source_str and summary_str and source_str == summary_str:
        similarity_score = 1.0
    else:
        similarity_score = 0
    return similarity_score


def spacy_string_similarity(source_str: Optional[str], summary_str: Optional[str]):
    if source_str and summary_str:
        nlp = et.load_spacy_model()
        doc1 = nlp(source_str)
        doc2 = nlp(summary_str)
        similarity_score = round(doc1.similarity(doc2), 2)
    else:
        similarity_score = 0
    return similarity_score


def rouge_precision_string_similarity(
    source_str: Optional[str], summary_str: Optional[str]
):
    if source_str and summary_str:
        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
        scores = scorer.score(source_str, summary_str)
        precision = scores["rouge1"][0]
        similarity_score = round(precision, 2)
    else:
        similarity_score = 0
    return similarity_score


def character_error_rate_string_similarity(
    source_str: Optional[str], summary_str: Optional[str]
):
    if source_str and summary_str:
        error = wer(source_str, summary_str)
        similarity_score = round(error, 2)
    else:
        similarity_score = 0
    return similarity_score


def calculate_string_similarity(
    source_str: Optional[str], summary_str: Optional[str], method: str
) -> float:
    methods = {
        "exact": exact_match_string_similarity,
        "spacy": spacy_string_similarity,
        "rouge": rouge_precision_string_similarity,
        "CER": character_error_rate_string_similarity,
    }
    return methods.get(method)(source_str, summary_str)


def compare_two_tuples(
    source_tuple: tuple, generated_tuple: tuple, string_comparison_method: str
) -> float:
    """
    This function calculates consistency score of two tuples.
    """
    # hard-coded values for the semantic roles
    weights = [1 / 3, 0, 1 / 3, 1 / 3, 0, 0, 0]
    indic = np.array([1 if x else 0 for x in generated_tuple])
    pairwise_similarity_scores = [
        calculate_string_similarity(source_str, generated_str, string_comparison_method)
        for source_str, generated_str in zip(source_tuple, generated_tuple)
    ]

    normalized_weight = 1 / (np.sum(indic * weights))
    consistency_score = normalized_weight * np.sum(
        [indic * pairwise_similarity_scores * weights]
    )
    return round(consistency_score, 2)


def compare_tuple_with_relevant_tuples(
    relevant_source_tuples: List[tuple],
    generated_tup: tuple,
    string_comparison_method: str,
) -> float:
    """
    This function compares a generated tuple with all of its relevant tuples from source text and takes the max as final score.
    """
    tuple_final_score = 0
    for relevant_source_tup in relevant_source_tuples:
        consistency_score = compare_two_tuples(
            source_tuple=relevant_source_tup,
            generated_tuple=generated_tup,
            string_comparison_method=string_comparison_method,
        )
        if consistency_score >= tuple_final_score:
            tuple_final_score = consistency_score

    return tuple_final_score


if __name__ == "__main__":
    print(
        compare_two_tuples(
            ("Hans", None, "bought", "a vase", "sister", None, None),
            ("Hans", None, "send", "a vase", None, None, None),
            string_comparison_method="rouge",
        )
    )

    print(calculate_string_similarity("apple is good", "Apple good", "CER"))
    print(calculate_string_similarity("apple good", "apple is good", "CER"))
    print(calculate_string_similarity("a", None, "CER"))
    print(calculate_string_similarity(None, "a", "CER"))
    print(calculate_string_similarity(None, None, "CER"))

    print(calculate_string_similarity("a big apple", "apple", "CER"))
    print(calculate_string_similarity("a big apple", "apple", "rouge"))

