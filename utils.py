"""
This script provides necessary functions that help the evaluation in eval_qags.py

"""

from typing import List, Dict, Tuple
import json


import rouge_score.scoring
from scipy.stats import pearsonr, spearmanr
from rouge_score import rouge_scorer
from tqdm import tqdm
from mosestokenizer import *


def most_common_list_element(lst: List):
    """
    find the most common elements from a list, from ['yes', 'no', 'yes'] will return "yes"
    """
    return max(set(lst), key=lst.count)


def get_scorer(fast: bool = False) -> rouge_scorer.RougeScorer:
    # Skip LCS computation for 10x speedup during debugging.
    if fast:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2"], use_stemmer=True)
    else:
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

    return scorer


def generate_rouge_scores(
    generated_summary: str, source_text: str
) -> Dict[str, rouge_score.scoring.Score]:
    rouge_score = get_scorer().score(source_text, generated_summary)
    return rouge_score


def get_qag_whole_summary_sents(sample: dict) -> str:
    """
    combine summaries sentences to a complete summary
    """
    summary = ""
    for sentence_dic in sample["summary_sentences"]:
        summary = summary + sentence_dic["sentence"] + " "
    return summary.rstrip()


def calculate_correlation_score(lst1: List[float], lst2: List[float]) -> Tuple[float]:

    pearson_corr, pearson_p_value = pearsonr(lst1, lst2)
    spearman_corr, spearman_p_value = spearmanr(lst1, lst2)

    print(
        f"pearson correlation is {pearson_corr} with pearson_p_value {pearson_p_value};\n"
        f"spearman correlation is {spearman_corr} with spearman_p_value {spearman_p_value}"
    )

    return pearson_corr, pearson_p_value, spearman_corr, spearman_p_value

# for CoCo score (but we have not implemented it successfully in the end)
def detokenize(text: str) -> str:
    detokenizer = MosesDetokenizer("en")
    words = text.split(" ")
    return detokenizer(words)


def get_src_sys_lines_for_BART(
    samples: List[dict], json_file_name: str
) -> Tuple[List[str], List[str]]:

    if json_file_name == "qags-cnndm.jsonl" or json_file_name == "qags-xsum.jsonl":
        src_lines: List[str] = [sample["article"] for sample in tqdm(samples)]
        sys_lines: List[str] = [
            get_qag_whole_summary_sents(sample) for sample in tqdm(samples)
        ]

    if json_file_name == "summeval.jsonl":
        src_lines: List[str] = [sample["text"] for sample in tqdm(samples)]
        sys_lines: List[str] = [sample["decoded"] for sample in tqdm(samples)]

    src_lines = [detokenize(line) for line in src_lines]
    sys_lines = [detokenize(line) for line in sys_lines]

    return src_lines, sys_lines


def save_data(data, arg):
    json.dump(data, open(arg.output, "w"))
