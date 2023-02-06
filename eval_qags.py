"""
Evaluates qags samples, calculate pearson and spearman correlation between human_annotated_scores and different metric_scores
"""
from typing import List, Dict
import json
import sys
import os

import rouge_score.scoring
from scipy.stats import pearsonr, spearmanr
from rouge_score import rouge_scorer
from tqdm import tqdm
import numpy as np

from calculate_faithful_score import CalculateFactualScore


def get_samples(json_file) -> List[dict]:
    """
    read json file and store each sample in a list
    """
    samples = []
    with open(os.path.join(sys.path[0], json_file)) as f:
        for line in f:
            samples.append(json.loads(line))

    return samples


def most_common_list_element(lst):
    """
    find the most common elements from a list, from ['yes', 'no', 'yes'] will return "yes"
    """
    return max(set(lst), key=lst.count)


def calculate_annotated_scores(samples: list) -> List[float]:
    """
    There are 3 annotations per sample in qags-cnndm and qags-xsum datasets.
    To obtain a single consistency score per summary, the authors first
    take the majority vote for each sentence in a summary, then average the binary scores 
    across summary sentences to produce a final score.
    """
    sample_scores = []
    for id, sample in tqdm(enumerate(samples)):
        sentence_scores = []

        for sentence_dic in sample["summary_sentences"]:
            # extract the three votes for each summary sentence, like ['yes', 'no', 'yes']
            responses = [el["response"] for el in sentence_dic["responses"]]
            # take the majority vote for each sentence, "yes"
            sentence_scores.append(most_common_list_element(responses))
            # print(responses)

        indic = np.array([1 if x == "yes" else 0 for x in sentence_scores])
        sample_score = round(np.mean(indic), 2)
        # print(sentence_scores, indic, sample_score)
        # print(f"sample {id} was processed!")
        sample_scores.append(sample_score)

    return sample_scores


def get_scorer(fast: bool = False) -> rouge_scorer.RougeScorer:
    # Skip LCS computation for 10x speedup during debugging.
    if fast:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2"], use_stemmer=True)
    else:
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

    return scorer


def generate_rouge_scores(generated_summary: str, source_text: str) -> Dict[str, rouge_score.scoring.Score]:
    rouge_score = get_scorer().score(source_text, generated_summary)
    return rouge_score


def get_whole_summary_sents(sample: dict) -> str:
    """
    combine summaries sentences to a complete summary
    """
    summary = ""
    for sentence_dic in sample["summary_sentences"]:
        summary = summary + sentence_dic["sentence"] + " "
    return summary.rstrip()


def compute_rouge_scores(samples: list) -> List[float]:
    rouge_scores: List[float] = [
            generate_rouge_scores(get_whole_summary_sents(sample), sample["article"])["rouge1"].fmeasure
            for sample in samples
    ]
    return rouge_scores


def compute_srl_metric_scores(samples: list) -> List[float]:
    input_do_coref: str = input("Do coreference resolution? (True or False): ")
    do_coref = True if input_do_coref == "True" else False

    method: str = input("Which similarity method? (exact, spacy or rouge): ")
    if method not in ["exact", "spacy", "rouge"]:
        raise ValueError("Only comparison methods 'exact', 'spacy' and 'rouge' are supported!")

    calcu = CalculateFactualScore(do_coref=do_coref, string_comparison_method=method)
    srl_scores: List[float] = [
        calcu.calculate_factual_score(
            sample["article"], get_whole_summary_sents(sample)
        )
        for sample in tqdm(samples, desc="processing sample: ")
    ]

    # remove summary scores where no tuples can be extracted from summary
    srl_scores = [score for score in srl_scores if score != -1]
    return srl_scores


def compute_metric_scores(samples: list, metric: str) -> List[float]:
    metrics = {
        "rouge": compute_rouge_scores,
        "srl": compute_srl_metric_scores,
    }
    return metrics.get(metric)(samples)


def print_correlation_score(lst1: List[float], lst2: List[float]) -> None:

    pearson_corr, pearson_p_value = pearsonr(lst1, lst2)
    spearman_corr, spearman_p_value = spearmanr(lst1, lst2)

    print(
        f"pearson correlation is {pearson_corr} with pearson_p_value {pearson_p_value};\n"
        f"spearman correlation is {spearman_corr} with spearman_p_value {spearman_p_value}"
    )


if __name__ == "__main__":

    # sys.argv[1] can be "qags-cnndm.jsonl" or "qags-xsum.jsonl"
    samples: List[dict] = get_samples(sys.argv[1])

    human_annotated_scores: List[float] = calculate_annotated_scores(samples)

    metric_scores = compute_metric_scores(samples, sys.argv[2])
    print("human_annotated_scores", human_annotated_scores)
    print("metric_scores,", metric_scores)

    print_correlation_score(human_annotated_scores, metric_scores)
# {
#     "article": "Vitamin and mineral supplements are becoming more and more popular as health conscious shoppers focus on good nutrition, but do we really need pills to optimise our diet? Not according to nutritionist and author sarah flower, who says that cooking with the right ingredients should give you all the goodness you need. ` the cleaner your diet - using fresh ingredients and cooking at home - the less likely you are to need to rely on supplements to boost your health.' She told mailonline. Scroll down for video. It's time to ditch vitamin pills for a diet rich in clean, fresh and unprocessed foods, says sarah flower. ` the typical western diet is heavily processed and sugar ridden,' explains sarah, `this makes us more susceptible to vitamin and mineral deficiencies.' And while it may seem like common sense to eat more unprocessed and raw foods, ms flower believes we are still not doing enough. ` we are living in a society where it is possible to be overweight and deficient in essential nutrients.' She continued.' A diet rich in oily fish, whole grains, lean protein, fruit and vegetables should provide enough nutrients,' she said. Other factors to consider include your ability to absorb the food - digestive complaints can often impede our ability to absorb nutrients. ` pregnancy, ill health and the elderly may need more support,' she said. And menstruating women may benefit from adding oils ( evening primrose oil ) and a multivitamin rich in magnesium to help alleviate pms symptoms ( ms flowers recommends magnesium citrate ). Always opt for steaming over boiling vegetables and eat as many raw pieces as you can every day. ` fruit and vegetables not only contain vitamins but also vital phytonutrients, which have an amazing ability to protect us against degenerative diseases such as cancer, alzheimer's and heart disease,'",
#     "summary_sentences": [
#         {
#             "sentence": "` the typical western diet is heavily processed and sugar ridden,' says author sarah flower.",
#             "responses": [
#                 {"worker_id": 0, "response": "yes"},
#                 {"worker_id": 3, "response": "no"},
#                 {"worker_id": 25, "response": "yes"},
#             ],
#         },
#         {
#             "sentence": "A diet rich in oily fish, whole grains, lean protein, fruit and vegetables should provide enough nutrients.",
#             "responses": [
#                 {"worker_id": 0, "response": "yes"},
#                 {"worker_id": 3, "response": "yes"},
#                 {"worker_id": 25, "response": "yes"},
#             ],
#         },
#         {
#             "sentence": "Ms flower believes we are still not doing enough.",
#             "responses": [
#                 {"worker_id": 0, "response": "yes"},
#                 {"worker_id": 3, "response": "yes"},
#                 {"worker_id": 25, "response": "yes"},
#             ],
#         },
#     ],
# }

