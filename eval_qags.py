"""
Evaluates qags samples, calculate pearson and spearman correlation between human_annotated_scores and different metric_scores
"""
from statistics import mean, median, variance
import statistics as stat
import argparse
import sys
import os

import numpy as np
import time
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from sklearn.metrics import mean_squared_error

from utils import *
from calculate_faithful_score import CalculateFactualScore
from baselines.bart_score import BARTScorer


def get_samples(arg) -> List[dict]:
    """
    read json file and store each sample in a list
    """
    samples = []
    with open(os.path.join("./data/", arg.json_file)) as f:
        for line in f:
            samples.append(json.loads(line))

    return samples


def compute_rouge_scores(arg) -> List[float]:

    samples: List[dict] = get_samples(arg)

    if arg.json_file == "qags-cnndm.jsonl" or arg.json_file == "qags-xsum.jsonl":
        rouge_scores: List[float] = [
            generate_rouge_scores(
                get_qag_whole_summary_sents(sample), sample["article"]
            )["rouge1"].fmeasure
            for sample in samples
        ]

    if arg.json_file == "summeval.jsonl":
        rouge_scores: List[float] = [
            generate_rouge_scores(sample["decoded"], sample["text"])["rouge1"].fmeasure
            for sample in samples
        ]
    return rouge_scores


def compute_meteor_scores(arg) -> List[float]:
    samples: List[dict] = get_samples(arg)

    if arg.json_file == "qags-cnndm.jsonl" or arg.json_file == "qags-xsum.jsonl":
        meteor_scores: List[float] = [
            meteor_score(
                [sample["article"].split()], get_qag_whole_summary_sents(sample).split()
            )
            for sample in samples
        ]

    if arg.json_file == "summeval.jsonl":
        meteor_scores: List[float] = [
            meteor_score([sample["text"].split()], sample["decoded"].split())
            for sample in samples
        ]

    return meteor_scores


def compute_bleu_scores(arg) -> List[float]:
    samples: List[dict] = get_samples(arg)

    if arg.json_file == "qags-cnndm.jsonl" or arg.json_file == "qags-xsum.jsonl":
        bleu_scores: List[float] = [
            sentence_bleu(
                [sample["article"].split()], get_qag_whole_summary_sents(sample).split()
            )
            for sample in samples
        ]

    if arg.json_file == "summeval.jsonl":
        bleu_scores: List[float] = [
            sentence_bleu([sample["text"].split()], sample["decoded"].split())
            for sample in samples
        ]

    return bleu_scores


def compute_annotated_scores(arg) -> List[float]:
    """
    There are 3 annotations per sample in qags-cnndm and qags-xsum datasets.
    To obtain a single consistency score per summary, the authors first
    take the majority vote for each sentence in a summary, then average the binary scores 
    across summary sentences to produce a final score.
    """

    samples: List[dict] = get_samples(arg)

    if arg.json_file == "qags-cnndm.jsonl" or arg.json_file == "qags-xsum.jsonl":
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
            sample_score = np.mean(indic)
            # print(sentence_scores, indic, sample_score)
            # print(f"sample {id} was processed!")
            sample_scores.append(sample_score)

    if arg.json_file == "summeval.jsonl":
        sample_scores = []
        for id, sample in tqdm(enumerate(samples)):
            sample_score = mean(d["consistency"] for d in sample["expert_annotations"])
            sample_scores.append(sample_score)

    return sample_scores


def compute_bartscore(arg) -> List[float]:

    samples: List[dict] = get_samples(arg)
    src_lines, sys_lines = get_src_sys_lines_for_BART(samples, arg.json_file)

    # load bart models: BARTScore or BARTScore-CNN
    if arg.metric_name == "bartscore":
        bart_scorer = BARTScorer(device="cuda:0", checkpoint="facebook/bart-large")
    if arg.metric_name == "bartscore_cnn":
        bart_scorer = BARTScorer(device="cuda:0", checkpoint="facebook/bart-large-cnn")
    if arg.metric_name == "bartscore_para":
        bart_scorer = BARTScorer(device="cuda:0", checkpoint="facebook/bart-large-cnn")
        bart_scorer.load()

    return bart_scorer.score(src_lines, sys_lines, batch_size=4)


def compute_goodrich_inspired_score(arg) -> List[float]:
    samples: List[dict] = get_samples(arg)

    weights = [1 / 3, 0, 1 / 3, 1 / 3, 0, 0, 0]

    calcu = CalculateFactualScore(
        do_coref=False, string_comparison_method="goodrich", weights=weights
    )

    if arg.json_file == "qags-cnndm.jsonl" or arg.json_file == "qags-xsum.jsonl":
        srl_scores: List[float] = [
            calcu.goodrich_inspired_score(
                sample["article"], get_qag_whole_summary_sents(sample)
            )
            for sample in tqdm(samples, desc="processing sample: ")
        ]

    if arg.json_file == "summeval.jsonl":
        srl_scores: List[float] = [
            calcu.goodrich_inspired_score(sample["text"], sample["decoded"])
            for sample in tqdm(samples, desc="processing sample: ")
        ]

    return srl_scores


def compute_srl_metric_scores(arg) -> List[float]:

    samples: List[dict] = get_samples(arg)

    input_do_coref: str = arg.do_coref
    do_coref = True if input_do_coref == "True" else False

    method: str = arg.string_comparison_method

    if arg.weights == "1/3":
        weights = [1 / 3, 0, 1 / 3, 1 / 3, 0, 0, 0]
    elif arg.weights == "leave_agent":
        weights = [0, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]
    elif arg.weights == "leave_negation":
        weights = [1 / 6, 0, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]
    elif arg.weights == "leave_relation":
        weights = [1 / 6, 1 / 6, 0, 1 / 6, 1 / 6, 1 / 6, 1 / 6]
    elif arg.weights == "leave_patient":
        weights = [1 / 6, 1 / 6, 1 / 6, 0, 1 / 6, 1 / 6, 1 / 6]
    elif arg.weights == "leave_recipient":
        weights = [1 / 6, 1 / 6, 1 / 6, 1 / 6, 0, 1 / 6, 1 / 6]
    elif arg.weights == "leave_time":
        weights = [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 0, 1 / 6]
    elif arg.weights == "leave_location":
        weights = [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 0]
    else:
        weights = [1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7]

    calcu = CalculateFactualScore(
        do_coref=do_coref, string_comparison_method=method, weights=weights
    )

    if arg.json_file == "qags-cnndm.jsonl" or arg.json_file == "qags-xsum.jsonl":
        srl_scores: List[float] = [
            calcu.calculate_factual_score(
                sample["article"], get_qag_whole_summary_sents(sample)
            )
            for sample in tqdm(samples, desc="processing sample: ")
        ]

    if arg.json_file == "summeval.jsonl":
        srl_scores: List[float] = [
            calcu.calculate_factual_score(sample["text"], sample["decoded"])
            for sample in tqdm(samples, desc="processing sample: ")
        ]

    return srl_scores


def compute_metric_scores(arg) -> List[float]:
    metrics = {
        "rouge": compute_rouge_scores,
        "meteor": compute_meteor_scores,
        "bleu": compute_bleu_scores,
        "srl": compute_srl_metric_scores,
        "goodrich": compute_goodrich_inspired_score,
        "bartscore": compute_bartscore,
        "bartscore_cnn": compute_bartscore,
        "bartscore_para": compute_bartscore,
    }
    return metrics.get(arg.metric_name)(arg)


def eval(arg):

    human_annotated_scores: List[float] = compute_annotated_scores(arg)

    start = time.time()
    metric_scores: List[float] = compute_metric_scores(arg)
    end = time.time()

    print(
        f"human_annotated_scores: {human_annotated_scores}, length is {len(human_annotated_scores)}\n",
        f"metric_scores: {metric_scores}, length is {len(metric_scores)}\n",
    )

    (
        pearson_corr,
        pearson_p_value,
        spearman_corr,
        spearman_p_value,
    ) = calculate_correlation_score(human_annotated_scores, metric_scores)

    results = {
        "setting": f"{arg.json_file} + {arg.metric_name} + weights {arg.weights} + Do coref: {arg.do_coref} + {arg.string_comparison_method} similarity",
        "processing_time": f"Processing {arg.json_file} file took {(end - start):.6f} s.",
        "human_annotated_scores": human_annotated_scores,
        "metric_scores": metric_scores,
        "mean_metric_scores": mean(metric_scores),
        "min_metric_scores": min(metric_scores),
        "max_metric_scores": max(metric_scores),
        "median_metric_scores": median(metric_scores),
        "std": stat.stdev(metric_scores),
        "variance": variance(metric_scores),
        "mean_squared_error": mean_squared_error(human_annotated_scores, metric_scores),
        "pearson_correlation": pearson_corr,
        "pearson_p_value": pearson_p_value,
        "spearman_corr": spearman_corr,
        "spearman_p_value": spearman_p_value,
        "correlation_description": f"pearson correlation is {pearson_corr} with pearson_p_value {pearson_p_value}; spearman correlation is {spearman_corr} with spearman_p_value {spearman_p_value}",
    }

    return results


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser()

    PARSER.add_argument(
        "--json_file",
        type=str,
        help="Json file name: 'qags-cnndm.jsonl' or 'qags-xsum.jsonl' or 'summeval.jsonl'",
    )

    PARSER.add_argument(
        "--metric_name",
        type=str,
        help="Metric name: 'rouge', 'bleu', 'meteor', 'srl', 'goodrich', 'bartscore', 'bartscore_cnn', 'bartscore_para'",
    )

    PARSER.add_argument(
        "--weights",
        type=str,
        help="default is 1/7 weights; enter '1/3' to change to openIE equivalent",
    )

    PARSER.add_argument(
        "--string_comparison_method",
        type=str,
        help="string_comparison_method: 'exact' or 'rouge' or 'spacy'",
    )

    PARSER.add_argument("--do_coref", type=str, help="Please enter 'True' or 'False'")

    PARSER.add_argument("--output", type=str, help="Path for saving data")

    ARGS = PARSER.parse_args()

    if ARGS.json_file not in [
        "qags-cnndm.jsonl",
        "qags-xsum.jsonl",
        "summeval.jsonl",
    ]:
        raise RuntimeError(
            "To run script, please specify 'qags-cnndm.jsonl' or 'qags-xsum.jsonl' or 'summeval.jsonl'"
        )

    if ARGS.metric_name not in [
        "rouge",
        "srl",
        "goodrich",
        "bartscore",
        "bartscore_cnn",
        "bartscore_para",
        "meteor",
        "bleu",
    ]:
        raise RuntimeError(
            "To run script, please specify 'srl' or 'rouge' or 'goodrich' or 'bartscore' or 'bartscore_cnn'"
        )

    if ARGS.metric_name == "srl" and ARGS.do_coref not in ["True", "False"]:
        raise RuntimeError(
            "To use SRL metric please specify do_coref as 'True' or 'False' first"
        )

    if ARGS.metric_name == "srl" and ARGS.string_comparison_method not in [
        "exact",
        "spacy",
        "rouge",
    ]:
        raise RuntimeError(
            "To run script please specify comparison methods 'exact', 'spacy' or 'rouge'"
        )

    results = eval(ARGS)
    save_data(results, ARGS)
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

# {
#     "article": 'Venezuela\'s acting president nicolas maduro says he will turn the office where the late president hugo chavez worked into a museum. Mr maduro said the room would be kept intact and a wing of the presidential palace turned into a monument to mr chavez\'s "bolivarian revolution". Mr chavez died of cancer last month. Following his death, presidential elections were called for 14 april, pitting mr maduro against main opposition candidate henrique capriles. Mr capriles says that despite mr maduro\'s attempt at portraying himself as mr chavez\'s heir, the acting president lacks the late leader\'s charisma. "i\'m going to look after it for a few years, but it\'ll always be chavez\'s home," mr maduro said referring to the miraflores palace in caracas, where venezuela\'s presidents have their office. Full of confidence ahead of sunday\'s polls, mr maduro said he would "occupy a small office in miraflores in another wing", so that venezuelans could roam the rooms of the presidential palace and "learn how the commander had lived, and what he had eaten". Fierce battle. Since mr chavez died on 5 march, mr maduro has cast himself as his natural successor in office. He has called himself "chavez\'s son" and said that the late president had appeared to him in the form of a little bird. Speaking last week at the official start of the presidential campaign in the house where mr chavez was born, mr maduro said a small bird had flown around him three times and looked at him "oddly", at which point he he had felt in his soul it was a message from mr chavez. "i felt its blessing, telling us:\'today the battle begins, go for victory, you have my blessing\'," mr maduro said. Henrique capriles mocked mr maduro saying that he had "not seen a little bird, but swallowed one, the one he has in his head!" He has also questioned mr maduro\'s ability to lead the country saying that "whatever the outcome, i don\'t see how nicolas maduro has the capacity to stay for an extended time in government". Mr capriles has dismissed polls that suggest mr maduro has an unassailable lead over him. "of course i can win," he told afp news agency, "maduro lacks charisma and leadership". Mr maduro said his rival was "jealous" and promised to beat him by 10 million votes in sunday\'s polls.',
#     "summary_sentences": [
#         {
#             "sentence": "Venezuelan acting president nicolas maduro has said he will keep the room where he was born as a shrine to late leader hugo chavez.",
#             "responses": [
#                 {"worker_id": 85, "response": "no"},
#                 {"worker_id": 87, "response": "no"},
#                 {"worker_id": 84, "response": "yes"},
#             ],
#         }
#     ],
# }

