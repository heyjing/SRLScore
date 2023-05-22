"""
Test results for significance in the respective experiments.
"""

from typing import List
import json
from scipy.stats import pearsonr, spearmanr

from summaries.evaluation import permutation_test


def pearson(x, y):
    """
    Wrapper function around pearson correlation that only returns the correlation score.
    """
    return pearsonr(x, y)[0]


def spearman(x, y):
    """
    Wrapper function around spearman correlation that only returns the correlation score.
    """
    return spearmanr(x, y)[0]


def get_sample_scores(fn):
    with open(fn) as f:
        data = json.load(f)
    return data["metric_scores"]


def get_gold_labels(fn):
    with open(fn) as f:
        data = json.load(f)
    return data["human_annotated_scores"]


def get_coco_scores(fn):
    with open(fn, "r") as f:
        scores = list(f)

    return [float(score) for score in scores]


def improvement_over_baseline():
    truth = {
        "cnndm": get_gold_labels("./baselines/eval_results/cnndm_srl_goodrich.json"),
        "xsum": get_gold_labels("./baselines/eval_results/xsum_srl_goodrich.json"),
        "summeval": get_gold_labels("./baselines/eval_results/summeval_srl_goodrich.json")
    }

    for dataset in ["cnndm", "xsum", "summeval"]:
        base_scores = {}
        for baseline in ["rouge", "bleu", "meteor"]:
            base_scores[baseline] = get_sample_scores(f"./baselines/eval_results/{dataset}_{baseline}.json")

        if dataset in ["cnndm", "summeval"]:
            srl_scores = get_sample_scores(f"./baselines/eval_results/{dataset}_srl_fullset_False_rouge.json")
        else:  # QAGS-XSUM
            srl_scores = get_sample_scores(f"./baselines/eval_results/{dataset}_srl_fullset_False_spacy.json")


        for metric, scores in base_scores.items():
            print(f"{dataset}, {metric}")
            print(f"Pearson: {round(pearson(truth[dataset], scores), 2)}, "
                  f"Spearman: {round(spearman(truth[dataset], scores), 2)}")

            pearson_p = permutation_test(truth[dataset], srl_scores, scores, pearson)
            spearman_p = permutation_test(truth[dataset], srl_scores, scores, spearman)

            print(f"Corrected Bonferroni thresholds: (p < {0.05 / 3:.4f}) or (p < {0.01 / 3:.4f})")
            print(f"{dataset} {metric} Pearson p-value of improvement: {pearson_p}")
            print(f"{dataset} {metric} Spearman p-value of improvement: {spearman_p}")

        print("\n\n")


def significance_between_methods():
    truth = {
        "cnndm": get_gold_labels("./baselines/eval_results/cnndm_srl_goodrich.json"),
        "xsum": get_gold_labels("./baselines/eval_results/xsum_srl_goodrich.json"),
        "summeval": get_gold_labels("./baselines/eval_results/summeval_srl_goodrich.json")
    }

    for dataset in ["cnndm", "xsum", "summeval"]:
        bart_scores = {}
        for variant in ["", "cnn", "para"]:
            if variant == "":
                bart_scores[variant] = get_sample_scores(f"./baselines/eval_results/{dataset}_bartscore.json")
            else:
                bart_scores[variant] = get_sample_scores(f"./baselines/eval_results/{dataset}_bartscore_{variant}.json")

        coco_scores = {}
        # Fix naming convention
        coco_dataset = f"qags-{dataset}" if dataset in ["cnndm", "xsum"] else dataset
        for variant in ["span", "sent"]:
            coco_scores[variant] = get_coco_scores(f"./coco_scores/{coco_dataset}-{variant}-scores.txt")

        srl_scores = {}
        # Equivalent to ["_base", "_coref"] variants in the paper
        for variant in ["fullset_False", "fullset_True"]:
            if dataset in ["cnndm", "summeval"]:
                srl_scores[variant] = get_sample_scores(f"./baselines/eval_results/{dataset}_srl_{variant}_rouge.json")
            else:  # QAGS-XSUM
                srl_scores[variant] = get_sample_scores(f"./baselines/eval_results/{dataset}_srl_{variant}_spacy.json")

        other_scores = list(srl_scores.values()) + list(coco_scores.values())
        for variant, variant_scores in bart_scores.items():
            print(f"{dataset}, BARTScore {variant} against rest")
            test_against_other_methods(variant_scores, truth[dataset], other_scores)

        other_scores = list(bart_scores.values()) + list(coco_scores.values())
        for variant, variant_scores in srl_scores.items():
            print(f"{dataset}, SRLScore {variant} against rest")
            test_against_other_methods(variant_scores, truth[dataset], other_scores)

        other_scores = list(bart_scores.values()) + list(srl_scores.values())
        for variant, variant_scores in coco_scores.items():
            print(f"{dataset}, CoCo {variant} against rest")
            test_against_other_methods(variant_scores, truth[dataset], other_scores)

        print(f"\n\n")


def test_against_other_methods(scores_to_test: List, ground_truth: List, other_method_scores: List[List]):
    correction_level = len(other_method_scores)

    pearson_005 = True
    pearson_001 = True
    spearman_005 = True
    spearman_001 = True

    for other_scores in other_method_scores:
        pearson_p = permutation_test(ground_truth, scores_to_test, other_scores, pearson)
        if pearson_p > 0.05 / correction_level:
            pearson_005 = False
        if pearson_p > 0.01 / correction_level:
            pearson_001 = False

        spearman_p = permutation_test(ground_truth, scores_to_test, other_scores, spearman)
        if spearman_p > 0.05 / correction_level:
            spearman_005 = False
        if spearman_p > 0.01 / correction_level:
            spearman_001 = False

    if pearson_001:
        print(f"Highly significant Pearson improvement (p < 0.01)")
    if pearson_005:
        print(f"Significant Pearson improvement (p < 0.05)")

    if not pearson_005 and not pearson_001:
        print(f"No Pearson significance detected")

    if spearman_001:
        print(f"Highly significant Spearman improvement (p < 0.01)")
    if spearman_005:
        print(f"Significant Spearman improvement (p < 0.05)")

    if not spearman_005 and not spearman_001:
        print(f"No Spearman significance detected")


if __name__ == '__main__':
    gold_scores = {
        "qags-cnndm": get_gold_labels("./baselines/eval_results/cnndm_srl_goodrich.json"),
        "qags-xsum": get_gold_labels("./baselines/eval_results/xsum_srl_goodrich.json"),
        "summeval": get_gold_labels("./baselines/eval_results/summeval_srl_goodrich.json")
    }

    # Results of Table 3
    print("Results of Table 3:")
    static_scores = {
        "qags-cnndm": get_sample_scores("./baselines/eval_results/cnndm_srl_fullset_staticweights_False_rouge.json"),
        "qags-xsum": get_sample_scores("./baselines/eval_results/xsum_srl_fullset_staticweights_False_spacy.json"),
        "summeval": get_sample_scores("./baselines/eval_results/summeval_srl_fullset_staticweights_False_rouge.json")
    }
    dynamic_scores = {
        "qags-cnndm": get_sample_scores("./baselines/eval_results/cnndm_srl_fullset_False_rouge.json"),
        "qags-xsum": get_sample_scores("./baselines/eval_results/xsum_srl_fullset_False_spacy.json"),
        "summeval": get_sample_scores("./baselines/eval_results/summeval_srl_fullset_False_rouge.json")
    }

    for dataset, gold in gold_scores.items():
        print(f"{dataset}, static")
        print(f"Pearson: {round(pearson(gold, static_scores[dataset]), 2)}, "
              f"Spearman: {round(spearman(gold, static_scores[dataset]), 2)}")

    for dataset, gold in gold_scores.items():
        print(f"{dataset}, dynamic")
        print(f"Pearson: {round(pearson(gold, dynamic_scores[dataset]), 2)}, "
              f"Spearman: {round(spearman(gold, dynamic_scores[dataset]), 2)}")

    for dataset, gold in gold_scores.items():
        pearson_p = permutation_test(gold, dynamic_scores[dataset], static_scores[dataset], pearson)
        spearman_p = permutation_test(gold, dynamic_scores[dataset], static_scores[dataset], spearman)

        print(f"{dataset} Pearson p-value of improvement: {pearson_p}")
        print(f"{dataset} Spearman p-value of improvement: {spearman_p}")

    # Reproducing significance levels between BARTScore, CoCo and SRLScore
    print("Testing for significance between groups of factuality metrics:")
    significance_between_methods()

    # Reproduce significance over baseline metrics
    print("Testing for significance between SRLScore and baseline scores:")
    improvement_over_baseline()

    # Results of Table 2
    print("Results of Table 2:")
    for sim_function in ["exact", "rouge", "spacy"]:
        openie = {
            "qags-cnndm": get_sample_scores(f"./baselines/eval_results/cnndm_srl_baseline_False_{sim_function}.json"),
            "qags-xsum": get_sample_scores(f"./baselines/eval_results/xsum_srl_baseline_False_{sim_function}.json"),
            "summeval": get_sample_scores(f"./baselines/eval_results/summeval_srl_baseline_False_{sim_function}.json")
        }

        base = {
            "qags-cnndm": get_sample_scores(f"./baselines/eval_results/cnndm_srl_fullset_False_{sim_function}.json"),
            "qags-xsum": get_sample_scores(f"./baselines/eval_results/xsum_srl_fullset_False_{sim_function}.json"),
            "summeval": get_sample_scores(f"./baselines/eval_results/summeval_srl_fullset_False_{sim_function}.json")
        }

        for dataset, gold in gold_scores.items():
            print(f"{dataset}, OpenIE {sim_function}")
            print(f"Pearson: {round(pearson(gold, openie[dataset]), 2)}, "
                  f"Spearman: {round(spearman(gold, openie[dataset]), 2)}")

        for dataset, gold in gold_scores.items():
            print(f"{dataset}, Base {sim_function}")
            print(f"Pearson: {round(pearson(gold, base[dataset]), 2)}, "
                  f"Spearman: {round(spearman(gold, base[dataset]), 2)}")

        for dataset, gold in gold_scores.items():
            pearson_p = permutation_test(gold, base[dataset], openie[dataset], pearson)
            spearman_p = permutation_test(gold, base[dataset], openie[dataset], spearman)

            print(f"{dataset} {sim_function} Pearson p-value of improvement: {pearson_p}")
            print(f"{dataset} {sim_function} Spearman p-value of improvement: {spearman_p}")

    # Improvements over exact matching in Table 2:
    print(f"Significance levels of various similarity functions over exact matches for SRL_base")
    for dataset in ["cnndm", "xsum", "summeval"]:
        exact = get_sample_scores(f"./baselines/eval_results/{dataset}_srl_fullset_False_exact.json")
        rouge = get_sample_scores(f"./baselines/eval_results/{dataset}_srl_fullset_False_rouge.json")
        spacy = get_sample_scores(f"./baselines/eval_results/{dataset}_srl_fullset_False_spacy.json")

        if dataset in ["cnndm", "xsum"]:
            gold = gold_scores[f"qags-{dataset}"]
        else:
            gold = gold_scores[dataset]
        pearson_p = permutation_test(gold, rouge, exact, pearson)
        spearman_p = permutation_test(gold, rouge, exact, spearman)
        print(f"{dataset} ROUGE:")
        print(f"Pearson: {pearson_p}")
        print(f"Spearman: {spearman_p}")

        pearson_p = permutation_test(gold, spacy, exact, pearson)
        spearman_p = permutation_test(gold, spacy, exact, spearman)
        print(f"{dataset} spaCy:")
        print(f"Pearson: {pearson_p}")
        print(f"Spearman: {spearman_p}")


    # Results of Table 4
    print("Results of Table 4:")
    goodrich_scores = {"qags-cnndm": get_sample_scores("./baselines/eval_results/cnndm_srl_goodrich.json"),
                       "qags-xsum": get_sample_scores("./baselines/eval_results/xsum_srl_goodrich.json"),
                       "summeval": get_sample_scores("./baselines/eval_results/summeval_srl_goodrich.json")}
    openie_scores = {"qags-cnndm": get_sample_scores("./baselines/eval_results/cnndm_srl_baseline_False_rouge.json"),
                     "qags-xsum": get_sample_scores("./baselines/eval_results/xsum_srl_baseline_False_spacy.json"),
                     "summeval": get_sample_scores("./baselines/eval_results/summeval_srl_baseline_False_rouge.json")}

    for dataset, gold in gold_scores.items():
        print(f"{dataset}, Goodrich")
        print(f"Pearson: {round(pearson(gold, goodrich_scores[dataset]), 2)}, "
              f"Spearman: {round(spearman(gold, goodrich_scores[dataset]), 2)}")

    for dataset, gold in gold_scores.items():
        print(f"{dataset}, OpenIE")
        print(f"Pearson: {round(pearson(gold, openie_scores[dataset]), 2)}, "
              f"Spearman: {round(spearman(gold, openie_scores[dataset]), 2)}")

    for dataset, gold in gold_scores.items():
        pearson_p = permutation_test(gold, openie_scores[dataset], goodrich_scores[dataset], pearson)
        spearman_p = permutation_test(gold, openie_scores[dataset], goodrich_scores[dataset], spearman)

        print(f"{dataset} Pearson p-value of improvement: {pearson_p}")
        print(f"{dataset} Spearman p-value of improvement: {spearman_p}")
