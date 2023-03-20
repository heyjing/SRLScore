"""
Script to compute the correlation of scores between CoCo and human ratings.
"""

from utils import calculate_correlation_score
import json

import numpy as np

if __name__ == '__main__':
    for dataset in ["qags-cnndm", "qags-xsum", "summeval"]:
        with open(f"./coco_scores/{dataset}-gold.json", "r") as f:
            gold_labels = json.load(f)
        gold_labels = gold_labels["human_annotated_scores"]
        for method in ["span", "sent"]:
            with open(f"./coco_scores/{dataset}-{method}-scores.txt", "r") as f:
                scores = list(f)

            scores = [float(score) for score in scores]

            print(f"Dataset: {dataset}; Variant: CoCo-{method}")
            calculate_correlation_score(scores, gold_labels)

        rng = np.random.default_rng(seed=545454)

        if "qags" in dataset:
            random_baseline = rng.uniform(low=0.0, high=1.0000001, size=len(gold_labels)).tolist()
        else:
            random_baseline = rng.integers(low=0, high=6, size=len(gold_labels)).tolist()
        print(f"Random baseline for {dataset}:")
        calculate_correlation_score(random_baseline, gold_labels)