"""
This script evaluates the test set of dataset. 
"""

from typing import List
from statistics import mean
import random
import sys
import os

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import time

from calculate_faithful_score import CalculateFactualScore


def get_dailymail_samples(sample_num: int = 15, split: str = "train",) -> List[dict]:
    """
    This script randomly select articles from the cnn_dailymail dataset.
    """
    data = load_dataset("cnn_dailymail", "3.0.0")[split]

    # my_file = open(os.path.join(sys.path[0], "dailymail_faithful_examples.txt"), "r")

    # # reading the file
    # # replacing end splitting the text
    # # when newline ('\n') is seen.
    # data_into_list = my_file.read().split("\n")
    # my_file.close()

    # if do_filtering == True:
    #     data = data.filter(lambda example: example["id"] in data_into_list)
    # else:
    #     data = data.filter(lambda example: example["id"] not in data_into_list)

    # print(len(data))

    random.seed(42)
    samples: List[dict] = random.sample(list(data), sample_num)
    short_samples = [
        sample for sample in samples if len(sample["article"].split(" ")) < 350
    ]

    return short_samples


def get_xsum_samples(
    do_filtering: bool, sample_num: int = 15, split: str = "train"
) -> List[dict]:
    """
    This script randomly select sample articles from the cnn_dailymail dataset.
    """
    data = load_dataset("xsum")[split]
    if do_filtering == True:
        my_file = open(os.path.join(sys.path[0], "xsum_faithful_examples.txt"), "r")

        # reading the file
        # replacing end splitting the text
        # when newline ('\n') is seen.
        data_into_list = my_file.read().split("\n")
        my_file.close()

        data = data.filter(lambda example: example["id"] in data_into_list)

    print(len(data))
    random.seed(42)
    samples: List[dict] = random.sample(list(data), sample_num)
    # short_samples = [
    #     sample for sample in samples if len(sample["document"].split(" ")) < 350
    # ]
    print(samples)

    return samples


if __name__ == "__main__":
    calcu = CalculateFactualScore("rouge")
    samples = get_dailymail_samples()
    scores = [
        calcu.calculate_factual_score(sample["article"], sample["highlights"])
        for sample in samples
    ]

    # remove summary scores where no tuples can be extracted from summary
    scores = [score for score in scores if score != -1]
    print(scores)

