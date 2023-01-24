"""
This script evaluates the test set of dataset. 
"""

from typing import List
from statistics import mean
import random
import sys
import os

from datasets import load_dataset

import extract_tuples as et


def get_dailymail_samples(
    do_filtering: bool, sample_num: int = 15, split: str = "train",
) -> List[dict]:
    """
    This script randomly select articles from the cnn_dailymail dataset.
    """
    data = load_dataset("cnn_dailymail", "3.0.0")[split]

    if do_filtering == True:

        my_file = open(
            os.path.join(sys.path[0], "dailymail_faithful_examples.txt"), "r"
        )

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
    #     sample for sample in samples if len(sample["article"].split(" ")) < 350
    # ]

    print(samples)

    return samples


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
    model = et.ModelConfigurator(
        use_coref_solver=False,
        use_func_words_remover=True,
        string_comparison_method="spacy",
    )
    samples = get_dailymail_samples(do_filtering=False)

    scores = [
        model.calculate_summary_score(sample["article"], sample["highlights"])
        for sample in samples
    ]
    print("number of samples: ", len(samples))
    print("scores: ", scores)
    print("mean score: ", mean(scores))

    examples = "Peter gave his book to his sister Mary yesterday in Berlin. She is a young girl. He wants to make her happy"
    # examples = "Palmer College of Chiropractic in Davenport must readmit Aaron Cannon and allow him to use an assistant to interpret X-rays and other medical images"
    # examples = "5-2 decision forces school to readmit student Aaron Cannon and allow him to use X-rays and other medical images"
    # examples = "In the twilight, I am horrified to see a wolf howling at the end of the garden."
    # examples = "The Seasiders have been struggling to recruit players to fulfil their opening fixture but on Wednesday they completed a deal for Ishmael Miller and John Lundstram has joined on loan from Everton."
    # examples = "John can't keep up with Mary 's rapid mood swings"
    # examples = "South Korea has opened its market to foreign cigarettes."
    # examples = "I can't do it"
    # examples = "Paul has bought an apple for Anna. She is very happy."
    # examples = "National Archives Yes, it’s that time again, folks. It’s the first Friday of the month, when for one ever-so-brief moment the interests of Wall Street, Washington and Main Street are all aligned on one thing: Jobs. A fresh update on the U.S. employment situation for January hits the wires at 8:30 a.m. New York time offering one of the most important snapshots on how the economy fared during the previous month. Expectations are for 203,000 new jobs to be created, according to economists polled by Dow Jones Newswires, compared to 227,000 jobs added in February. The unemployment rate is expected to hold steady at 8.3%. Here at MarketBeat HQ, we’ll be offering color commentary before and after the data crosses the wires. Feel free to weigh-in yourself, via the comments section. And while you’re here, why don’t you sign up to follow us on Twitter. Enjoy the show. ||||| Employers pulled back sharply on hiring last month, a reminder that the U.S. economy may not be growing fast enough to sustain robust job growth. The unemployment rate dipped, but mostly because more Americans stopped looking for work. The Labor Department says the economy added 120,000 jobs in March, down from more than 200,000 in each of the previous three months. The unemployment rate fell to 8.2 percent, the lowest since January 2009. The rate dropped because fewer people searched for jobs. The official unemployment tally only includes those seeking work. The economy has added 858,000 jobs since December _ the best four months of hiring in two years. But Federal Reserve Chairman Ben Bernanke has cautioned that the current hiring pace is unlikely to continue without more consumer spending."
    # examples = "The unemployment rate dropped to 8.2% last month, but the economy only added 120,000 jobs, when 203,000 new jobs had been predicted, according to today's jobs report. Reaction on the Wall Street Journal's MarketBeat Blog was swift: 'Woah!!! Bad number.' The unemployment rate, however, is better news; it had been expected to hold steady at 8.3%. But the AP notes that the dip is mostly due to more Americans giving up on seeking employment."

    # truth_tuples: List[tuple] = model.extract_tuples(examples)
    # print(truth_tuples)
