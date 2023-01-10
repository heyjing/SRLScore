"""
This is a script that implements the tuple comparison algorithm. It receives the summary text and the source text 
and generates a factual consistency score of the summary.
"""


from typing import List, Optional
from statistics import mean
import warnings
import numpy as np
from tqdm import tqdm
import extract_tuples as et


def calculate_string_similarity(
    source_str: Optional[str], summary_str: Optional[str]
) -> float:
    if source_str and summary_str:
        # nlp = et.load_language_model()
        # doc1 = nlp(source_str)
        # doc2 = nlp(summary_str)
        # similarity_score = doc1.similarity(doc2)
        # ----------------------------------
        if source_str == summary_str:
            similarity_score = 1.0
        else:
            # rouge_scores = Rouge().get_scores(summary_str, source_str)
            # similarity_score = rouge_scores[0]["rouge-1"]["p"]
            similarity_score = 0

    else:
        similarity_score = 0
    return similarity_score


def compare_tuples(source_tuple: tuple, generated_tuple: tuple) -> float:
    """
    This function calculates consistency score of two tuples.
    """
    # hard-coded values for the semantic roles
    weights = [1 / 3, 0, 1 / 3, 1 / 3, 0, 0, 0]
    indic = np.array([1 if x else 0 for x in generated_tuple])
    pairwise_similarity_scores = [
        calculate_string_similarity(source_str, generated_str)
        for source_str, generated_str in zip(source_tuple, generated_tuple)
    ]

    normalized_weight = 1 / (np.sum(indic * weights))
    consistency_score = normalized_weight * np.sum(
        [indic * pairwise_similarity_scores * weights]
    )
    return round(consistency_score, 2)


def determine_tuple_final_score(
    relevant_source_tuples: List[tuple], generated_tup: tuple
) -> float:
    """
    This function compares a generated tuple with all of its relevant tuples from source text and takes the max as final score.
    """
    tuple_final_score = 0
    for relevant_source_tup in relevant_source_tuples:
        consistency_score = compare_tuples(
            source_tuple=relevant_source_tup, generated_tuple=generated_tup
        )
        if consistency_score >= tuple_final_score:
            tuple_final_score = consistency_score

    return tuple_final_score


def calculate_summary_score(source_text: str, generated_summary: str) -> float:
    """
    This function builds a SRL tuple database for source text and generated summary, and 
    calculates the consistency score of the generated summary.
    """

    if len(source_text.split(" ")) >= 450:
        warnings.warn("Source text might exceed input length limit for SRL extraction!")
    if len(generated_summary.split(" ")) >= 450:
        warnings.warn(
            "Generated summary might exceed input length limit for SRL extraction!"
        )

    print("-----extract tuples from summary text-----")
    generated_tuples: List[tuple] = et.extract_tuples(generated_summary)
    print("---generated tuples are---:", generated_tuples)

    print("-----extract tuples from source text-----")
    source_tuples: List[tuple] = et.extract_tuples(source_text)
    print("---source tuples are---：", source_tuples)

    tuple_final_scores = []
    for generated_tup in tqdm(generated_tuples, desc="calculate_summary_score"):
        tuple_final_score = determine_tuple_final_score(source_tuples, generated_tup)
        tuple_final_scores.append(tuple_final_score)
    print("---Score of each tuple in generated text：", tuple_final_scores)
    return round(mean(tuple_final_scores), 2)


if __name__ == "__main__":
    print(
        compare_tuples(
            ("Hans", None, "bought", "a vase", "sister", None, None),
            ("Hans", None, "send", "a vase", None, None, None),
        )
    )

    source_text = "National Archives Yes, it’s that time again, folks. It’s the first Friday of the month, when for one ever-so-brief moment the interests of Wall Street, Washington and Main Street are all aligned on one thing: Jobs. A fresh update on the U.S. employment situation for January hits the wires at 8:30 a.m. New York time offering one of the most important snapshots on how the economy fared during the previous month. Expectations are for 203,000 new jobs to be created, according to economists polled by Dow Jones Newswires, compared to 227,000 jobs added in February. The unemployment rate is expected to hold steady at 8.3%. Here at MarketBeat HQ, we’ll be offering color commentary before and after the data crosses the wires. Feel free to weigh-in yourself, via the comments section. And while you’re here, why don’t you sign up to follow us on Twitter. Enjoy the show. ||||| Employers pulled back sharply on hiring last month, a reminder that the U.S. economy may not be growing fast enough to sustain robust job growth. The unemployment rate dipped, but mostly because more Americans stopped looking for work. The Labor Department says the economy added 120,000 jobs in March, down from more than 200,000 in each of the previous three months. The unemployment rate fell to 8.2 percent, the lowest since January 2009. The rate dropped because fewer people searched for jobs. The official unemployment tally only includes those seeking work. The economy has added 858,000 jobs since December _ the best four months of hiring in two years. But Federal Reserve Chairman Ben Bernanke has cautioned that the current hiring pace is unlikely to continue without more consumer spending."
    gold_summary = "The unemployment rate dropped to 8.2% last month, but the economy only added 120,000 jobs, when 203,000 new jobs had been predicted, according to today's jobs report. Reaction on the Wall Street Journal's MarketBeat Blog was swift: 'Woah!!! Bad number.' The unemployment rate, however, is better news; it had been expected to hold steady at 8.3%. But the AP notes that the dip is mostly due to more Americans giving up on seeking employment."
    print(calculate_summary_score(source_text, gold_summary))
