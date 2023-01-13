"""
This script 
1. evaluates if a summary is factual consistent. It receives the summary text and the source text 
and generates a factual consistency score of the summary.
"""

from typing import List
from numbers import Number
from statistics import mean
import random
import warnings

from tqdm import tqdm
from datasets import load_dataset
import extract_tuples as et
import tuple_comparison as tc


class ModelConfigurator:
    def __init__(self, coref_solver, func_words_remover, string_comparison_method):
        self.coref_solver = coref_solver
        self.func_words_remover = func_words_remover
        self.string_comparison_method = string_comparison_method

    def calculate_summary_score(
        self, source_text: str, generated_summary: str
    ) -> float:
        """
        This function builds a SRL tuple database for source text and generated summary, and
        calculates the consistency score of the generated summary.
        """

        if len(source_text.split(" ")) >= 450:
            warnings.warn(
                "Source text might exceed input length limit for SRL extraction!"
            )
        if len(generated_summary.split(" ")) >= 450:
            warnings.warn(
                "Generated summary might exceed input length limit for SRL extraction!"
            )

        print("-----extract tuples from summary text-----")
        generated_tuples: List[tuple] = et.extract_tuples(
            generated_summary, self.coref_solver, self.func_words_remover
        )
        print("---generated tuples are---:", generated_tuples)

        print("-----extract tuples from source text-----")
        source_tuples: List[tuple] = et.extract_tuples(
            source_text, self.coref_solver, self.func_words_remover
        )
        print("---source tuples are---：", source_tuples)

        tuple_final_scores = []
        for generated_tup in tqdm(generated_tuples, desc="calculate_summary_score"):
            tuple_final_score = tc.compare_tuple_with_relevant_tuples(
                source_tuples, generated_tup, self.string_comparison_method,
            )
            tuple_final_scores.append(tuple_final_score)
        print("---Score of each tuple in generated text：", tuple_final_scores)

        summary_score = round(mean(tuple_final_scores), 2)
        return summary_score

    def evaluate_dailymail(
        self, sample_num: Number = 15, split: str = "train"
    ) -> List[float]:
        """
        This script randomly select sample articles from the cnn_dailymail dataset.
        Then it applys the calculate_summary_score function many times for each random sample article.
        """
        data = load_dataset("cnn_dailymail", "3.0.0")[split]
        random.seed(42)
        samples: List[dict] = random.sample(list(data), sample_num)
        short_samples = [
            sample for sample in samples if len(sample["article"].split(" ")) < 400
        ]
        print(short_samples)

        scores = [
            self.calculate_summary_score(sample["article"], sample["highlights"])
            for sample in short_samples
        ]
        print(scores)


if __name__ == "__main__":
    model = ModelConfigurator(
        coref_solver=False, func_words_remover=False, string_comparison_method="exact",
    )
    model.evaluate_dailymail()

    source_text = "National Archives Yes, it’s that time again, folks. It’s the first Friday of the month, when for one ever-so-brief moment the interests of Wall Street, Washington and Main Street are all aligned on one thing: Jobs. A fresh update on the U.S. employment situation for January hits the wires at 8:30 a.m. New York time offering one of the most important snapshots on how the economy fared during the previous month. Expectations are for 203,000 new jobs to be created, according to economists polled by Dow Jones Newswires, compared to 227,000 jobs added in February. The unemployment rate is expected to hold steady at 8.3%. Here at MarketBeat HQ, we’ll be offering color commentary before and after the data crosses the wires. Feel free to weigh-in yourself, via the comments section. And while you’re here, why don’t you sign up to follow us on Twitter. Enjoy the show. ||||| Employers pulled back sharply on hiring last month, a reminder that the U.S. economy may not be growing fast enough to sustain robust job growth. The unemployment rate dipped, but mostly because more Americans stopped looking for work. The Labor Department says the economy added 120,000 jobs in March, down from more than 200,000 in each of the previous three months. The unemployment rate fell to 8.2 percent, the lowest since January 2009. The rate dropped because fewer people searched for jobs. The official unemployment tally only includes those seeking work. The economy has added 858,000 jobs since December _ the best four months of hiring in two years. But Federal Reserve Chairman Ben Bernanke has cautioned that the current hiring pace is unlikely to continue without more consumer spending."
    gold_summary = "The unemployment rate dropped to 8.2% last month, but the economy only added 120,000 jobs, when 203,000 new jobs had been predicted, according to today's jobs report. Reaction on the Wall Street Journal's MarketBeat Blog was swift: 'Woah!!! Bad number.' The unemployment rate, however, is better news; it had been expected to hold steady at 8.3%. But the AP notes that the dip is mostly due to more Americans giving up on seeking employment."
    print(model.calculate_summary_score(source_text, gold_summary))
