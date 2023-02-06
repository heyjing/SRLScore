from typing import List
from statistics import mean

from itertools import chain
import numpy as np
from tqdm import tqdm

from tuple_comparison import StringSimilarityMethods
from processor import Processor


class CalculateFactualScore:
    string_comparison_method: str
    do_coref: bool

    def __init__(self, string_comparison_method: str, do_coref: bool):
        self.string_comparison_method = string_comparison_method
        self.do_coref = do_coref

    def _compare_two_tuples(self, source_tuple: tuple, generated_tuple: tuple) -> float:
        """
        This function calculates consistency score of two tuples.
        """
        # hard-coded values for the semantic roles
        weights = [1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7]
        indic = np.array([1 if x else 0 for x in generated_tuple])

        pairwise_similarity_scores = [
            StringSimilarityMethods(self.string_comparison_method).calculate(
                source_str, generated_str
            )
            for source_str, generated_str in zip(source_tuple, generated_tuple)
        ]

        normalized_weight = 1 / (np.sum(indic * weights))
        consistency_score = normalized_weight * np.sum(
            [indic * pairwise_similarity_scores * weights]
        )
        return round(consistency_score, 2)

    def _compare_tuple_with_relevant_tuples(
        self, source_tuples: List[tuple], generated_tup: tuple
    ) -> float:
        """
        This function compares a generated tuple with all of its relevant tuples from source text and takes the max as final score.
        """
        tuple_final_score = 0
        for source_tup in source_tuples:
            consistency_score = self._compare_two_tuples(
                source_tuple=source_tup, generated_tuple=generated_tup,
            )
            if consistency_score > tuple_final_score:
                tuple_final_score = consistency_score
                print("generated tup: ", generated_tup)
                print("relevant_source_tup: ", source_tup)

        return tuple_final_score

    def calculate_factual_score(self, source_text: str, generated_text: str) -> float:
        """
        This function calculates the consistency score of the generated summary.
        """
        proc = Processor(self.do_coref)

        source_tuples: List[List[tuple]] = proc.process_text(source_text)
        # Flattening a list of lists
        source_tuples: List[tuple] = list(chain(*source_tuples))

        generated_summary_tuples: List[List[tuple]] = proc.process_text(generated_text)
        print("source_tuples: ", source_tuples)
        print("generated_summary_tuples: ", generated_summary_tuples)

        if generated_summary_tuples != []:
            Summary_score = []

            for tup_clusters in generated_summary_tuples:
                tup_clusters_score = []
                for tup in tqdm(tup_clusters, desc="calculate_summary_score"):
                    tup_score = self._compare_tuple_with_relevant_tuples(
                        source_tuples, tup
                    )
                    tup_clusters_score.append(tup_score)
                print("---tup_clusters_score：", tup_clusters_score)

                tup_clusters_final_score = round(max(tup_clusters_score), 2)
                print("---tup_clusters_final_score: ", tup_clusters_final_score)
                Summary_score.append(tup_clusters_final_score)
            print("summary score: ", Summary_score, "------a sample is fertig----")
            return round(mean(Summary_score), 2)
        else:
            return -1


if __name__ == "__main__":

    calcu = CalculateFactualScore("rouge", False)
    score = calcu._compare_two_tuples(
        ("Peter", None, "send", "one gift", " to his sister", None, None),
        ("Peter", None, "send", "a gift", None, None, None),
    )
    print(score)
