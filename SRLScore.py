from typing import List, Tuple, Optional
from statistics import mean

from itertools import chain
import numpy as np
from tqdm import tqdm

from tuple_comparison import StringSimilarityMethods
from processor import Processor


class SRLScore:
    string_comparison_method: str
    do_coref: bool
    weights: List[float]

    def __init__(
        self,
        string_comparison_method: str = "rouge",
        do_coref: bool = False,
        weights: List[float] = (1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7),
    ):
        if string_comparison_method not in ["rouge", "spacy", "exact"]:
            raise ValueError(f"String comparison method for SRLScore must be either one of "
                             f"'rouge', 'spacy', or 'exact'!")
        self.string_comparison_method = string_comparison_method
        self.do_coref = do_coref
        if len(weights) != 7:
            raise ValueError("Need to specify weights for all seven attributes! "
                             "You may want to default to 0 weights for unspecified attributes.")
        self.weights = weights

    def _compare_two_tuples(self, source_tuple: tuple, generated_tuple: tuple) -> float:
        """
        This function calculates consistency score of two tuples.
        """
        indic = np.array([1 if x else 0 for x in generated_tuple])

        pairwise_similarity_scores = [
            StringSimilarityMethods(self.string_comparison_method).calculate(
                source_str, generated_str
            )
            for source_str, generated_str in zip(source_tuple, generated_tuple)
        ]

        normalized_weight = 1 / (np.sum(indic * self.weights))
        consistency_score = normalized_weight * np.sum(
            [indic * pairwise_similarity_scores * self.weights]
        )
        return consistency_score

    def _compare_tuple_with_relevant_tuples(
        self, source_tuples: List[tuple], generated_tup: tuple
    ) -> float:
        """
        This function compares a generated tuple with all of its relevant tuples from the input text and
        takes the maximum attained score as the final prediction.
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

            # save loops in case the max final score of a tuple achieved earlier
            if tuple_final_score == 1:
                break

        return tuple_final_score

    def calculate_score(self, source_text: str, generated_text: str) -> float:
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
            summary_score = []

            for tup_clusters in generated_summary_tuples:
                tup_clusters_score = []
                for tup in tqdm(tup_clusters, desc="calculate_summary_score"):
                    tup_score = self._compare_tuple_with_relevant_tuples(
                        source_tuples, tup
                    )
                    tup_clusters_score.append(tup_score)

                tup_clusters_final_score = mean(tup_clusters_score)
                print(
                    f"tup_clusters_score is {tup_clusters_score}; tup_clusters_final_score is {tup_clusters_final_score}"
                )
                summary_score.append(tup_clusters_final_score)
            print("summary score: ", summary_score, "------a sample is fertig----")
            return mean(summary_score)
        else:
            return 0

    def _check_two_tuples_relevant_goodrich_inspired(
        self, tup1: tuple, tup2: tuple
    ) -> bool:
        """
        return True if the agent and relation of two tuples are the same
        """
        return True if tup1[0] == tup2[0] and tup1[2] == tup2[2] else False

    def _check_if_tuple_has_relevant_tuples_goodrich_inspired(
        self, relevant_tuples: List[tuple], tup: tuple
    ) -> bool:
        """
        for a source tuple, check if a generated tuple with the same agent and relation exists.
        for a generated tuple, check if a source tuple with the same agent and relation exists.
        """
        for relevant_tup in relevant_tuples:
            validity = self._check_two_tuples_relevant_goodrich_inspired(
                tup, relevant_tup
            )
            if validity:
                break
        return validity

    def _filter_tuples_goodrich(
        self, source_tuples: List[tuple], generated_tuples: List[tuple]
    ) -> Tuple[List[tuple], List[tuple]]:
        """
        filter list of tuples according to Goodrich (2021): Assessing The Factual Accuracy of Generated Text;
        e.g. bool_values_source_tuples = [True, False, False, True] means the first and last tuple have relevant tuples 
        and these two tuples will build a new list of source tuples. 
        """
        bool_values_source_tuples = [
            self._check_if_tuple_has_relevant_tuples_goodrich_inspired(
                generated_tuples, source_tup
            )
            for source_tup in source_tuples
        ]
        bool_values_generated_tuples = [
            self._check_if_tuple_has_relevant_tuples_goodrich_inspired(
                source_tuples, generated_tup
            )
            for generated_tup in generated_tuples
        ]
        filtered_source_tuples = [
            source_tup
            for (source_tup, bool_index) in zip(
                source_tuples, bool_values_source_tuples
            )
            if bool_index
        ]
        filtered_generated_tuples = [
            generated_tup
            for (generated_tup, bool_index) in zip(
                generated_tuples, bool_values_generated_tuples
            )
            if bool_index
        ]
        return filtered_source_tuples, filtered_generated_tuples

    def goodrich_inspired_score(self, source_text: str, generated_text: str) -> float:
        proc = Processor(do_coref=False)

        source_tuples: List[List[tuple]] = proc.process_text(source_text)
        # Flattening a list of lists
        source_tuples: List[tuple] = list(chain(*source_tuples))

        generated_summary_tuples: List[List[tuple]] = proc.process_text(generated_text)
        generated_summary_tuples: List[tuple] = list(chain(*generated_summary_tuples))

        if generated_summary_tuples != []:
            (
                filtered_source_tuples,
                filtered_generated_tuples,
            ) = self._filter_tuples_goodrich(source_tuples, generated_summary_tuples)

            # because of weights = [1 / 3, 0, 1 / 3, 1 / 3, 0, 0, 0]
            filtered_source_tuples = [
                (tup[0], tup[2], tup[3]) for tup in filtered_source_tuples
            ]
            filtered_generated_tuples = [
                (tup[0], tup[2], tup[3]) for tup in filtered_generated_tuples
            ]

            intersection = list(
                set(filtered_source_tuples) & set(filtered_generated_tuples)
            )

            if filtered_generated_tuples != []:
                accuracy_score = len(intersection) / len(filtered_generated_tuples)
            else:
                accuracy_score = 0
        else:
            accuracy_score = 0
        return accuracy_score


if __name__ == "__main__":

    # Initialize with default weights
    scorer = SRLScore("rouge", False, None)
    score = scorer._compare_two_tuples(
        ("Peter", None, "send", "one gift", " to his sister", None, None),
        ("Peter", None, "send", "a gift", None, None, None),
    )
    print(score)
