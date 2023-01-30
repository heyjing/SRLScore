"""
This is a script that implements different string similarity comparison methods. 
"""


from typing import Optional
from rouge_score import rouge_scorer
from jiwer import wer
import extract_tuples as et


class StringSimilarityMethods:
    source_str: Optional[str]
    summary_str: Optional[str]
    method: str

    def __init__(
        self, source_str: Optional[str], summary_str: Optional[str], method: str
    ):
        self.source_str = source_str
        self.summary_str = summary_str
        self.method = method

    def exact_match_string_similarity(self):
        if self.source_str and self.summary_str and self.source_str == self.summary_str:
            similarity_score = 1.0
        else:
            similarity_score = 0
        return similarity_score

    def spacy_string_similarity(self):
        if self.source_str and self.summary_str:
            nlp = et.load_spacy_model()
            doc1 = nlp(self.source_str)
            doc2 = nlp(self.summary_str)
            similarity_score = round(doc1.similarity(doc2), 2)
        else:
            similarity_score = 0
        return similarity_score

    def rouge_precision_string_similarity(self):
        if self.source_str and self.summary_str:
            scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
            scores = scorer.score(self.source_str, self.summary_str)
            precision = scores["rouge1"][0]
            similarity_score = round(precision, 2)
        else:
            similarity_score = 0
        return similarity_score

    def character_error_rate_string_similarity(self):
        if self.source_str and self.summary_str:
            error = wer(self.source_str, self.summary_str)
            similarity_score = round(error, 2)
        else:
            similarity_score = 0
        return similarity_score

    def calculate(self) -> float:
        methods = {
            "exact": self.exact_match_string_similarity,
            "spacy": self.spacy_string_similarity,
            "rouge": self.rouge_precision_string_similarity,
            "CER": self.character_error_rate_string_similarity,
        }
        return methods.get(self.method)()


if __name__ == "__main__":
    # compare = StringSimilarityMethods("apple is good", "Apple good", "WER")
    # compare = StringSimilarityMethods("apple good", "apple is good", "WER")
    # compare = StringSimilarityMethods("a", None, "WER")
    # compare = StringSimilarityMethods(None, "a", "WER")
    # compare = StringSimilarityMethods(None, None, "WER")
    compare = StringSimilarityMethods("one gift", "a gift", "rouge")
    # compare = StringSimilarityMethods("mueller", "robert mueller", "rouge")
    # compare = StringSimilarityMethods("mueller", "robert mueller", "spacy")
    # compare = StringSimilarityMethods("alligator", "gator", "rouge")

    print(compare.calculate())
