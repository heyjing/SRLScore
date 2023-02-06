"""
This is a script that implements different string similarity comparison methods. 
"""


from rouge_score import rouge_scorer
from jiwer import wer
import extract_tuples as et


class StringSimilarityMethods:
    method: str

    def __init__(self, method: str):
        self.method = method

    def exact_match_string_similarity(self, source_str, summary_str):
        if source_str and summary_str and source_str == summary_str:
            similarity_score = 1.0
        else:
            similarity_score = 0
        return similarity_score

    def spacy_string_similarity(self, source_str, summary_str):
        if source_str and summary_str:
            nlp = et.load_spacy_model()
            doc1 = nlp(source_str)
            doc2 = nlp(summary_str)
            similarity_score = round(doc1.similarity(doc2), 2)
        else:
            similarity_score = 0
        return similarity_score

    def rouge_precision_string_similarity(self, source_str, summary_str):
        if source_str and summary_str:
            scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
            scores = scorer.score(source_str, summary_str)
            precision = scores["rouge1"][0]
            similarity_score = round(precision, 2)
        else:
            similarity_score = 0
        return similarity_score

    def word_error_rate_string_similarity(self, source_str, summary_str):
        if source_str and summary_str:
            error = wer(source_str, summary_str)
            similarity_score = round(error, 2)
        else:
            similarity_score = 0
        return similarity_score

    def calculate(self, source_str, summary_str) -> float:
        methods = {
            "exact": self.exact_match_string_similarity,
            "spacy": self.spacy_string_similarity,
            "rouge": self.rouge_precision_string_similarity,
            "WER": self.word_error_rate_string_similarity,
        }
        return methods.get(self.method)(source_str, summary_str)


if __name__ == "__main__":

    compare = StringSimilarityMethods("rouge")

    print(compare.calculate(None, "a"))
    print(compare.calculate("one gift", "gift"))
    print(compare.calculate("apple good", "apple is good"))
    print(compare.calculate(None, "gift"))
    print(compare.calculate(None, None))
    print(compare.calculate("mueller", "robert mueller"))

