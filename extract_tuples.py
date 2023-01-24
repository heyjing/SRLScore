"""
This is a script that extracts facts from texts and again represents them 
using semantic role labels. For an input text, the output will be a fact database: List[tuple].
The function calculate_summary_score() in the ModelConfigurator class returns a faithfulness score of 
the corresponding summary against its source text.
"""


from typing import List
from functools import lru_cache
from statistics import mean
import warnings
import re

from allennlp.predictors.predictor import Predictor
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import tokenize
from tqdm import tqdm
import numpy as np
import spacy

import tuple_comparison as tc


class SRLTuple:
    def __init__(
        self,
        agent=None,
        negation=None,
        relation=None,
        patient=None,
        recipient=None,
        time=None,
        location=None,
    ):
        self.agent = agent
        self.negation = negation
        self.relation = relation
        self.patient = patient
        self.recipient = recipient
        self.time = time
        self.location = location

    def format_tuple(self) -> tuple:
        return tuple(
            [
                self.agent,
                self.negation,
                self.relation,
                self.patient,
                self.recipient,
                self.time,
                self.location,
            ]
        )


@lru_cache(maxsize=1)
def load_spacy_model():
    nlp = spacy.load("en_core_web_lg")
    return nlp


@lru_cache(maxsize=1)
def load_srl_model():
    srl_model = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz",
        cuda_device=0,
    )
    return srl_model


@lru_cache(maxsize=1)
def load_coref_model():
    coref_model = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz",
        cuda_device=1,
    )
    return coref_model


class ModelConfigurator:
    use_coref_solver: bool
    use_func_words_remover: bool
    string_comparison_method: str

    def __init__(
        self,
        use_coref_solver: bool,
        use_func_words_remover: bool,
        string_comparison_method: str,
    ):
        self.use_coref_solver = use_coref_solver
        self.use_func_words_remover = use_func_words_remover
        self.string_comparison_method = string_comparison_method

    def text_preprocessing(self, text: str) -> dict:
        """
        This script processes texts before extracting tuples, like coreference resolution and annotate semantic roles.
        """

        # step 1: AllenNLP coreference resolution
        if self.use_coref_solver == True:
            print("-----text preprocessing: coreference resolution-----")
            coref_model = load_coref_model()
            text = coref_model.coref_resolved(text)

        # step 2: AllenNLP SRL
        print("-----text preprocessing: SRL-----")
        srl_model = load_srl_model()
        srl_annotated_text = srl_model.predict(sentence=text)

        return srl_annotated_text

    def text_postprocessing(self, string: str) -> str:
        """
        Strings in extracted tuples may be long.
        This function removes leading ADP(prepositions like in, to, auf etc.) and DET(determiner like this, that, a, an, diese etc.) 
        tags for each tuple element of string type, so that the tuples only contain the most important
        semantic information. The SPACY POS Tags List is the same for different languages. 
        """
        if self.use_func_words_remover:
            if len(string.split()) == 1:
                return string
            else:
                nlp = load_spacy_model()
                doc = nlp(string)
                for token in doc:
                    if token.pos_ != "ADP" and token.pos_ != "DET":
                        return string[token.idx :].strip()
                else:
                    return string

                # for i in range(len(doc)):
                #     if doc[i].pos_ != "ADP" and doc[i].pos_ != "DET":
                #         break
                #     else:
                #         string = string[len(doc[i]) :].lstrip()
                # return string
        else:
            return string

    def extract_tuples(self, text: str) -> List[tuple]:
        """
        This function takes an text as input and returns a list of extracted SRL truth tuples.

        """
        annotated_data: dict = self.text_preprocessing(text)
        # build an empty tuple database
        tuple_database = []

        for i in tqdm(range(len(annotated_data["verbs"])), desc="Extracting tuples"):
            res = re.findall(r"\[.*?\]", annotated_data["verbs"][i]["description"])
            # EXAMPLE res: ['[ARG0: I]', '[V: see]', '[ARG1: a wolf howling]', '[ARGM-TMP: at the end of the garden]']
            nr_of_roles = 0
            extracted_SRLTuple = SRLTuple()
            for j in res:
                former = j[j.find("[") + 1 : j.find(":")]
                # The strings in the extracted tuples are all lowercase
                latter = j[j.find(" ") + 1 : j.find("]")].casefold()
                if former == "ARG0":
                    extracted_SRLTuple.agent = self.text_postprocessing(latter)
                    nr_of_roles += 1
                    continue
                if former == "ARGM-NEG":
                    extracted_SRLTuple.negation = latter
                    nr_of_roles += 1
                    continue
                if former == "V":
                    extracted_SRLTuple.relation = WordNetLemmatizer().lemmatize(
                        latter, "v"
                    )
                    nr_of_roles += 1
                    continue
                if former == "ARG1":
                    extracted_SRLTuple.patient = self.text_postprocessing(latter)
                    nr_of_roles += 1
                    continue
                if former == "ARG2":
                    extracted_SRLTuple.recipient = self.text_postprocessing(latter)
                    nr_of_roles += 1
                    continue
                if former == "ARGM-TMP":
                    extracted_SRLTuple.time = self.text_postprocessing(latter)
                    nr_of_roles += 1
                    continue
                if former == "ARGM-LOC":
                    extracted_SRLTuple.location = self.text_postprocessing(latter)
                    nr_of_roles += 1
                    continue
            # Only tuples with at least two roles will be added to the tuple database of an article
            if nr_of_roles >= 2:
                tuple_database.append(extracted_SRLTuple.format_tuple())
        return tuple_database

    def compare_two_tuples(self, source_tuple: tuple, generated_tuple: tuple) -> float:
        """
        This function calculates consistency score of two tuples.
        """
        # hard-coded values for the semantic roles
        weights = [1 / 3, 0, 1 / 3, 1 / 3, 0, 0, 0]
        indic = np.array([1 if x else 0 for x in generated_tuple])

        pairwise_similarity_scores = [
            tc.StringSimilarityMethods(
                source_str, generated_str, self.string_comparison_method
            ).calculate_string_similarity()
            for source_str, generated_str in zip(source_tuple, generated_tuple)
        ]

        normalized_weight = 1 / (np.sum(indic * weights))
        consistency_score = normalized_weight * np.sum(
            [indic * pairwise_similarity_scores * weights]
        )
        return round(consistency_score, 2)

    def compare_tuple_with_relevant_tuples(
        self, relevant_source_tuples: List[tuple], generated_tup: tuple
    ) -> float:
        """
        This function compares a generated tuple with all of its relevant tuples from source text and takes the max as final score.
        """
        tuple_final_score = 0
        for relevant_source_tup in relevant_source_tuples:
            consistency_score = self.compare_two_tuples(
                source_tuple=relevant_source_tup, generated_tuple=generated_tup,
            )
            if consistency_score > tuple_final_score:
                tuple_final_score = consistency_score
                print("generated tup: ", generated_tup)
                print("relevant_source_tup: ", relevant_source_tup)

        return tuple_final_score

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
        summary_texts: List[str] = tokenize.sent_tokenize(generated_summary)
        summary_tuples = []
        for sentence in summary_texts:
            sentence_tuples: List[tuple] = self.extract_tuples(sentence)
            summary_tuples.append(sentence_tuples)
        summary_tuples = [item for sublist in summary_tuples for item in sublist]
        print("---generated tuples are---:", summary_tuples)

        print("-----extract tuples from source text-----")
        source_texts: List[str] = tokenize.sent_tokenize(source_text)
        source_tuples = []
        for sentence in source_texts:
            sentence_tuples: List[tuple] = self.extract_tuples(sentence)
            source_tuples.append(sentence_tuples)
        source_tuples = [item for sublist in source_tuples for item in sublist]
        print("---source tuples are---：", source_tuples)

        # # whole source text as input for SRL model
        # print("-----extract tuples from summary text-----")
        # summary_tuples: List[tuple] = self.extract_tuples(generated_summary)
        # print("---generated tuples are---:", summary_tuples)

        # print("-----extract tuples from source text-----")
        # source_tuples: List[tuple] = self.extract_tuples(source_text)
        # print("---source tuples are---：", source_tuples)

        tuple_final_scores = []
        for summary_tup in tqdm(summary_tuples, desc="calculate_summary_score"):
            tuple_final_score = self.compare_tuple_with_relevant_tuples(
                source_tuples, summary_tup
            )
            tuple_final_scores.append(tuple_final_score)
        print("---Score of each tuple in generated text：", tuple_final_scores)

        summary_score = round(mean(tuple_final_scores), 2)
        return summary_score


# {
#     "verbs": [
#         {
#             "verb": "am",
#             "description": "[ARGM-LOC: In the twilight] , [ARG1: I] [V: am] [ARG2: horrified to see a wolf howling at the end of the garden]",
#             "tags": [
#                 "B-ARGM-LOC",
#                 "I-ARGM-LOC",
#                 "I-ARGM-LOC",
#                 "O",
#                 "B-ARG1",
#                 "B-V",
#                 "B-ARG2",
#                 "I-ARG2",
#                 "I-ARG2",
#                 "I-ARG2",
#                 "I-ARG2",
#                 "I-ARG2",
#                 "I-ARG2",
#                 "I-ARG2",
#                 "I-ARG2",
#                 "I-ARG2",
#                 "I-ARG2",
#                 "I-ARG2",
#             ],
#         },
#         {
#             "verb": "see",
#             "description": "In the twilight , [ARG0: I] am horrified to [V: see] [ARG1: a wolf howling] [ARGM-TMP: at the end of the garden]",
#             "tags": [
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "B-ARG0",
#                 "O",
#                 "O",
#                 "O",
#                 "B-V",
#                 "B-ARG1",
#                 "I-ARG1",
#                 "I-ARG1",
#                 "B-ARGM-TMP",
#                 "I-ARGM-TMP",
#                 "I-ARGM-TMP",
#                 "I-ARGM-TMP",
#                 "I-ARGM-TMP",
#                 "I-ARGM-TMP",
#             ],
#         },
#     ],
#     "words": [
#         "In",
#         "the",
#         "twilight",
#         ",",
#         "I",
#         "am",
#         "horrified",
#         "to",
#         "see",
#         "a",
#         "wolf",
#         "howling",
#         "at",
#         "the",
#         "end",
#         "of",
#         "the",
#         "garden",
#     ],
# }


# {
#     "verbs": [
#         {
#             "verb": "ca",
#             "description": "John [V: ca] n't keep up with Mary 's rapid mood swings .",
#             "tags": ["O", "B-V", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"],
#         },
#         {
#             "verb": "keep",
#             "description": "[ARG0: John] [ARGM-MOD: ca] [ARGM-NEG: n't] [V: keep] up [ARG1: with Mary 's rapid mood swings] .",
#             "tags": [
#                 "B-ARG0",
#                 "B-ARGM-MOD",
#                 "B-ARGM-NEG",
#                 "B-V",
#                 "O",
#                 "B-ARG1",
#                 "I-ARG1",
#                 "I-ARG1",
#                 "I-ARG1",
#                 "I-ARG1",
#                 "I-ARG1",
#                 "O",
#             ],
#         },
#     ],
#     "words": [
#         "John",
#         "ca",
#         "n't",
#         "keep",
#         "up",
#         "with",
#         "Mary",
#         "'s",
#         "rapid",
#         "mood",
#         "swings",
#         ".",
#     ],
# }

# {
#     "verbs": [
#         {
#             "verb": "gave",
#             "description": "[ARG0: Peter] [V: gave] [ARG1: his book] [ARG2: to his sister Mary] [ARGM-TMP: yesterday] [ARGM-LOC: in Berlin] . She is a young girl . He wants to make her happy",
#             "tags": [
#                 "B-ARG0",
#                 "B-V",
#                 "B-ARG1",
#                 "I-ARG1",
#                 "B-ARG2",
#                 "I-ARG2",
#                 "I-ARG2",
#                 "I-ARG2",
#                 "B-ARGM-TMP",
#                 "B-ARGM-LOC",
#                 "I-ARGM-LOC",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#             ],
#         },
#         {
#             "verb": "is",
#             "description": "Peter gave his book to his sister Mary yesterday in Berlin . [ARG1: She] [V: is] [ARG2: a young girl] . He wants to make her happy",
#             "tags": [
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "B-ARG1",
#                 "B-V",
#                 "B-ARG2",
#                 "I-ARG2",
#                 "I-ARG2",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#             ],
#         },
#         {
#             "verb": "wants",
#             "description": "Peter gave his book to his sister Mary yesterday in Berlin . She is a young girl . [ARG0: He] [V: wants] [ARG1: to make her happy]",
#             "tags": [
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "B-ARG0",
#                 "B-V",
#                 "B-ARG1",
#                 "I-ARG1",
#                 "I-ARG1",
#                 "I-ARG1",
#             ],
#         },
#         {
#             "verb": "make",
#             "description": "Peter gave his book to his sister Mary yesterday in Berlin . She is a young girl . [ARG0: He] wants to [V: make] [ARG1: her happy]",
#             "tags": [
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "O",
#                 "B-ARG0",
#                 "O",
#                 "O",
#                 "B-V",
#                 "B-ARG1",
#                 "I-ARG1",
#             ],
#         },
#     ],
#     "words": [
#         "Peter",
#         "gave",
#         "his",
#         "book",
#         "to",
#         "his",
#         "sister",
#         "Mary",
#         "yesterday",
#         "in",
#         "Berlin",
#         ".",
#         "She",
#         "is",
#         "a",
#         "young",
#         "girl",
#         ".",
#         "He",
#         "wants",
#         "to",
#         "make",
#         "her",
#         "happy",
#     ],
# }

