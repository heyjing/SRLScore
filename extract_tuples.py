"""
DEPRECATION WARNING: THIS SCRIPT IS OUTDATED, PLEASE USE THE CLASSES IN processor.py


This is a script that extracts facts from texts and represents them using semantic role labels.
For an input text, the output will be a fact database in the form of List[tuple].
The function calculate_summary_score() in the ModelConfigurator class returns a faithfulness score of 
the corresponding summary against its source text.
"""


from typing import List
from functools import lru_cache
import re

from allennlp.predictors.predictor import Predictor
from nltk.stem.wordnet import WordNetLemmatizer
from tqdm import tqdm
import spacy

from custom_datatypes import SRLTuple


@lru_cache(maxsize=2)
def load_spacy_model():
    nlp = spacy.load("en_core_web_lg")
    return nlp


@lru_cache(maxsize=1)
def load_srl_model(cuda: bool = True):
    # TODO: Allow custom CUDA device indices
    if cuda:
        srl_model = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz",
            cuda_device=0,
        )
    else:
        srl_model = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz",
        )
    return srl_model


@lru_cache(maxsize=1)
def load_coref_model(cuda: bool = True):
    # TODO: Allow custom CUDA device indices
    if cuda:
        coref_model = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz",
            cuda_device=1,
        )
    else:
        coref_model = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz",
        )
    return coref_model


def text_preprocessing(text: str, coref_solver: bool) -> str:
    """
    This script processes texts before extracting tuples, like coreference resolution and annotate semantic roles.
    """

    # step 1: AllenNLP coreference resolution
    if coref_solver:
        print("-----text preprocessing: coreference resolution-----")
        coref_model = load_coref_model()
        text = coref_model.coref_resolved(text)

    # step 2: AllenNLP SRL
    print("-----text preprocessing: SRL-----")
    srl_model = load_srl_model()
    srl_annotated_text = srl_model.predict(sentence=text)

    return srl_annotated_text


def tuple_postprocessing(string: str, func_words_remover: bool) -> str:
    """
    Strings in extracted tuples may be long.
    This function removes leading ADP (prepositions like in, to, auf etc.) or
    DET (determiner like this, that, a, an, diese etc.)
    tags for each tuple element of string type, so that the tuples only contain the most important
    semantic information. The SPACY POS Tags List is the same for different languages. 
    """
    if func_words_remover:
        if len(string.split()) == 1:
            return string
        else:
            nlp = load_spacy_model()
            doc = nlp(string)
            for i in range(len(doc)):
                if doc[i].pos_ != "ADP" and doc[i].pos_ != "DET":
                    break
                else:
                    string = string[len(doc[i]) :].lstrip()
            return string
    else:
        return string


def extract_tuples(
    text: str, coref_solver: bool, func_words_remover: bool
) -> List[tuple]:
    """
    This function takes a text as input and returns a list of extracted SRL truth tuples.

    """
    annotated_data: dict = text_preprocessing(text, coref_solver)
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
                extracted_SRLTuple.agent = tuple_postprocessing(
                    latter, func_words_remover
                )
                nr_of_roles += 1
            if former == "ARGM-NEG":
                extracted_SRLTuple.negation = latter
                nr_of_roles += 1
            if former == "V":
                extracted_SRLTuple.relation = WordNetLemmatizer().lemmatize(latter, "v")
                nr_of_roles += 1
            if former == "ARG1":
                extracted_SRLTuple.patient = tuple_postprocessing(
                    latter, func_words_remover
                )
                nr_of_roles += 1
            if former == "ARG2":
                extracted_SRLTuple.recipient = tuple_postprocessing(
                    latter, func_words_remover
                )
                nr_of_roles += 1
            if former == "ARGM-TMP":
                extracted_SRLTuple.time = tuple_postprocessing(
                    latter, func_words_remover
                )
                nr_of_roles += 1
            if former == "ARGM-LOC":
                extracted_SRLTuple.location = tuple_postprocessing(
                    latter, func_words_remover
                )
                nr_of_roles += 1
        # Only tuples with at least two roles will be added to the tuple database of an article
        if nr_of_roles >= 2:
            tuple_database.append(extracted_SRLTuple.format_tuple())
    return tuple_database


if __name__ == "__main__":

    # examples = "Peter gave his book to his sister Mary yesterday in Berlin. She is a young girl. He wants to make her happy"
    # examples = "He walks slowly"
    examples = "In the twilight, I am horrified to see a wolf howling at the end of the garden."
    # examples = "John can't keep up with Mary 's rapid mood swings"
    # examples = "South Korea has opened its market to foreign cigarettes."
    # examples = "I can't do it"
    # examples = "Paul has bought an apple for Anna. She is very happy."

    truth_tuples: List[tuple] = extract_tuples(examples, False, True)
    print(truth_tuples)

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

