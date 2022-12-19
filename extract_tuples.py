"""
This is a script that first uses the ready-to-use AllenNLP coreference resolution and then extracts facts and again represents them 
using semantic role labels. For an input text, the output will be a fact database: List[tuple].
"""


from typing import List
from functools import lru_cache
import re
from allennlp.predictors.predictor import Predictor
from nltk.stem.wordnet import WordNetLemmatizer
from tqdm import tqdm
import spacy


@lru_cache(maxsize=1)
def load_language_model():
    nlp = spacy.load("en_core_web_lg")
    return nlp


def remove_function_words(string: str) -> str:
    """
    This function removes ADP(prepositions like in, to, auf etc.), DET(determiner like this, that, a, an, diese etc.) 
    and PUNCT(punctuation) tags from a string. The SPACY POS Tags List is the same for different languages. 
    """
    # if len(string.split()) == 1:
    #     return string
    # else:
    #     nlp = load_language_model()
    #     doc = nlp(string)
    #     string = ""
    #     for token in doc:
    #         if token.pos_ != "ADP" and token.pos_ != "DET" and token.pos_ != "PUNCT":
    #             if token.text == "'s":
    #                 string += token.text
    #             else:
    #                 string += " " + token.text
    #     return string.lstrip()
    return string


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


@lru_cache(maxsize=2)
def load_srl_models():
    # coref_model = Predictor.from_path(
    #     "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz",
    #     cuda_device=1,
    # )
    srl_model = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz",
        cuda_device=0,
    )

    # return coref_model, srl_model
    return srl_model


def text_preprocessing(text: str) -> str:
    # # step 1: AllenNLP coreference resolution
    # print("-----text preprocessing: coreference resolution-----")
    # coref_model, srl_model = load_srl_models()
    # coref_resolved_text = coref_model.coref_resolved(text)
    # step 2: AllenNLP SRL
    print("-----text preprocessing: SRL-----")
    srl_model = load_srl_models()
    srl_annotated_text = srl_model.predict(sentence=text)

    return srl_annotated_text


def extract_tuples(text: str) -> List[tuple]:
    """
    This function takes an text as input and returns a list of extracted SRL truth tuples.

    """
    annotated_data: dict = text_preprocessing(text)
    # build an empty tuple database
    tuple_database = []

    for i in tqdm(range(len(annotated_data["verbs"])), desc="Extracting tuples"):
        res = re.findall(r"\[.*?\]", annotated_data["verbs"][i]["description"])
        # EXAMPLE res: ['[ARG0: I]', '[V: see]', '[ARG1: a wolf howling]', '[ARGM-TMP: at the end of the garden]']
        nr_of_roles = 0
        extracted_SRLTuple = SRLTuple()
        for j in res:
            former = j[j.find("[") + 1 : j.find(":")]
            latter = j[j.find(" ") + 1 : j.find("]")]
            if former == "ARG0":
                extracted_SRLTuple.agent = remove_function_words(latter)
                nr_of_roles += 1
            if former == "ARGM-NEG":
                extracted_SRLTuple.negation = latter
                nr_of_roles += 1
            if former == "V":
                extracted_SRLTuple.relation = WordNetLemmatizer().lemmatize(latter, "v")
                nr_of_roles += 1
            if former == "ARG1":
                extracted_SRLTuple.patient = remove_function_words(latter)
                nr_of_roles += 1
            if former == "ARG2":
                extracted_SRLTuple.recipient = remove_function_words(latter)
                nr_of_roles += 1
            if former == "ARGM-TMP":
                extracted_SRLTuple.time = latter
                nr_of_roles += 1
            if former == "ARGM-LOC":
                extracted_SRLTuple.location = latter
                nr_of_roles += 1
        # Only tuples with at least two roles will be added to the tuple database of an article
        if nr_of_roles >= 2:
            tuple_database.append(extracted_SRLTuple.format_tuple())
    return tuple_database


if __name__ == "__main__":

    examples = "Peter gave his book to his sister Mary yesterday in Berlin. She is a young girl. He wants to make her happy"
    # examples = "He walks slowly"
    # examples = "In the twilight, I am horrified to see a wolf howling at the end of the garden"
    # examples = "John can't keep up with Mary 's rapid mood swings"
    # examples = "South Korea has opened its market to foreign cigarettes."
    # examples = "I can't do it"
    # examples = "Paul has bought an apple for Anna. She is very happy."

    truth_tuples: List[tuple] = extract_tuples(examples)
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

