"""
This is a script that first uses the ready-to-use AllenNLP coreference resolution and then extracts facts and again represents them 
using semantic role labels. For an input text, the output will be a fact database: List[tuple].
"""


from typing import List
from allennlp.predictors.predictor import Predictor
from nltk.stem.wordnet import WordNetLemmatizer
import re
import spacy


nlp = spacy.load("en_core_web_sm")


# remove prepositions (in, to etc.), determiner (this, that, a, an etc.) and punctuation.
def remove_function_words(string: str) -> str:
    if len(string.split()) == 1:
        return string
    else:
        doc = nlp(string)
        string = ""
        for token in doc:
            if token.pos_ != "ADP" and token.pos_ != "DET" and token.pos_ != "PUNCT":
                if token.text == "'s":
                    string += token.text
                else:
                    string += " " + token.text
        return string.lstrip()


class Tuple:
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
        self.agent = agent if agent else "Ｏ"
        self.negation = negation if negation else "Ｏ"
        self.relation = relation if relation else "Ｏ"
        self.patient = patient if patient else "Ｏ"
        self.recipient = recipient if recipient else "Ｏ"
        self.time = time if time else "Ｏ"
        self.location = location if location else "Ｏ"

    def format_tupel(self) -> tuple:
        lst = list()
        lst.append(self.agent)
        lst.append(self.negation)
        lst.append(self.relation)
        lst.append(self.patient)
        lst.append(self.recipient)
        lst.append(self.time)
        lst.append(self.location)
        return tuple(lst)

    def extract_tuples(self, annotated_data: dict) -> tuple:

        # build an empty tuple database
        tuples = []

        for i in range(len(annotated_data["verbs"])):
            res = re.findall(r"\[.*?\]", annotated_data["verbs"][i]["description"])
            print(res)
            # EXAMPLE res: ['[ARG0: I]', '[V: see]', '[ARG1: a wolf howling]', '[ARGM-TMP: at the end of the garden]']
            nr_of_roles = 0
            for j in res:
                former = j[j.find("[") + 1 : j.find(":")]
                latter = j[j.find(" ") + 1 : j.find("]")]
                if former == "ARG0":
                    self.agent = remove_function_words(latter)
                    nr_of_roles += 1
                if former == "ARGM-NEG":
                    self.negation = latter
                    nr_of_roles += 1
                if former == "V":
                    self.relation = WordNetLemmatizer().lemmatize(latter, "v")
                    nr_of_roles += 1
                if former == "ARG1":
                    self.patient = remove_function_words(latter)
                    nr_of_roles += 1
                if former == "ARG2":
                    self.recipient = remove_function_words(latter)
                    nr_of_roles += 1
                if former == "ARGM-TMP":
                    self.time = latter
                    nr_of_roles += 1
                if former == "ARGM-LOC":
                    self.location = latter
                    nr_of_roles += 1
            # Only tuples with at least two roles will be added to the tuple database of an article
            if nr_of_roles >= 2:
                tuples.append(self.format_tupel())
            self.__init__()
        return tuples


if __name__ == "__main__":

    # step 1: AllenNLP coreference resolution
    coref_resolved_text = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
    ).coref_resolved(
        "Peter gave his book to his sister Mary yesterday in Berlin. She is a young girl. He wants to make her happy"
        # "He walks slowly"
        # "In the twilight, I am horrified to see a wolf howling at the end of the garden"
        # "John can't keep up with Mary 's rapid mood swings"
        # "South Korea has opened its market to foreign cigarettes."
        # "I can't do it"
        # "Paul has bought an apple for Anna. She is very happy."
    )

    # step 2: AllenNLP SRL
    srl_annotated_text = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
    ).predict(sentence=coref_resolved_text)

    tup = Tuple()
    tuple_database: List[tuple] = tup.extract_tuples(srl_annotated_text)
    print(tuple_database)

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

