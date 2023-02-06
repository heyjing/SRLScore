"""
Class that provides functionality to merge the predictions of a coref and srl module.
"""

from typing import Dict, Tuple, List, Union
from functools import lru_cache
import warnings

import spacy
from spacy.language import Doc
from allennlp_models.coref import CorefPredictor
from allennlp_models.structured_prediction import SemanticRoleLabelerPredictor
from nltk.stem.wordnet import WordNetLemmatizer

from extract_tuples import load_srl_model, load_coref_model
from custom_datatypes import SRLTuple, EntityToken, CustomSpan


@lru_cache(maxsize=1)
def load_spacy_model():
    return spacy.load("en_core_web_sm", disable=("ner",))


class Processor:
    """
    Alternative processing class, unifying annotations from the coreference and SRL modules.
    """

    srl_model: SemanticRoleLabelerPredictor
    coref_model: CorefPredictor
    nlp: spacy.language.Language

    ent_dict: Dict
    ent_lookup: Dict
    attribute_map: Dict

    do_coref: bool

    def __init__(self, do_coref: bool, use_cuda: bool = False):
        self.coref_model = load_coref_model(cuda=use_cuda)
        self.srl_model = load_srl_model(cuda=use_cuda)
        self.nlp = load_spacy_model()

        self.attribute_map = {
            "ARG0": "agent",
            "ARGM-NEG": "negation",
            "V": "relation",
            "ARG1": "patient",
            "ARG2": "recipient",
            "ARGM-TMP": "time",
            "ARGM-LOC": "location",
        }

        self.do_coref = do_coref

    def process_text(self, text: str) -> List[List[Tuple]]:
        """
        Function that extracts tuples from a text.
        """
        # Initial processing with spacy
        doc = self.nlp(text)

        # SRL extraction works only at sentence-level.
        srl = []
        # Also specifically catch invalid sentences, usually due to tables, which will be ignored.
        for sent in doc.sents:
            # Catching errors likely due to incorrect token normalization
            try:
                result = self.srl_model.predict(sent.text)
            # FIXME: This may cause problem in later iterations, so make sure that empty lists are caught
            #  elsewhere, too!
            except RuntimeError as e:
                result = []
                warnings.warn(
                    f"Processing sentence caused an error in the SRL model! You might want to investigate!\n"
                    f"Error message: '{e}'\n"
                    f"Responsible sentence '{sent.text}'"
                )
            srl.append(result)

        # Coref resolution works on longer inputs, as it internally deals with sentence-level processing.
        if self.do_coref:
            coref = self.coref_model.predict(text)
            self._initialize_entity_lookups(coref, doc)

        return self.extract_tuples(srl, doc)

    def _initialize_entity_lookups(self, coref: Dict, doc: Doc) -> None:
        """
        Will create the entity dictionary, mapping from an entity index (integer) to the list of expressions,
        as well as creating the reverse lookup, which allows to get the entity index based on a token span.
        """
        # Create an entity dictionary that synonymous expressions for all spans in a cluster,
        # based on the results from the coreference resolution step
        self.ent_dict = {
            idx: [doc[begin : end + 1].text.casefold() for begin, end in clusters]
            for idx, clusters in enumerate(coref["clusters"])
        }

        # Also create an inverse lookup to find which index a particular tuple has
        self.ent_lookup = {}
        for idx, clusters in enumerate(coref["clusters"]):
            for begin, end in clusters:
                self.ent_lookup[CustomSpan(start=begin, end=end + 1)] = idx

    def extract_tuples(self, srl, doc: Doc) -> List[List[Tuple]]:
        all_tuples = []

        for srl_annotations, sentence in zip(srl, doc.sents):
            # Catch cases where annotations were running into problems
            if srl_annotations == []:
                continue

            for annotation in srl_annotations["verbs"]:

                # We can always set the verb to the relation attribute already
                curr_tuple = SRLTuple()

                spans = self._convert_tags_to_spans(
                    annotation["tags"], offset=sentence.start
                )

                # If only the relational verb is known, skip this sentence.
                if len(spans) < 2:
                    continue

                # Assign SRL values based on the extracted spans
                for span, attribute in spans:
                    attribute_tuple, attribute_value = self._generate_attribute_tuple(
                        span, doc
                    )

                    # Complicated way of assigning the correct attribute with the span value
                    if attribute in self.attribute_map.keys() and (
                        attribute_value != "" or attribute_tuple
                    ):
                        if attribute_tuple:
                            curr_tuple.__setattr__(
                                self.attribute_map[attribute], attribute_tuple
                            )
                        else:  # implies attribute_value != ""
                            curr_tuple.__setattr__(
                                self.attribute_map[attribute], attribute_value
                            )

                # Need at least two "relevant" arguments in the relation
                if self._count_non_zero_entries(curr_tuple) >= 2:
                    if self.do_coref == True:
                        all_tuples.append(curr_tuple.explode_tuple(self.ent_dict))
                    else:  # implies we do not need to explode tuples
                        all_tuples.append([curr_tuple.format_tuple()])

        return all_tuples

    @staticmethod
    def _convert_tags_to_spans(
        tags: List[str], offset: int
    ) -> List[Tuple[CustomSpan, str]]:
        """
        Method that converts a BIO tagging sequence (e.g., ["O", "B-ARG1", "B-ARG2", "B-V", "O", "O"])
        into a sequence of spans with associated labels (e.g., "[([1, 2], "ARG1"), ([2, 3], "ARG2"), ([3, 4], "V")].
        This can be offset by an integer amount, which is required for multi-sentence matching to work.
        """
        all_spans = []
        curr_span = []
        curr_label = ""
        for idx, tag in enumerate(tags):
            if tag == "O" or tag.startswith("B-"):
                # We have some previous span. Finish it off with the index and then return
                if curr_span != []:
                    curr_span.append(idx)
                    all_spans.append(
                        (
                            CustomSpan(
                                start=curr_span[0] + offset, end=curr_span[1] + offset
                            ),
                            curr_label,
                        )
                    )
                    if tag == "O":
                        curr_span = []
                        curr_label = ""
                    else:
                        curr_span = [idx]
                        curr_label = tag[2:]
                # No entry in the span, define a starting position
                else:
                    if tag.startswith("B-"):
                        curr_label = tag[2:]
                        curr_span.append(idx)

            # For intermediate tags, simply continue, since we're only interested in boundaries
            if tag.startswith("I-"):
                continue

        # Finish any last elements
        if curr_span != []:
            # idx is guaranteed to exist, since otherwise curr_span would be empty
            curr_span.append(idx + 1)
            all_spans.append(
                (
                    CustomSpan(start=curr_span[0] + offset, end=curr_span[1] + offset),
                    curr_label,
                )
            )

        return all_spans

    def _generate_attribute_tuple(
        self, span: CustomSpan, doc: Doc
    ) -> Tuple[Union[Tuple, None], Union[int, None]]:
        # Attempt to find entity matches in the current span
        if self.do_coref:
            attribute_tuple, entity_span_start = self._extract_partial_matches(
                span, doc
            )
            if attribute_tuple is not None:
                attribute_value = attribute_tuple[0]
            else:
                attribute_value = None
        else:
            attribute_tuple = None
            attribute_value = None
            entity_span_start = None

        # If none are found, default back to extracting the full string
        if attribute_value is None:
            # Exact matching on doc, since we offset the span indices
            attribute_value = doc[span.start : span.end].text.casefold()
        # Also adjust the end position in case there are no entities found
        if entity_span_start is None:
            entity_span_start = span.end

        # Converting a verb to its base form, removes leading ADP(prepositions like in, to, auf etc.) and
        # DET(determiner like this, that, a, an, diese etc.)
        # For partial entity matches, use only the pre-entity text as a filter.
        for token in doc[span.start : entity_span_start]:
            if token.pos_ == "VERB":
                attribute_value = WordNetLemmatizer().lemmatize(attribute_value, "v")
                break
            elif token.pos_ != "ADP" and token.pos_ != "DET":
                break
            else:
                attribute_value = attribute_value[len(token) :].lstrip()

        # Re-assign the cleaned pre-entity text for the updated attribute tuple
        if attribute_tuple:
            attribute_tuple = (attribute_value, attribute_tuple[1], attribute_tuple[2])

        return attribute_tuple, attribute_value

    def _extract_partial_matches(
        self, span: CustomSpan, doc: spacy.language.Doc
    ) -> Tuple[Union[Tuple, None], Union[int, None]]:
        # Extract the longest possible entity match
        longest_match_span = None
        # TODO: Optimize this lookup, as it currently runs in O(N)!
        if len(span) == 1:
            if self.ent_lookup.get(span) is not None:
                longest_match_span = span
        if len(span) > 1:
            span_indexes = [
                index
                for index in self.ent_lookup.keys()
                if index.start >= span.start and index.end <= span.end
            ]

            if len(span_indexes) >= 2:
                longest_match_span = max(span_indexes, key=len)
            if len(span_indexes) == 1:
                longest_match_span = span_indexes[0]

        # In case we found a match, alter the attribute value to a list of string/EntityToken entries
        if longest_match_span:
            # Generates a list of string spans and entity token spans;
            # For example, "his sister Mary Jane Austin", coupled with the reference "sister Mary", would return
            # ["his", EntityToken("sister Mary"), "Jane Austin"]
            pre_entity_text = doc[span.start : longest_match_span.start].text.casefold()
            post_entity_text = doc[longest_match_span.end : span.end].text.casefold()

            entity_token = EntityToken(
                doc[longest_match_span.start : longest_match_span.end].text.casefold(),
                self.ent_lookup[longest_match_span],
            )

            final_attribute = (pre_entity_text, entity_token, post_entity_text)
            return final_attribute, longest_match_span.start

        # No suitable partial/full match found
        else:
            return None, None

    @staticmethod
    def _count_non_zero_entries(tup: SRLTuple) -> int:
        return sum(x is not None for x in tup.format_tuple())


if __name__ == "__main__":
    sample = {
        "article": "By . Ray Massey, Transport Editor . PUBLISHED: . 19:31 EST, 19 September 2013 . | . UPDATED: . 19:32 EST, 19 September 2013 . The number of parking tickets issued on a Sunday has rocketed after scores of councils introduced seven-day patrols. Figures show motorists are being stung by almost 900,000 parking fines a month at a cost of £30million – a 4 per cent rise on the previous year. And tickets issued on Sundays have increased by 13 per cent – with nearly 300,000 tickets issued on that day of the week in the first five months of 2013. Increase: Motorists are being stung by almost 900,000 parking fines a month at a cost of £30million - a 4 per cent rise on the previous year. While tickets issued on Sundays have increased by 13 per cent . It is believed the rise of Sunday shopping has prompted more town halls to crack down on parking on a day when rules were traditionally relaxed. And the AA says some traffic wardens had even ‘targeted churchgoers and choristers’. The figures were revealed by LV= car insurance in a series of freedom of information requests. The AA says some traffic wardens had even 'targeted churchgoers and choristers' The company said: ‘While there has been\xa0 a general increase across all council areas, there has been\xa0 a significant spike in the number of tickets being issued on Sundays.’ Westminster Council in London has given out the largest number of Sunday parking tickets so far this year at 16,464, followed by the London borough of Lambeth (6,590), Birmingham City Council (3,909), the London borough of Bexley (3,786) and Bristol (1,686). Councils across the UK now hand out an average of 162 parking tickets a day, compared to 154 in 2012, according to the LV= report. But drivers suffer a postcode lottery when it comes to rules on Sunday parking. John O’Roarke, managing\xa0 director of LV=, said: ‘Parking on a Sunday is becoming increasingly difficult and it’s easy to get caught out if you don’t know the local rules.’ AA president Edmund King said it was ‘as if nothing is sacred’, adding: ‘It’s mean-spirited to fine people on a Sunday. ‘The traditional day of rest – when even motorists deserve a bit of relief – is being eroded\xa0 in favour of revenue raising. Money destined for the collection plate is instead flowing into council coffers.’",
        "highlights": "Motorists are being handed nearly 900,000 parking fines a month .\nTickets issued on Sundays have increased by 13 per cent .\nThe AA says some traffic wardens 'target churchgoers and choristers'",
        "id": "e32de69bba488379354ecb86d67deb46d7b4cc3a",
    }

    # text = "Jeve Jobs acts as Managing Director of Apple. He is also a man."
    # text = "Peter gave his book to his sister Mary yesterday in Berlin. She is a young girl. He wants to make her happy"
    text = "In the twilight, I am horrified to see a wolf howling at the end of the garden. The wolf is very big."
    proc = Processor(do_coref=True)
    # tuples = proc.process_text(sample["article"])
    tuples = proc.process_text(text)
    print(tuples, len(tuples), type(tuples), type(tuples[0]))
