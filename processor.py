"""
Class that provides functionality to merge the predictions of a coref and srl module.
"""

from typing import Dict, Tuple, List
from functools import lru_cache

import spacy
from spacy.tokens import Span
from allennlp_models.coref import CorefPredictor
from allennlp_models.structured_prediction import SemanticRoleLabelerPredictor

from extract_tuples import load_srl_model, load_coref_model, SRLTuple


class EntityToken:
    """
    Class that allows for easy comparison to `str`, but contains additional fields
    """
    text: str
    entity_ref: str

    def __init__(self, text: str, entity_ref: str):
        self.text = text
        self.entity_ref = entity_ref

    def __eq__(self, other):
        if isinstance(other, EntityToken):
            # Could be changed to accommodate only matching on entity_ref, too.
            return self.text == other.text and self.entity_ref == other.entity_ref
        elif isinstance(other, str):
            return self.text == other
        else:
            raise NotImplementedError(f"Comparison between EntityToken and {type(other)} not defined!")

    def __repr__(self):
        return f"EntityToken({self.text}, {self.entity_ref})"


class CustomSpan:
    start: int
    end: int

    def __init__(self, start: int, end: int):
        if end < start:
            raise ValueError(f"Span cannot be initialized for negative range! `end` must be larger or equal to `start`")
        self.start = start
        self.end = end

    def __contains__(self, item):
        if isinstance(item, CustomSpan):
            if item.start >= self.start and item.end <= self.end:
                return True
            else:
                return False
        elif isinstance(item, tuple) or isinstance(item, list):
            if item[0] >= self.start and item[1] <= self.end:
                return True
            else:
                return False
        else:
            raise NotImplementedError(f"Comparison between CustomSpan and {type(item)} not supported!")

    def __len__(self):
        return self.end - self.start

    def __repr__(self):
        return f"({self.start}, {self.end})"

    def __eq__(self, other):
        if isinstance(other, CustomSpan):
            if other.start == self.start and other.end == self.end:
                return True
            else:
                return False
        elif isinstance(other, tuple) or isinstance(other, list):
            if other[0] == self.start and other[1] == self.end:
                return True
            else:
                return False
        else:
            return NotImplementedError(f"Comparison between CustomSpan and {type(other)} not supported!")

    def __hash__(self):
        return hash((self.start, self.end))


@lru_cache(maxsize=1)
def load_spacy_model():
    return spacy.load("en_core_web_sm", disable=("ner",))


class Processor:
    srl_model: SemanticRoleLabelerPredictor
    coref_model: CorefPredictor
    nlp: spacy.language.Language

    ent_dict: Dict
    ent_lookup: Dict
    attribute_map: Dict

    def __init__(self):
        self.coref_model = load_coref_model(cuda=False)
        self.srl_model = load_srl_model(cuda=False)
        self.nlp = load_spacy_model()

        self.attribute_map = {
            "ARG0": "agent",
            "ARGM-NEG": "negation",
            "V": "relation",
            "ARG1": "patient",
            "ARG2": "recipient",
            "ARGM-TMP": "time",
            "ARGM-LOC": "location"
        }

    def process_text(self, text: str):

        # Coref resolution works on longer inputs, as it internally deals with sentence-level processing.
        coref = self.coref_model.predict(text)

        # Initial processing with spacy
        doc = self.nlp(text)

        # SRL extraction works only at sentence-level.
        srl = [self.srl_model.predict(sent.text) for sent in doc.sents]

        self._initialize_entity_lookups(coref, doc)
        self._set_spacy_extension()
        # Re-process once we have the entities
        doc = self.nlp(text)

        return self.extract_tuples(srl, doc), doc

    def _initialize_entity_lookups(self, coref: Dict, doc: spacy.language.Doc) -> None:
        """
        Will create the entity dictionary, mapping from an entity index (integer) to the list of expressions,
        as well as creating the reverse lookup, which allows to get the entity index based on a token span.
        """
        # Create an entity dictionary that synonymous expressions for all spans in a cluster,
        # based on the results from the coreference resolution step
        self.ent_dict = {idx: [doc[begin:end + 1].text for begin, end in clusters]
                         for idx, clusters in enumerate(coref["clusters"])}

        # Also create an inverse lookup to find which index a particular tuple has
        self.ent_lookup = {}
        for idx, clusters in enumerate(coref["clusters"]):
            for begin, end in clusters:
                self.ent_lookup[CustomSpan(start=begin, end=end + 1)] = idx

    def _set_spacy_extension(self) -> None:
        """
        Enables a small spaCy Span annotation that allows entity reverse lookups based on the coref mapping.
        """
        # Define an extension that sets a new spaCy span attribute called "coref_entity".
        # This will return the index of the associated entity if is in the clusters.
        def coref_entity(span: Span) -> int:
            if (span.start, span.end) in self.ent_lookup.keys():
                return self.ent_lookup[(span.start, span.end)]
            else:
                return -1
        # Taken from examples on: https://spacy.io/api/span
        Span.set_extension("coref_entity", getter=coref_entity, force=True)

    def extract_tuples(self, srl, doc: spacy.language.Doc) -> List[SRLTuple]:
        all_tuples = []

        for srl_annotations, sentence in zip(srl, doc.sents):
            for annotation in srl_annotations["verbs"]:
                # We can always set the verb to the relation attribute already
                curr_tuple = SRLTuple()

                spans = self._convert_tags_to_spans(annotation["tags"], offset=sentence.start)

                # If only the relation is known, skip this sentence.
                if len(spans) < 2:
                    continue

                # Assign SRL values based on the extracted spans
                for span, attribute in spans:
                    # TODO: Implement coref resolution on partial matches
                    # self._extract_partial_matches(span, sentence)

                    # Exact matching on doc, since we offset the span indices
                    attribute_value = doc[span.start:span.end].text
                    if span in self.ent_lookup.keys():
                        attribute_value = EntityToken(attribute_value, self.ent_lookup[span])

                    # Complicated way of assigning the correct attribute with the span value
                    if attribute in self.attribute_map.keys():
                        curr_tuple.__setattr__(self.attribute_map[attribute], attribute_value)

                all_tuples.append(curr_tuple)

        return all_tuples

    @staticmethod
    def _convert_tags_to_spans(tags: List[str], offset: int) -> List[Tuple]:
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
                    all_spans.append((CustomSpan(start=curr_span[0]+offset, end=curr_span[1]+offset), curr_label))
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
            curr_span.append(idx+1)
            all_spans.append((CustomSpan(start=curr_span[0]+offset, end=curr_span[1]+offset), curr_label))

        return all_spans

    def _extract_partial_matches(self, span: CustomSpan, sentence: spacy.tokens.Span):
        # Extract the longest possible entity match
        longest_match_length = 0
        longest_match_span = None
        # TODO: Optimize this lookup, as it currently runs in O(N)!
        for ref_span, idx in self.ent_lookup.items():
            if ref_span in span:
                if len(ref_span) > longest_match_length:
                    longest_match_span = ref_span

        attribute_value = sentence[span.start:span.end].text
        # In case we found a match, alter the attribute value to a token instead
        # FIXME: THis currently overwrites the entire sequence, but should only work for parts of the sequence.
        if longest_match_span:
            attribute_value = EntityToken(attribute_value, self.ent_lookup[longest_match_span])

        raise NotImplementedError("This function is incomplete!")


if __name__ == '__main__':
    sample = {'article': "By . Ray Massey, Transport Editor . PUBLISHED: . 19:31 EST, 19 September 2013 . | . UPDATED: . 19:32 EST, 19 September 2013 . The number of parking tickets issued on a Sunday has rocketed after scores of councils introduced seven-day patrols. Figures show motorists are being stung by almost 900,000 parking fines a month at a cost of £30million – a 4 per cent rise on the previous year. And tickets issued on Sundays have increased by 13 per cent – with nearly 300,000 tickets issued on that day of the week in the first five months of 2013. Increase: Motorists are being stung by almost 900,000 parking fines a month at a cost of £30million - a 4 per cent rise on the previous year. While tickets issued on Sundays have increased by 13 per cent . It is believed the rise of Sunday shopping has prompted more town halls to crack down on parking on a day when rules were traditionally relaxed. And the AA says some traffic wardens had even ‘targeted churchgoers and choristers’. The figures were revealed by LV= car insurance in a series of freedom of information requests. The AA says some traffic wardens had even 'targeted churchgoers and choristers' The company said: ‘While there has been\xa0 a general increase across all council areas, there has been\xa0 a significant spike in the number of tickets being issued on Sundays.’ Westminster Council in London has given out the largest number of Sunday parking tickets so far this year at 16,464, followed by the London borough of Lambeth (6,590), Birmingham City Council (3,909), the London borough of Bexley (3,786) and Bristol (1,686). Councils across the UK now hand out an average of 162 parking tickets a day, compared to 154 in 2012, according to the LV= report. But drivers suffer a postcode lottery when it comes to rules on Sunday parking. John O’Roarke, managing\xa0 director of LV=, said: ‘Parking on a Sunday is becoming increasingly difficult and it’s easy to get caught out if you don’t know the local rules.’ AA president Edmund King said it was ‘as if nothing is sacred’, adding: ‘It’s mean-spirited to fine people on a Sunday. ‘The traditional day of rest – when even motorists deserve a bit of relief – is being eroded\xa0 in favour of revenue raising. Money destined for the collection plate is instead flowing into council coffers.’",
              'highlights': "Motorists are being handed nearly 900,000 parking fines a month .\nTickets issued on Sundays have increased by 13 per cent .\nThe AA says some traffic wardens 'target churchgoers and choristers'",
              'id': 'e32de69bba488379354ecb86d67deb46d7b4cc3a'}

    text = "Jeve Jobs acts as Managing Director of Apple. He is also a man."
    proc = Processor()
    # tuples = proc.process_text(sample["article"])
    tuples, doc = proc.process_text(text)
    print(tuples)