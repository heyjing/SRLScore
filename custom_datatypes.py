"""
A collection of custom data types for interfacing SRL and coreference information.
"""
import itertools


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
        """
        Represent the SRLTuple information as a static tuple for serialization purposes.
        """
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

    def explode_tuple(self, ent_dict: dict) -> list[tuple]:
        """
        Generates a list of
        If the SRLTuple contains EntityTokens instead of raw strings, it will generate multiple string tuples
        with the combinations of all possible combinations of expressions.

        Example:
            entity_dict = {0: {"Peter", "his"}}
            tup = SRLTuple(EntityToken("Peter", 0))
            res = tup.format_tuple()
            print(res)
            # Two samples generated, namely:
            # [(Peter, None, ...),
            #  (his, None, ...)]
        """
        all_combinations = []
        # FIXME: This seems awfully error-prone for changed attribute names?
        attributes = ["agent", "negation", "relation", "patient", "recipient", "time", "location"]
        for attribute in attributes:
            all_combinations.append(self._get_attribute_values(attribute, ent_dict))

        return list(itertools.product(*all_combinations))

    def _get_attribute_values(self, attr_name: str, ent_dict: dict) -> list:
        """
        Will generate all possible variations a particular attribute value could hold.
        This is important in case we store a entity-like string, which may cause our method to introduce alternatives.
        """
        curr_attribute_value = self.__getattribute__(attr_name)

        if curr_attribute_value is None or isinstance(curr_attribute_value, str):
            return [curr_attribute_value]
        elif isinstance(curr_attribute_value, tuple):
            pre_string, entity, post_string = curr_attribute_value
            entity_alternatives = set(ent_dict[entity.entity_ref])
            # Just in case, add the current entity text to it, too.
            entity_alternatives.add(entity.text.casefold())

            all_alternatives = []
            for alternative in entity_alternatives:
                # Stripping is necessary in case pre/post strings are empty
                all_alternatives.append(f"{pre_string} {alternative} {post_string}".strip())
            return all_alternatives
        else:
            raise ValueError(f"Invalid attribute type encountered: attribute '{attr_name}' "
                             f"holds a value of type '{type(curr_attribute_value)}'")

    def __repr__(self):
        """
        Represent a SRLTuple by printing its attributes in sequence.
        """
        return f"SRLTuple(agent: {self.agent}, " \
               f"negation: {self.negation}, " \
               f"relation: {self.relation}, " \
               f"patient: {self.patient}, " \
               f"recipient: {self.recipient}, " \
               f"time: {self.time}, " \
               f"location: {self.location})"


class EntityToken:
    """
    Class that allows for easy comparison to `str`, but contains additional fields
    """

    text: str
    entity_ref: int

    def __init__(self, text: str, entity_ref: int):
        self.text = text
        self.entity_ref = entity_ref

    def __eq__(self, other):
        """
        Overload equality checks, which allows us to do stuff like
            str("Test") == EntityToken("Test", 0)
        etc.
        """
        if isinstance(other, EntityToken):
            # Could be changed to accommodate only matching on entity_ref, too.
            return self.text == other.text and self.entity_ref == other.entity_ref
        elif isinstance(other, str):
            return self.text == other
        else:
            raise NotImplementedError(
                f"Comparison between EntityToken and {type(other)} not defined!"
            )

    def __repr__(self):
        """
        Defines a "surface form representation" for the class. Among other things, will print nicer.
        """
        return f"EntityToken({self.text}, {self.entity_ref})"


class CustomSpan:
    """
    Custom Span class, which allows for easier equality/range checks.
    """

    start: int
    end: int

    def __init__(self, start: int, end: int):
        if end < start:
            raise ValueError(
                f"Span cannot be initialized for negative range! `end` must be larger or equal to `start`"
            )
        self.start = start
        self.end = end

    def __contains__(self, item):
        """
        Overload behavior for checks like
            CustomSpan(0, 2) in CustomSpan(-1, 3)
        """
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
            raise NotImplementedError(
                f"Comparison between CustomSpan and {type(item)} not supported!"
            )

    def __len__(self):
        """
        Define the "length" of a span.
        """
        return self.end - self.start

    def __repr__(self):
        """
        Surface form representation.
        """
        return f"({self.start}, {self.end})"

    def __eq__(self, other):
        """
        Overload comparison functionality, allowing for checks with other CustomSpans and tuples/lists
        """
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
            return NotImplementedError(
                f"Comparison between CustomSpan and {type(other)} not supported!"
            )

    def __hash__(self):
        """
        Once __eq__ is defined, __hash__ also needs to be re-defined to avoid `Unhashable` errors.
        """
        return hash((self.start, self.end))

