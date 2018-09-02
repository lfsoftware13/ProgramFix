import array
import random
import sys
from abc import ABCMeta, abstractmethod
from itertools import zip_longest
from typing import Dict, Iterable, NewType, Optional, Sequence, Sized, cast, Tuple, Type, Any, Iterator, IO, List

Vind = NewType('Vind', int)

def clamp(x: float, lower=0., upper=1.) -> float:
    """
    Clamps a float to within a range (default [0, 1]).
    """
    from math import isnan
    if x <= lower:
        return lower
    elif x >= upper:
        return upper
    elif isnan(x):
        raise FloatingPointError('clamp is undefined for NaN')
    return x

Serialization = Tuple[str, int, Optional[Vind], Optional[Vind]]
PartialSerialization = Tuple[int, Optional[Vind], Optional[Vind]]


class SourceVector(Sequence[Vind]):
    """
    A sequence of vocabulary indices.
    """
    __slots__ = ('tokens',)

    def __init__(self, tokens: Iterable[Vind]) -> None:
        self.tokens = tuple(tokens)

    def __eq__(self, other: Any) -> bool:
        """
        True when both programs are token for token equivalent.

        >>> a = SourceVector([23, 48, 70])
        >>> b = SourceVector([23, 48, 70])
        >>> a == b
        True
        >>> c = SourceVector([23, 48])
        >>> a == c
        False
        """
        if isinstance(other, SourceVector):
            return all(a == b for a, b in zip_longest(self, other))
        else:
            return False

    def __iter__(self) -> Iterator[Vind]:
        return iter(self.tokens)

    # XXX: intentionally leave __getitem__ untyped, because it's annoying.
    def __getitem__(self, index):
        return self.tokens[index]

    def __len__(self) -> int:
        return len(self.tokens)

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return f"{clsname}([{', '.join(str(x) for x in self)}])"

    def print(self, file: IO[str]=sys.stdout) -> None:
        """
        Prints the tokens to a file, using real tokens.
        """
        # from sensibility.language import language
        # for token in self:
        #     print(language.vocabulary.to_text(token), file=file, end=' ')
        # # Print a final newline.
        # print(file=file)
        raise NotImplementedError

    def to_source_code(self) -> bytes:
        """
        Returns the source vector as bytes.
        """
        raise NotImplementedError

    def random_token_index(self) -> int:
        """
        Return the index of a random token in the file.
        """
        return random.randrange(0, len(self))

    def random_insertion_point(self) -> int:
        """
        Return a random insertion point in the program.  That is, an index in
        the program to insert BEFORE. This the imaginary token after the last
        token in the file.
        """
        return random.randint(0, len(self))

    def with_substitution(self, index: int, token: Vind) -> 'SourceVector':
        """
        Return a new program, swapping out the token at index with the given
        token.
        """
        # TODO: O(1) applying edits
        sequence: List[Vind] = []
        sequence.extend(self.tokens[:index])
        sequence.append(token)
        sequence.extend(self.tokens[index + 1:])
        return SourceVector(sequence)

    def with_token_removed(self, index: int) -> 'SourceVector':
        """
        Return a new program with the token at the given index removed.
        """
        assert len(self.tokens) > 0
        # TODO: O(1) applying edits
        assert 0 <= index < len(self)
        sequence: List[Vind] = []
        sequence.extend(self.tokens[:index])
        sequence.extend(self.tokens[index + 1:])
        return SourceVector(sequence)

    def with_token_inserted(self, index: int, token: Vind) -> 'SourceVector':
        """
        Return a new program with the token at the given index removed.
        """
        # TODO: O(1) applying edits
        assert 0 <= index <= len(self)
        sequence: List[Vind] = []
        sequence.extend(self.tokens[:index])
        sequence.append(token)
        sequence.extend(self.tokens[index:])
        return SourceVector(sequence)

    def to_array(self) -> array.array:
        """
        Convert to a dense array.array, suitable for compact serialization.
        """
        return array.array('B', self.tokens)

    def to_bytes(self) -> bytes:
        """
        Convert to bytes, for serialization.
        """
        return self.to_array().tobytes()

    @classmethod
    def from_bytes(self, byte_string: bytes):
        """
        Return an array of vocabulary entries given a byte string produced by
        serialize_token().tobytes()

        >>> SourceVector.from_bytes(b'VZD')
        SourceVector([86, 90, 68])
        """
        as_array = array.array('B', byte_string)
        return SourceVector(tuple(cast(Sequence[Vind], as_array)))

class Edit(metaclass=ABCMeta):
    """
    An abstract base class for edits:

     * Insertion
     * Deletion
     * Substitution

    All edits MUST hold the following property:

        program + edit + (-edit) == program
    """

    code: str
    index: int
    _subclasses: Dict[str, Type['Edit']] = {}

    def __init_subclass__(cls: Type['Edit']) -> None:
        """
        Registers each subclass (Insertion, Deletion, Substitution) with a
        single-letter code. Used for serialization and deserialization.
        """
        assert hasattr(cls, 'code')
        code = cls.code
        assert code not in Edit._subclasses, (
            f"Error creating {cls.__name__}: code {code!r} already exists"
        )
        Edit._subclasses[code] = cls

    @abstractmethod
    def additive_inverse(self) -> 'Edit':
        """
        Return the additive inverse of this edit.

        That is, adding this edit, and then adding the inverse will result in
        the original program:::

            program + edit + (-edit) == program
        """

    @abstractmethod
    def apply(self, program: SourceVector) -> SourceVector:
        """
        Applies the edit to a program.
        """

    @abstractmethod
    def serialize_components(self) -> PartialSerialization:
        """
        Return a tuple of the edit location (token stream index) and any
        relelvant vocabulary index.
        """

    @classmethod
    @abstractmethod
    def create_random_mutation(cls, program: SourceVector) -> 'Edit':
        """
        Creates a random mutation of this kind for the given program.
        """

    # The rest of the functions are given for free.

    @property
    def name(self) -> str:
        """
        Returns the name of this class.
        """
        return type(self).__name__.lower()

    def serialize(self) -> Serialization:
        """
        Return a quadruplet (4-tuple) of
        (code, location, token, original_token),
        useful for serializing and recreating Edit instances.
        """
        return (self.code, *self.serialize_components())

    def __neg__(self) -> 'Edit':
        """
        Return the additive inverse of this edit.
        """
        return self.additive_inverse()

    def __radd__(self, other: SourceVector) -> SourceVector:
        """
        Applies the edit to a program.
        """
        return self.apply(other)

    def __eq__(self, other: Any) -> bool:
        if type(self) == type(other):
            return self.serialize() == other.serialize()
        else:
            return False

    def __hash__(self) -> int:
        return hash(self.serialize())

    @classmethod
    def deserialize(cls, code: str, location: int,
                    token: Optional[Vind],
                    original_token: Optional[Vind]) -> 'Edit':
        """
        Deserializes an edit from tuple notation.
        """
        # subclass = cls._subclasses[code]
        # if subclass is Insertion:
        #     assert original_token is None
        #     return Insertion(location, not_none(token))
        # elif subclass is Deletion:
        #     return Deletion(location, not_none(original_token))
        # else:
        #     assert subclass is Substitution
        #     return Substitution(
        #         location,
        #         replacement=not_none(token),
        #         original_token=not_none(original_token)
        #     )
        raise NotImplementedError


class Insertion(Edit):
    """
    An edit that wedges in a token at a random position in the file, including
    at the very end.

        A token is chosen randomly in the file. A random token from the
        vocabulary is inserted before this token (the end of the file is also
        considered a “token” for the purposes of the insertion operation).

    Index refers to the index in the token stream to insert BEFORE. Hence it
    has a range of [0, len(file)] inclusive, where inserting at index 0 means
    inserting BEFORE the first token, and inserting at index len(file) means
    inserting after the last token in the file (pedantically, it means
    inserting before the imaginary end-of-file token).
    """

    __slots__ = 'index', 'token'

    code = 'i'

    def __init__(self, index: int, token: Vind) -> None:
        self.token = token
        self.index = index

    def __repr__(self) -> str:
        # from sensibility.language import language
        # text_token = language.vocabulary.to_text(self.token)
        return f'Insertion({self.index}, {self.token}'

    def additive_inverse(self) -> Edit:
        return Deletion(self.index, self.token)

    def apply(self, program: SourceVector) -> SourceVector:
        return program.with_token_inserted(self.index, self.token)

    def serialize_components(self) -> PartialSerialization:
        return (self.index, self.token, None)

    @staticmethod
    def create_mutation(program: SourceVector,
                        index: int, token: Vind) -> 'Insertion':
        return Insertion(index, token)

    @classmethod
    def create_random_mutation(cls, program: SourceVector) -> 'Insertion':
        """
        Creates a random insertion for the given program.
        """
        # index = program.random_insertion_point()
        # token = random_vocabulary_entry()
        # return Insertion(index, token)
        raise NotImplementedError


class Deletion(Edit):
    """
    An edit that deletes one token from the program

        A token is chosen randomly in the file. This token is removed from the
        file.
    """

    __slots__ = 'index', 'original_token'

    code = 'x'

    def __init__(self, index: int, original_token: Vind) -> None:
        self.index = index
        self.original_token = original_token

    def __repr__(self) -> str:
        # from sensibility.language import language
        # as_text = language.vocabulary.to_text(self.original_token)
        return f'Deletion({self.index}, {self.original_token}'

    def additive_inverse(self) -> Edit:
        # Insert the deleted token back again
        return Insertion(self.index, self.original_token)

    def apply(self, program: SourceVector) -> SourceVector:
        return program.with_token_removed(self.index)

    def serialize_components(self) -> PartialSerialization:
        return (self.index, None, self.original_token)

    @staticmethod
    def create_mutation(program: SourceVector, index: int) -> 'Deletion':
        return Deletion(index, program[index])

    @classmethod
    def create_random_mutation(cls, program: SourceVector) -> 'Deletion':
        """
        Creates a random deletion for the given program.
        """
        # index = program.random_token_index()
        # return Deletion(index, program[index])
        raise NotImplementedError


class Substitution(Edit):
    """
    An edit that swaps one token for another one.

        A token is chosen randomly in the file. This token is replaced with a
        random token from the vocabulary.
    """

    __slots__ = 'token', 'index', 'original_token'

    code = 's'

    def __init__(self, index: int, *,
                 original_token: Vind,
                 replacement: Vind) -> None:
        self.token = replacement
        self.original_token = original_token
        self.index = index

    def __repr__(self) -> str:
        # from sensibility.language import language
        # new_text = language.vocabulary.to_text(self.token)
        # old_text = language.vocabulary.to_text(self.original_token)
        return (
            f'Substitution({self.index}, '
            f'original_token={self.original_token} , '
            f'new_token={self.token}'
        )

    def additive_inverse(self) -> 'Substitution':
        # Simply swap the tokens again.
        return Substitution(self.index,
                            original_token=self.token,
                            replacement=self.original_token)

    def apply(self, program: SourceVector) -> SourceVector:
        return program.with_substitution(self.index, self.token)

    def serialize_components(self) -> PartialSerialization:
        return (self.index, self.token, self.original_token)

    @staticmethod
    def create_mutation(program: SourceVector,
                        index: int, token: Vind) -> 'Substitution':
        return Substitution(index,
                            original_token=program[index],
                            replacement=token)

    @classmethod
    def create_random_mutation(cls, program: SourceVector) -> 'Substitution':
        """
        Creates a random substitution for the given program.

        Ensures that the new token is NOT the same as the old token!
        """
        # index = program.random_token_index()
        # original_token = program[index]
        #
        # # Generate a token that is NOT the same as the one that is already in
        # # the program!
        # token = original_token
        # while token == program[index]:
        #     token = random_vocabulary_entry()
        #
        # return Substitution(index,
        #                     original_token=original_token,
        #                     replacement=token)
        raise NotImplementedError
