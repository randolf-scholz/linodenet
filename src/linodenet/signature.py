r"""Module for parsing and representing function signatures between vector spaces."""

__all__ = [
    # Types
    "SignatureType",
    "ShapeType",
    "ArgType",
    "ArgList",
    # Classes
    "Identifier",
    "GenericType",
    "Parser",
    "Token",
    "TokenKind",
    "DimKind",
    # dimension types
    "Dim",
    "DimType",
    "ConstantDim",
    "StaticDim",
    "DynamicDim",
    "UnknownDim",
    # Functions
    "parse_arglist",
    "parse_signature",
    "tokenize",
]

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from types import EllipsisType
from typing import Final, Literal, TypeIs, overload

type ShapeType = tuple[DimType, ...] | tuple[EllipsisType, *tuple[DimType, ...]]
type DimType = ConstantDim | StaticDim | DynamicDim | UnknownDim
type ArgType = ShapeType | Identifier | GenericType
type ArgList = list[ArgType]
type QMARK = Literal["?"]


def _shape_to_str(shape: ShapeType) -> str:
    if len(shape) > 0 and shape[0] is Ellipsis:
        return f"(..., {', '.join(str(dim) for dim in shape[1:])})"
    return f"({', '.join(str(dim) for dim in shape)})"


def _arg_to_str(arg: ArgType) -> str:
    if isinstance(arg, tuple):
        return _shape_to_str(arg)
    return str(arg)


class Identifier(str):
    r"""Class for representing identifier types."""

    __slots__ = ()

    def __init__(self, name: str) -> None:
        if not is_identifier(name):
            raise ValueError(f"Invalid identifier: {name}")


def is_identifier(s: str, /) -> TypeIs[Identifier]:
    r"""Check if the string is a valid identifier."""
    return s.isidentifier() and not s.startswith("_")


@dataclass(frozen=True, slots=True)
class GenericType:
    r"""Class for representing generic types with type arguments."""

    id: Identifier
    arglist: ArgList

    def __str__(self) -> str:
        return f"{self.id!s}[{','.join(map(_arg_to_str, self.arglist))}]"


class DimKind(StrEnum):
    r"""Enumeration of dimension kinds."""

    CONSTANT = "constant"  # '1', '2', '3', ... fixed-size dimension
    STATIC = "static"  # 'n' variable (fixed at initialization time)
    DYNAMIC = "dynamic"  # '*n' variable size dimension
    UNKNOWN = "unknown"  # for future use


@dataclass(frozen=True, slots=True)
class Dim:
    r"""Class for representing a single dimension."""

    value: Identifier | int | QMARK
    kind: DimKind

    def __str__(self) -> str:
        return f"{self.value!s}"


@dataclass(frozen=True, slots=True)
class ConstantDim(Dim):
    r"""Class for representing constant dimensions."""

    value: int
    kind: Final[DimKind] = DimKind.CONSTANT


@dataclass(frozen=True, slots=True)
class StaticDim(Dim):
    r"""Class for representing static dimensions."""

    value: Identifier
    kind: Final[DimKind] = DimKind.STATIC


@dataclass(frozen=True, slots=True)
class DynamicDim(Dim):
    r"""Class for representing dynamic dimensions."""

    value: Identifier
    kind: Final[DimKind] = DimKind.DYNAMIC

    def __repr__(self) -> str:
        return f"*{self.value!s}"


@dataclass(frozen=True, slots=True)
class UnknownDim(Dim):
    r"""Class for representing unknown dimensions."""

    value: Final[QMARK] = "?"
    kind: Final[DimKind] = DimKind.UNKNOWN


@dataclass(frozen=True, slots=True)
class SignatureType:
    r"""Class for representing signatures of functions between vector spaces.

    A signature is of the form:

    - ``arg1 -> ret1`` for single input single output (SISO)
      - ``[arg1] -> ret1`` is also allowed
    - ``[arg1, arg2, ...] -> ret1`` for multiple input single output (MISO)
    - ``arg1 -> [ret1, ret2, ...]`` for single input multiple output (SIMO)
    - ``[arg1, arg2, ...] -> [ret1, ret2, ...]`` for multiple input multiple output (MIMO)

    Arguments can be one of:

    - tensor shapes represented as literal tuples of dimensions
    - tensor types represented as generic types
    - python types like `int`, `float`, `str`, etc.

    Note: A single tensor is represented by a tuple with elements
        - integers for fixed-size dimensions
        - strings (`"name"` ) for a single axis of fixed size
        - strings (`"*name"` ) for a single axis of variable size
        - strings (`"**xs"`) for a variable number of axes of variable size
        - Ellipsis (`...`) for axes that are vectorized over
        - at maximum one Ellipsis is allowed per tensor.
    """

    argument_types: ArgList
    return_types: ArgList

    def __str__(self) -> str:
        r"""Convert the signature to a string."""
        args = self.argument_types
        rets = self.return_types

        arg_strs: list[str] = []
        for arg in args:
            if isinstance(arg, tuple):
                arg_strs.append(_shape_to_str(arg))
            else:
                arg_strs.append(str(arg))

        ret_strs: list[str] = []
        for ret in rets:
            if isinstance(ret, tuple):
                ret_strs.append(_shape_to_str(ret))
            else:
                ret_strs.append(str(ret))

        arg_part = arg_strs[0] if len(args) == 1 else f"[{', '.join(arg_strs)}]"
        ret_part = ret_strs[0] if len(rets) == 1 else f"[{', '.join(ret_strs)}]"
        return f"{arg_part} -> {ret_part}"


class TokenKind(StrEnum):
    r"""Enumeration of token kinds for the ArgList / Signature parser."""

    IDENT = r"[A-Za-z]\w*]"
    INT = r"\d+"
    ELLIPSIS = "..."
    QMARK = "?"
    STAR = "*"
    ARROW = "->"
    LBRACKET = "["
    RBRACKET = "]"
    LPAREN = "("
    RPAREN = ")"
    COMMA = ","
    EOF = "EOF"


@dataclass(frozen=True, slots=True)
class Token:
    r"""Class representing a token in the ArgList / Signature parser."""

    pos: int  # character index in original string
    value: str
    kind: TokenKind  # "IDENT", "INT", "ELLIPSIS", "STAR", "DSTAR", symbols like "[", "]", "(", ")", ",", "EOF"

    @overload
    def __init__(
        self, pos: int, kind: Literal[TokenKind.IDENT, TokenKind.INT], value: str
    ) -> None: ...
    @overload
    def __init__(self, pos: int, kind: TokenKind) -> None: ...
    def __init__(self, pos: int, kind: TokenKind, value: str | None = None) -> None:
        if kind in (TokenKind.IDENT, TokenKind.INT) and value is None:
            raise AssertionError("IDENT and INT tokens require a value")
        if kind not in (TokenKind.IDENT, TokenKind.INT) and value is not None:
            raise AssertionError("Only IDENT and INT tokens may carry custom values")
        if kind not in (TokenKind.IDENT, TokenKind.INT):
            assert value is None
            value = kind.value

        object.__setattr__(self, "pos", int(pos))
        object.__setattr__(self, "value", str(value))
        object.__setattr__(self, "kind", TokenKind(kind))

    def __repr__(self) -> str:
        return f"Token({self.kind}, {self.value!r}, pos={self.pos})"


def tokenize(source: str, /) -> Iterator[Token]:
    i = 0
    n = len(source)

    while i < n:
        c = source[i]

        # Skip whitespace
        if c.isspace():
            i += 1
            continue

        # Multi-character punctuation: ellipsis, double star, arrow
        if source.startswith("...", i):
            yield Token(i, TokenKind.ELLIPSIS)
            i += 3
            continue

        if source.startswith("->", i):
            yield Token(i, TokenKind.ARROW)
            i += 2
            continue

        # Single-character punctuation / operators
        if c in "[](),*?":
            yield Token(i, TokenKind(c))  # maps "[" -> LBRACKET, etc.
            i += 1
            continue

        # Identifier: [A-Za-z]\w*
        if c.isalpha():
            start = i
            i += 1
            while i < n and (source[i].isalnum() or source[i] == "_"):
                i += 1
            value = source[start:i]
            yield Token(start, TokenKind.IDENT, value)
            continue

        # Integer literal: \d+
        if c.isdigit():
            start = i
            i += 1
            while i < n and source[i].isdigit():
                i += 1
            value = source[start:i]
            yield Token(start, TokenKind.INT, value)
            continue

        # If we get here, it's an invalid character
        raise SyntaxError(f"Unexpected character {c!r} at position {i}")

    yield Token(n, TokenKind.EOF)


class Parser:
    r"""Recursive-descent parser consuming an Iterator[Token]."""

    def __init__(self, tokens: Iterator[Token]) -> None:
        self._tokens = iter(tokens)
        # prime the first token
        try:
            self._current: Token = next(self._tokens)
        except StopIteration:
            # empty stream -> synthetic EOF at pos 0
            self._current = Token(0, TokenKind.EOF)

    @property
    def current(self) -> Token:
        return self._current

    def _advance(self) -> None:
        try:
            self._current = next(self._tokens)
        except StopIteration:
            # Once exhausted, stay on EOF
            if self._current.kind is not TokenKind.EOF:
                self._current = Token(self._current.pos, TokenKind.EOF)

    # Utility methods

    def consume(self, kind: TokenKind) -> Token:
        tok = self.current
        if tok.kind is not kind:
            raise SyntaxError(
                f"Expected {kind.name} at position {tok.pos}, "
                f"got {tok.kind.name} ({tok.value!r})"
            )
        self._advance()
        return tok

    def parse_signature(self) -> SignatureType:
        r"""Parse the grammar starting from SignatureType and ensure full consumption."""
        result = self._parse_signature()
        if self.current.kind is not TokenKind.EOF:
            tok = self.current
            raise SyntaxError(
                f"Unexpected token {tok.kind.name} {tok.value!r} at position {tok.pos}, "
                "expected end of input"
            )
        return result

    def parse_arglist(self) -> ArgList:
        r"""Parse the grammar starting from ArgList and ensure full consumption."""
        result = self._parse_arglist()
        if self.current.kind is not TokenKind.EOF:
            tok = self.current
            raise SyntaxError(
                f"Unexpected token {tok.kind.name} {tok.value!r} at position {tok.pos}, "
                "expected end of input"
            )
        return result

    # SignatureType ::= (ArgList | ArgType) "->" (ArgList | ArgType)

    def _parse_signature(self) -> SignatureType:
        """Parse the grammar starting from SignatureType and ensure full consumption.

        SignatureType ::= (ArgList | ArgType) "->" (ArgList | ArgType)
        """
        lhs = (
            self._parse_arglist()
            if self.current.kind is TokenKind.LBRACKET
            else [self._parse_argtype()]
        )

        self.consume(TokenKind.ARROW)

        rhs = (
            self._parse_arglist()
            if self.current.kind is TokenKind.LBRACKET
            else [self._parse_argtype()]
        )

        return SignatureType(lhs, rhs)

    # ArgList ::= "[" (ArgType ("," ArgType)*)? "]"

    def _parse_arglist(self) -> ArgList:
        self.consume(TokenKind.LBRACKET)
        args: ArgList = []

        if self.current.kind is not TokenKind.RBRACKET:
            args.append(self._parse_argtype())
            while self.current.kind is TokenKind.COMMA:
                self.consume(TokenKind.COMMA)
                args.append(self._parse_argtype())

        self.consume(TokenKind.RBRACKET)
        return args

    # ArgType ::= ShapeType | IdentifierType | GenericType

    def _parse_argtype(self) -> ArgType:
        match (tok := self.current).kind:
            # ShapeType starts with "("
            case TokenKind.LPAREN:
                return self._parse_shape_type()

            # IdentifierType ArgList?
            case TokenKind.IDENT:
                ident = self._parse_identifier()
                if self.current.kind is TokenKind.LBRACKET:
                    arglist = self._parse_arglist()
                    return GenericType(id=ident, arglist=arglist)
                return ident

            case _:
                raise SyntaxError(
                    f"Expected ArgType at position {tok.pos}, got {tok.kind.name} {tok.value!r}"
                )

    # ShapeType ::= "(" ( ("..." | DimType) ("," DimType)*)? ")"

    def _parse_shape_type(self) -> ShapeType:
        self.consume(TokenKind.LPAREN)

        dims: list[EllipsisType | DimType] = []

        # check first token after "("
        match self.current.kind:
            case TokenKind.RPAREN:  # empty shape, exit early
                self.consume(TokenKind.RPAREN)
                return ()

            case TokenKind.ELLIPSIS:
                self.consume(TokenKind.ELLIPSIS)
                dims.append(...)

            case _:
                dim = self._parse_dim_type()
                dims.append(dim)

        # parse remaining dimensions
        while self.current.kind is TokenKind.COMMA:
            self.consume(TokenKind.COMMA)
            dims.append(self._parse_dim_type())

        self.consume(TokenKind.RPAREN)
        return tuple(dims)

    # DimType ::= "?" | Number | ("*"? IdentifierType)

    def _parse_dim_type(self) -> DimType:
        match (tok := self.current).kind:
            case TokenKind.ELLIPSIS:
                raise SyntaxError(
                    "At most one Ellipsis (...) is allowed per ShapeType, "
                    "and if present, it must be the first item."
                )

            case TokenKind.INT:
                self.consume(TokenKind.INT)
                return ConstantDim(int(tok.value))

            case TokenKind.IDENT:
                ident = self._parse_identifier()
                return StaticDim(ident)

            case TokenKind.STAR:
                self.consume(TokenKind.STAR)
                ident = self._parse_identifier()
                return DynamicDim(ident)

            case TokenKind.QMARK:
                self.consume(TokenKind.QMARK)
                return UnknownDim()

            case _:
                raise SyntaxError(
                    f"Expected DimType at position {tok.pos}, got {tok.kind.name} {tok.value!r}"
                )

    # IdentifierType ::= /[A-Za-z]\w*/

    def _parse_identifier(self) -> Identifier:
        match (tok := self.current).kind:
            case TokenKind.IDENT:
                self.consume(TokenKind.IDENT)
                return Identifier(tok.value)
            case _:
                raise SyntaxError(
                    f"Expected identifier at position {tok.pos}, "
                    f"got {tok.kind.name} {tok.value!r}"
                )


def parse_arglist(source: str) -> ArgList:
    parser = Parser(tokenize(source))
    return parser.parse_arglist()


def parse_signature(source: str) -> SignatureType:
    parser = Parser(tokenize(source))
    return parser.parse_signature()
