r"""Parsing and representation of function signatures for the `@signature` decorator."""

__all__ = [
    # Config
    "LAZY_EVAL",
    # Types
    "FnType",
    "SignatureType",
    "ShapeType",
    "ArgType",
    "Arg",
    "KeyArg",
    "ArgList",
    # Classes
    "Identifier",
    "GenericType",
    "Parser",
    "Token",
    "TokenKind",
    "DimKind",
    "Sign",
    # dimension types
    "Dim",
    "DimType",
    "ConstantDim",
    "StaticDim",
    "AffineDim",
    "VariadicDim",
    "DynamicDim",
    # Functions
    "is_identifier",
    "tokenize",
    "signature",
]

from abc import abstractmethod
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from types import EllipsisType
from typing import ClassVar, Final, Literal, ReadOnly, TypeIs, overload

type ShapeType = tuple[DimType, ...] | tuple[EllipsisType, *tuple[DimType, ...]]
type DimType = ConstantDim | StaticDim | VariadicDim | AffineDim | DynamicDim
type ArgType = ShapeType | Identifier | GenericType | FnType
type ArgList = list[Arg]
type QMARK = Literal["?"]
type SignatureType = FnType


def _shape_to_str(shape: ShapeType, /) -> str:
    if len(shape) > 0 and shape[0] is Ellipsis:
        return f"(..., {', '.join(str(dim) for dim in shape[1:])})"
    return f"({', '.join(str(dim) for dim in shape)})"


def _arg_to_str(arg: ArgType, /) -> str:
    if isinstance(arg, tuple):
        return _shape_to_str(arg)
    return str(arg)


class Identifier(str):
    r"""String representing a Python identifier without a leading underscore.

    Construction raises `ValueError` if the name fails `is_identifier`.
    """

    __slots__ = ()

    def __init__(self, name: str, /) -> None:
        if not is_identifier(name):
            raise ValueError(f"Invalid identifier: {name}")


def is_identifier(s: str, /) -> TypeIs[Identifier]:
    r"""Check whether a string is a Python identifier without a leading underscore."""
    return s.isidentifier() and not s.startswith("_")


@dataclass(frozen=True, slots=True)
class Arg:
    r"""Argument type paired with a flag indicating whether it is optional.

    An optional argument is rendered with a trailing `?`.
    """

    value: ArgType
    optional: bool

    def __str__(self) -> str:
        return f"{_arg_to_str(self.value)}{'?' if self.optional else ''}"


@dataclass(frozen=True, slots=True)
class KeyArg:
    r"""Named argument type paired with a flag indicating whether it is optional.

    The string representation includes the value and optional `?` marker;
    the name is stored separately.
    """

    name: Identifier
    value: ArgType
    optional: bool

    def __str__(self) -> str:
        return f"{_arg_to_str(self.value)}{'?' if self.optional else ''}"


@dataclass(frozen=True, slots=True)
class GenericType:
    r"""Type identifier with a list of type arguments, such as `Tensor[(m, n)]`."""

    id: Identifier
    arglist: ArgList

    def __str__(self) -> str:
        return f"{self.id!s}[{','.join(map(str, self.arglist))}]"


class DimKind(StrEnum):
    r"""Enumeration of dimension kinds."""

    CONSTANT = "constant"  # '1', '2', '3', ... fixed-size dimension
    STATIC = "static"  # 'n' static shape / single axis
    VARIADIC = "variadic"  # '*n' static shape / bundle of axes
    DYNAMIC = "dynamic"  # '$n' variable sized axis
    AFFINE = "affine"  # "2n", "3n+1", "u+v"


@dataclass(frozen=True, slots=True)
class Dim:
    r"""Base representation for a shape dimension or a bundle of axes."""

    # FIXME: Using https://peps.python.org/pep-0767/
    kind: ClassVar[ReadOnly[DimKind]]  # type: ignore

    @abstractmethod
    def __str__(self) -> str: ...


@dataclass(frozen=True, slots=True)
class ConstantDim(Dim):
    r"""Literal integer dimension, such as `3`, or a signed affine offset.

    The class stores the integer without validating dimension size bounds.
    """

    kind: ClassVar[Final[Literal[DimKind.CONSTANT]]] = DimKind.CONSTANT

    value: int  # (>= 0)

    def __str__(self) -> str:
        return f"{self.value!s}"


@dataclass(frozen=True, slots=True)
class StaticDim(Dim):
    r"""Named single-axis size fixed for a call, such as `n`."""

    kind: ClassVar[Final[Literal[DimKind.STATIC]]] = DimKind.STATIC

    value: Identifier

    def __str__(self) -> str:
        return f"{self.value!s}"


@dataclass(frozen=True, slots=True)
class VariadicDim(Dim):
    r"""Named bundle of axes fixed for a call, such as `*xs`."""

    kind: ClassVar[Final[Literal[DimKind.VARIADIC]]] = DimKind.VARIADIC

    value: Identifier

    def __str__(self) -> str:
        return f"*{self.value!s}"


@dataclass(frozen=True, slots=True)
class DynamicDim(Dim):
    r"""Named single-axis size that may vary across calls, such as `$n`."""

    kind: ClassVar[Final[Literal[DimKind.DYNAMIC]]] = DimKind.DYNAMIC

    value: Identifier

    def __str__(self) -> str:
        return f"${self.value!s}"


class Sign(IntEnum):
    r"""Enumeration of signs for affine dimensions."""

    POS = +1
    NEG = -1


@dataclass(frozen=True, slots=True)
class AffineDim(Dim):
    r"""Affine combination of static or dynamic dimensions with an optional offset.

    Each term contains a sign $sᵢ$, integer coefficient $cᵢ$, and variable $xᵢ$.
    With offset $b$ (zero when omitted), the dimension is:

    $$ d = ∑ᵢ sᵢ cᵢ xᵢ + b $$

    Each static or dynamic variable may appear at most once; duplicate variables
    raise `ValueError`. Supported expressions include:

    - `2n - 1`
    - `3n + 1`
    - `u + v`
    - `a - 2b + 3c - 4d + 5`
    - `2$n + m`
    """

    kind: ClassVar[Final[Literal[DimKind.AFFINE]]] = DimKind.AFFINE

    terms: Sequence[tuple[Sign, ConstantDim, StaticDim | DynamicDim]]
    offset: ConstantDim | None = None

    def __post_init__(self) -> None:
        # if len(self.terms) == 0:
        # raise ValueError("AffineDim must have at least one term")
        if len(set(self.variables)) != len(self.terms):
            raise ValueError("AffineDim variables must be unique")

    @property
    def variables(self) -> list[StaticDim | DynamicDim]:
        return [var for _, _, var in self.terms]

    @property
    def coefficients(self) -> list[ConstantDim]:
        return [coef for _, coef, _ in self.terms]

    @property
    def signs(self) -> list[Sign]:
        return [sign for sign, _, _ in self.terms]

    def __str__(self) -> str:
        parts: list[str] = []

        for sign, coef, var in self.terms:
            body = f"{'' if coef.value == 1 else coef.value}{var!s}"
            if not parts:
                parts.append(body if sign is Sign.POS else f"-{body}")
                continue
            operator = "+" if sign is Sign.POS else "-"
            parts.append(f"{operator} {body}")

        if self.offset is not None:
            offset = self.offset.value
            if not parts:
                return str(offset)
            operator = "+" if offset >= 0 else "-"
            parts.append(f"{operator} {abs(offset)}")

        return " ".join(parts)


@dataclass(frozen=True, slots=True)
class FnType:
    r"""Parsed function signature with input and output argument lists.

    Both sides of `->` may contain a single argument or a bracketed list:

    - `arg1 -> ret1`: One input and one output; `[arg1] -> ret1` is also allowed.
    - `[arg1, arg2] -> ret1`: Multiple inputs and one output.
    - `arg1 -> [ret1, ret2]`: One input and multiple outputs.
    - `[arg1, arg2] -> [ret1, ret2]`: Multiple inputs and multiple outputs.

    Empty lists represent no arguments or no return values. A trailing `?` marks
    an optional argument. In bracketed lists, required arguments must precede
    optional arguments.

    Arguments can be tensor shapes such as `(m, n)`, generic types such as
    `Tensor[(m, n)]`, identifiers such as `int`, or nested function types such as
    `{A -> B}`. Top-level signatures may omit their outer braces; nested function
    types require braces.

    Note:
        Tensor shapes are represented internally as tuples of dimension objects
        and, optionally, a leading `Ellipsis`. Their textual syntax includes:

        - `3`: A literal axis size.
        - `n`: A named static axis size.
        - `*xs`: A named bundle of axes.
        - `$n`: A named dynamic axis size.
        - `2n + 1`: An affine combination of dimensions.
        - `...`: Leading axes to vectorize over; at most one is allowed per shape,
            and it must be first.

        The empty shape `()` represents a scalar. Double-star dimensions such as
        `**xs` are not supported by the parser.
    """

    argument_types: ArgList
    return_types: ArgList

    def __str__(self) -> str:
        r"""Convert the function type to its canonical braced string form."""
        arg_strs = list(map(str, self.argument_types))
        ret_strs = list(map(str, self.return_types))

        arg_part = arg_strs[0] if len(arg_strs) == 1 else f"[{', '.join(arg_strs)}]"
        ret_part = ret_strs[0] if len(ret_strs) == 1 else f"[{', '.join(ret_strs)}]"
        return f"{{{arg_part} -> {ret_part}}}"


class TokenKind(StrEnum):
    r"""Token categories for the argument-list and function-signature parser."""

    IDENT = r"[A-Za-z]\w*]"
    NUMBER = r"\d+"
    ELLIPSIS = "..."
    QMARK = "?"
    STAR = "*"
    ARROW = "->"
    LBRACKET = "["
    RBRACKET = "]"
    LPAREN = "("
    RPAREN = ")"
    COMMA = ","
    PLUS = "+"
    MINUS = "-"
    DOLLAR = "$"
    LBRACE = "{"
    RBRACE = "}"
    EOF = "EOF"


@dataclass(frozen=True, slots=True)
class Token:
    r"""Token with its source character position, category, and text.

    Identifier and number tokens require a custom value. All other tokens use
    their kind's fixed text and reject custom values.
    """

    pos: int  # character index in original string
    value: str
    kind: TokenKind  # "IDENT", "INT", "ELLIPSIS", "STAR", "DSTAR", symbols like "[", "]", "(", ")", ",", "EOF"

    @overload
    def __init__(
        self, pos: int, kind: Literal[TokenKind.IDENT, TokenKind.NUMBER], value: str
    ) -> None: ...
    @overload
    def __init__(self, pos: int, kind: TokenKind) -> None: ...
    def __init__(self, pos: int, kind: TokenKind, value: str | None = None) -> None:
        if kind in {TokenKind.IDENT, TokenKind.NUMBER} and value is None:
            raise AssertionError("IDENT and INT tokens require a value")
        if kind not in {TokenKind.IDENT, TokenKind.NUMBER} and value is not None:
            raise AssertionError("Only IDENT and INT tokens may carry custom values")
        if kind not in {TokenKind.IDENT, TokenKind.NUMBER}:
            assert value is None
            value = kind.value

        object.__setattr__(self, "pos", pos)
        object.__setattr__(self, "value", str(value))
        object.__setattr__(self, "kind", TokenKind(kind))

    def __repr__(self) -> str:
        return f"Token({self.kind}, {self.value!r}, pos={self.pos})"


def tokenize(source: str, /) -> Iterator[Token]:
    r"""Tokenize a signature DSL string.

    Args:
        source: Signature string to tokenize.

    Yields:
        Token: One token per lexical element, retaining its character position in
            `source`. A valid input ends with an EOF token at `len(source)`.

    Raises:
        SyntaxError: If an unexpected or invalid character is encountered.

    Note:
        - Whitespace is skipped.
        - Identifiers start with a letter and continue with letters, digits, or
            underscores, including Unicode letters and digits.
        - Integer literals contain digits; signs are separate tokens.
        - Punctuation includes `[ ] { } ( ) , * + - ? $`, the arrow `->`, and
            the ellipsis `...`.
    """
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
        if c in "[]{}(),*+-?$":
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
            yield Token(start, TokenKind.NUMBER, value)
            continue

        # If we get here, it's an invalid character
        raise SyntaxError(f"Unexpected character {c!r} at position {i}")

    yield Token(n, TokenKind.EOF)


def _has_wrapping_braces(source: str, /) -> bool:
    r"""Return whether `source` is enclosed by a single top-level `{...}` pair."""
    stripped = source.strip()
    if len(stripped) < 2 or stripped[0] != "{" or stripped[-1] != "}":
        return False

    depth = 0
    for idx, char in enumerate(stripped):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and idx != len(stripped) - 1:
                return False
            if depth < 0:
                return False

    return depth == 0


class Parser:
    r"""Entry points for functional recursive-descent parsing.

    Each entry point tokenizes its input once into a shared tuple ending in EOF.
    Private parsing functions receive that sequence and a token index, returning
    the parsed value and the next index without slicing or mutating the tokens.
    Complete input consumption is required. Lexical errors take precedence over
    grammar errors because tokenization finishes before parsing starts.
    """

    @staticmethod
    def parse_signature(arg: str, /) -> SignatureType:
        r"""Parse a complete function signature with optional outer braces.

        Raises:
            SyntaxError: If the first token cannot be read.
            RuntimeError: If a syntax error occurs after the first token is read.
                The original error is retained as the cause and its message is
                added as an exception note.
        """
        stream = tokenize(arg)
        first = next(stream)
        try:
            tokens = (first, *stream)
            result, pos = (
                _parse_fn_type(tokens, 0)
                if _has_wrapping_braces(arg)
                else _parse_fn_body(tokens, 0)
            )
            _check_eof(tokens, pos)
        except SyntaxError as parser_error:
            exc = RuntimeError(f"Failed to parse signature {arg!r}")
            exc.add_note(str(parser_error))
            raise exc from parser_error

        return result

    @staticmethod
    def parse_arglist(arg: str, /) -> ArgList:
        r"""Parse a complete bracketed argument list, which may be empty.

        Required arguments must precede optional arguments marked with `?`.
        Unexpected tokens or trailing input raise `SyntaxError`.
        """
        stream = tokenize(arg)
        first = next(stream)
        try:
            tokens = (first, *stream)
            result, pos = _parse_arglist(tokens, 0)
            _check_eof(tokens, pos)
        except SyntaxError as exc:
            exc.add_note(f"Failed to parse arglist from {arg!r}")
            raise

        return result

    @staticmethod
    def parse_identifier(arg: str, /) -> Identifier:
        r"""Parse a single identifier and reject trailing tokens.

        Unexpected tokens or trailing input raise `SyntaxError`.
        """
        stream = tokenize(arg)
        first = next(stream)
        try:
            tokens = (first, *stream)
            result, pos = _parse_identifier(tokens, 0)
            _check_eof(tokens, pos)
        except SyntaxError as exc:
            exc.add_note(f"Failed to parse identifier from {arg!r}")
            raise

        return result


def _consume(tokens: Sequence[Token], pos: int, /, *, kind: TokenKind) -> int:
    r"""Require a non-EOF token of `kind` and return the next token index.

    The shared sequence must end in EOF, and `pos` must index an existing token.
    Parsing functions leave EOF unconsumed so truncated input raises `SyntaxError`.
    """
    if (tok := tokens[pos]).kind is not kind:
        raise SyntaxError(
            f"Expected {kind.name} ({kind.value!r}) at position {tok.pos}, "
            f"got {tok.kind.name} ({tok.value!r})"
        )
    return pos + 1


def _check_eof(tokens: Sequence[Token], pos: int, /) -> None:
    if (tok := tokens[pos]).kind != TokenKind.EOF:
        raise SyntaxError(
            f"Unexpected token {tok.kind.name} {tok.value!r} at position {tok.pos}, "
            "expected end of input"
        )


# FnType ::= "{" FnBody "}"
def _parse_fn_type(tokens: Sequence[Token], pos: int, /) -> tuple[FnType, int]:
    r"""Parse a braced function type."""
    pos = _consume(tokens, pos, kind=TokenKind.LBRACE)
    fn_type, pos = _parse_fn_body(tokens, pos)
    pos = _consume(tokens, pos, kind=TokenKind.RBRACE)
    return fn_type, pos


# FnBody ::= (ArgList | Arg) "->" (ArgList | Arg)
def _parse_fn_body(tokens: Sequence[Token], pos: int, /) -> tuple[FnType, int]:
    r"""Parse an unbraced function type body."""
    match tokens[pos].kind:
        case TokenKind.LBRACKET:
            lhs, pos = _parse_arglist(tokens, pos)
        case _:
            arg, pos = _parse_arg(tokens, pos)
            lhs = [arg]

    pos = _consume(tokens, pos, kind=TokenKind.ARROW)

    match tokens[pos].kind:
        case TokenKind.LBRACKET:
            rhs, pos = _parse_arglist(tokens, pos)
        case _:
            arg, pos = _parse_arg(tokens, pos)
            rhs = [arg]

    return FnType(lhs, rhs), pos


# ArgList ::= "[" (Arg ("," Arg)*)? "]"
def _parse_arglist(tokens: Sequence[Token], pos: int, /) -> tuple[ArgList, int]:
    pos = _consume(tokens, pos, kind=TokenKind.LBRACKET)
    if tokens[pos].kind is TokenKind.RBRACKET:
        return [], pos + 1

    arg, pos = _parse_arg(tokens, pos)
    args = [arg]
    while tokens[pos].kind is TokenKind.COMMA:
        pos += 1
        next_arg, next_pos = _parse_arg(tokens, pos)
        if arg.optional and not next_arg.optional:
            raise SyntaxError(
                f"Non-optional argument at position {tokens[pos].pos} "
                "cannot follow an optional argument"
            )
        arg, pos = next_arg, next_pos
        args.append(arg)

    return args, _consume(tokens, pos, kind=TokenKind.RBRACKET)


# Arg ::= ArgType "?"?
def _parse_arg(tokens: Sequence[Token], pos: int, /) -> tuple[Arg, int]:
    value: ArgType
    match (tok := tokens[pos]).kind:
        case TokenKind.LBRACE:
            value, pos = _parse_fn_type(tokens, pos)

        case TokenKind.LPAREN:
            value, pos = _parse_shape_type(tokens, pos)

        case TokenKind.IDENT:
            ident, pos = _parse_identifier(tokens, pos)
            match tokens[pos].kind:
                case TokenKind.LBRACKET:
                    arglist, pos = _parse_arglist(tokens, pos)
                    value = GenericType(id=ident, arglist=arglist)
                case _:
                    value = ident

        case _:
            raise SyntaxError(
                f"Expected ArgType at position {tok.pos}, "
                f"got {tok.kind.name} {tok.value!r}"
            )

    optional = tokens[pos].kind is TokenKind.QMARK
    return Arg(value, optional=optional), pos + optional


# ShapeType ::= "(" ( ("..." | DimType) ("," DimType)* ","?)? ")"
def _parse_shape_type(tokens: Sequence[Token], pos: int, /) -> tuple[ShapeType, int]:
    pos = _consume(tokens, pos, kind=TokenKind.LPAREN)
    with_ellipsis = False
    dims: list[DimType] = []

    match tokens[pos].kind:
        case TokenKind.RPAREN:
            return (), pos + 1
        case TokenKind.ELLIPSIS:
            pos += 1
            with_ellipsis = True
        case _:
            dim, pos = _parse_dim_type(tokens, pos)
            dims.append(dim)

    while tokens[pos].kind is TokenKind.COMMA:
        pos += 1
        if tokens[pos].kind is TokenKind.RPAREN:
            break
        dim, pos = _parse_dim_type(tokens, pos)
        dims.append(dim)

    pos = _consume(tokens, pos, kind=TokenKind.RPAREN)
    shape: ShapeType = (Ellipsis, *dims) if with_ellipsis else tuple(dims)
    return shape, pos


# DimType ::= AffineDim | "*" IdentifierType
def _parse_dim_type(tokens: Sequence[Token], pos: int, /) -> tuple[DimType, int]:
    match (tok := tokens[pos]).kind:
        case TokenKind.ELLIPSIS:
            raise SyntaxError(
                "At most one Ellipsis (...) is allowed per ShapeType, "
                "and if present, it must be the first item."
            )

        case TokenKind.STAR:
            ident, pos = _parse_identifier(tokens, pos + 1)
            return VariadicDim(ident), pos

        case _:
            try:
                affine_dim, pos = _parse_affine_dim(tokens, pos)
            except SyntaxError as exc:
                raise SyntaxError(
                    f"Expected DimType at position {tok.pos}, got {tok.kind.name} {tok.value!r}"
                ) from exc

            # Simplify single-term affine dims.
            match affine_dim.terms, affine_dim.offset:
                case [], None:
                    raise AssertionError(
                        "AffineDim must have at least one term or offset"
                    )
                case [], offset:
                    assert offset is not None
                    return offset, pos
                case [(Sign.POS, ConstantDim(1), var)], None:
                    return var, pos
                case _:
                    return affine_dim, pos


def _parse_affine_dim(tokens: Sequence[Token], pos: int, /) -> tuple[AffineDim, int]:
    terms: list[tuple[Sign, ConstantDim, StaticDim | DynamicDim]] = []
    offset: ConstantDim | None = None
    sign: Sign
    coef: ConstantDim
    var: StaticDim | DynamicDim

    while True:
        # Parse sign.
        match tokens[pos].kind:
            case TokenKind.PLUS:
                pos += 1
                sign = Sign.POS
            case TokenKind.MINUS:
                pos += 1
                sign = Sign.NEG
            case _ if not terms:
                sign = Sign.POS
            case _:
                break

        # Parse coefficient.
        match (tok := tokens[pos]).kind:
            case TokenKind.NUMBER:
                coef = ConstantDim(int(tok.value))
                pos += 1
            case _:
                coef = ConstantDim(1)

        # Parse variable, or finish with an offset term.
        match tokens[pos].kind:
            case TokenKind.IDENT:
                ident, pos = _parse_identifier(tokens, pos)
                var = StaticDim(ident)
            case TokenKind.DOLLAR:
                ident, pos = _parse_identifier(tokens, pos + 1)
                var = DynamicDim(ident)
            case _:
                offset = ConstantDim(sign * coef.value)
                break

        terms.append((sign, coef, var))

    return AffineDim(terms=terms, offset=offset), pos


# IdentifierType ::= /[A-Za-z]\w*/
def _parse_identifier(tokens: Sequence[Token], pos: int, /) -> tuple[Identifier, int]:
    match (tok := tokens[pos]).kind:
        case TokenKind.IDENT:
            return Identifier(tok.value), pos + 1
        case _:
            raise SyntaxError(
                f"Expected identifier at position {tok.pos}, "
                f"got {tok.kind.name} {tok.value!r}"
            )


LAZY_EVAL = False
r"""Default for the decorator's `lazy` argument, captured when the class is defined.

`False` parses during decorator construction; `True` defers parsing until the
`parsed` property or string representation is requested.
"""


class signature:
    r"""Attach a parsed function signature as metadata without wrapping the function.

    The decorator sets `fn.signature` and returns the same function. It does not
    enforce argument types or tensor shapes at runtime.

    Args:
        sig_string: Signature text accepted by `Parser.parse_signature`.
        lazy: Whether to defer parsing until the `parsed` property or string
            representation is requested. Defaults to `LAZY_EVAL` as captured when
            the class is defined.

    Signature DSL:

    - `3`: An axis of size three.
    - `x`: A named static axis size.
    - `*xs`: A named bundle of axes.
    - `$n`: A named dynamic axis size.
    - `2n + 1`: An affine combination of dimensions.
    - `...`: Leading axes to vectorize over; it must be first in its shape.

    Append `?` to an argument type to mark it as optional. Required arguments must
    precede optional arguments in bracketed lists. Double-star dimensions such as
    `**xs` are not supported.
    """

    def __init__(self, sig_string: str, /, lazy: bool = LAZY_EVAL) -> None:
        self.original_signature = sig_string
        if not lazy:
            self._parse()

    def _parse(self) -> None:
        # parse the signature string and cache the result
        self._parsed = Parser.parse_signature(self.original_signature)

    @property
    def parsed(self) -> SignatureType:
        if not hasattr(self, "_parsed"):
            self._parse()
        return self._parsed

    def __repr__(self) -> str:
        return str(self.parsed)

    def __str__(self) -> str:
        return str(self.parsed)

    def __call__[Fn: Callable](self, fn: Fn) -> Fn:
        r"""Set `fn.signature` to this decorator instance and return `fn` unchanged."""
        fn.signature = self  # type: ignore
        return fn
