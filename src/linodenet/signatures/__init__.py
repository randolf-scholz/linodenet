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
    "VariadicDim",
    "UnknownDim",
    # Functions
    "is_identifier",
    "tokenize",
    "signature",
]
from linodenet.signatures.signatures import (
    ArgList,
    ArgType,
    ConstantDim,
    Dim,
    DimKind,
    DimType,
    GenericType,
    Identifier,
    Parser,
    ShapeType,
    SignatureType,
    StaticDim,
    Token,
    TokenKind,
    UnknownDim,
    VariadicDim,
    is_identifier,
    signature,
    tokenize,
)
