r"""Module for parsing and representing function signatures between vector spaces.

This module provides a small DSL for describing input/output shapes and a
`@signature` decorator to attach those signatures to functions. The grammar is
documented in `src/signatures/signature.bnf`, and the parser is implemented in
`src/signatures/signatures.py`.

Dimension types:
    Constant: A fixed integer axis size like `3`.
    Static: A named axis size fixed per call, like `n`.
    Variadic: A bundle of named axes fixed per call, like `*xs`.
    Dynamic: A named axis size that may vary across calls, like `$n`.
    Affine: A linear combination of static/dynamic dims, like `2n+1` or `u+v`.


Each dimension can occur both in the input and output shapes, and the same name implies equality.
For example, the signature `(m, n) -> (n, m)` describes a function that
takes an `(m, n)`-shaped input and produces an `(n, m)`-shaped output, such as a transpose operation.

Examples:
    Basic scalar-to-scalar:
        >>> signature("() -> ()")

    vector-to-vector, constant size (fixed at compile time):
        >>> signature("(3) -> (3)")

    Vector-to-vector, static size (fixed at runtime):
        >>> signature("(m) -> (n)")
        E.g. `nn.Linear.forward`

    Sequence-to-sequence, dynamic size (may vary across calls):
        >>> signature("($n) -> ($n)")
        E.g. `torch.cumsum`

    Ellipsis (vectorization) (at most one per signature):
        >>> signature("(...) -> (...)")
        E.g. `torch.tanh`

    Variadic axes (bundles of static axes):
        >>> signature("(*dims) -> ()")
        E.g. `torch.sum`

    Affine dims:
        >>> signature("(n) -> (n-1)")
        E.g. `torch.diff`

Examples:
    Matrix-multiplication:
        >>> signature("[(m, n), (n, p)] -> (m, p)")
        E.g. `torch.matmul`

    Attention (single head):
        >>> signature("[(b, $n, e_q), (b, $n, e_k), (b, $n, e_v)] -> (b, $n, e)")
        E.g. `torch.nn.functional.scaled_dot_product_attention`

    Matrix factorization (full SVD):
        >>> signature("(m, n) -> [(m, k), (k,), (n, k)]")
"""

__all__ = [
    # Types
    "FnType",
    "SignatureType",
    "ShapeType",
    "Arg",
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
    # Functions
    "is_identifier",
    "tokenize",
    "signature",
]
from .signatures import (
    Arg,
    ArgList,
    ArgType,
    ConstantDim,
    Dim,
    DimKind,
    DimType,
    FnType,
    GenericType,
    Identifier,
    Parser,
    ShapeType,
    SignatureType,
    StaticDim,
    Token,
    TokenKind,
    VariadicDim,
    is_identifier,
    signature,
    tokenize,
)
