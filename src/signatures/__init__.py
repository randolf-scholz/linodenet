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
        {() -> ()}

    vector-to-vector, constant size (fixed at compile time):
        >>> signature("(3) -> (3)")
        {(3) -> (3)}

    Vector-to-vector, static size (fixed at runtime), e.g. `nn.Linear.forward`:
        >>> signature("(m) -> (n)")
        {(m) -> (n)}

    Sequence-to-sequence, dynamic size (may vary across calls), e.g. `torch.cumsum`:
        >>> signature("($n) -> ($n)")
        {($n) -> ($n)}

    Ellipsis (vectorization) (at most one per signature), e.g. `torch.tanh`:
        >>> signature("(...) -> (...)")
        {(...) -> (...)}

    Variadic axes (bundles of static axes), e.g. `torch.sum`:
        >>> signature("(*dims) -> ()")
        {(*dims) -> ()}

    Affine dims (linear/affine combinations), e.g. `torch.diff`:
        >>> signature("(n) -> (n-1)")
        {(n) -> (n - 1)}

Examples:
    Matrix-multiplication, e.g. `torch.matmul`:
        >>> signature("[(m, n), (n, p)] -> (m, p)")
        {[(m, n), (n, p)] -> (m, p)}

    Attention (single head), e.g. `torch.nn.functional.scaled_dot_product_attention`:
        >>> signature("[(b, $n, e_q), (b, $n, e_k), (b, $n, e_v)] -> (b, $n, e)")
        {[(b, $n, e_q), (b, $n, e_k), (b, $n, e_v)] -> (b, $n, e)}

    Matrix factorization (full SVD):
        >>> signature("(m, n) -> [(m, k), (k,), (n, k)]")
        {(m, n) -> [(m, k), (k), (n, k)]}
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
