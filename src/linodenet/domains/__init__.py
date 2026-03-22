r"""WORK IN PROGRESS.

Domains should allow:

1. checking membership of tensors
2. checking subset relations between domains
3. performing some basic operations (e.g. product of domains, union, intersection)
"""

__all__ = [
    "Domain",
    "Interval",
    "IntervalUnion",
    "ScalarDomains",
    "TensorDomains",
    "VectorDomains",
    "MatrixDomains",
]


from .base import Domain
from .matrix_domains import MatrixDomains
from .scalar_domains import Interval, IntervalUnion, ScalarDomains
from .tensor_domains import TensorDomains
from .vector_domains import VectorDomains
