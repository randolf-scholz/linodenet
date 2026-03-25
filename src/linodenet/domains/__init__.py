r"""WORK IN PROGRESS.

Domains should allow:

1. checking membership of tensors
2. checking subset relations between domains
3. performing some basic operations (e.g. product of domains, union, intersection)
"""

__all__ = [
    # Protocols
    "Domain",
    "DomainMapping",
    "Inverse",
    "Union",
    "Intersection",
    "Join",
    # Domains
    "Interval",
    "MatrixDomain",
    "RealDomain",
    # Enums
    "ScalarDomains",
    "VectorDomains",
    "MatrixDomains",
    "TensorDomains",
]


from .base import Domain, DomainMapping, Intersection, Inverse, Join, Union
from .matrix_domains import MatrixDomain, MatrixDomains
from .scalar_domains import Interval, RealDomain, ScalarDomains
from .tensor_domains import TensorDomains
from .vector_domains import VectorDomains
