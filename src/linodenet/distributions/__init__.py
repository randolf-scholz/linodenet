r"""Distributions for probabilistic modeling.

There are two interesting classes of models:

1. Normalizing Flows.
2. Probabilistic Circuits.

There are several available libraries for normalizing flows [0]_, we considered the following:

+----------------------+---------------+--------------+---------------------------+
| library              | no extra deps | last release | subclasses `Distribution` |
+======================+===============+==============+===========================+
| `zuko`_              | ✅️           | 2024-01      | ✅️                       |
+----------------------+---------------+--------------+---------------------------+
| `flowtorch`_         | ✅️           | 2022-04      | ✅️                       |
+----------------------+---------------+--------------+---------------------------+
| `nflows`_            | ❌️           | 2020-12      | ✅️                       |
+----------------------+---------------+--------------+---------------------------+
| `normalizing-flows`_ | ✅️           | 2023-11      | ❌️                       |
+----------------------+---------------+--------------+---------------------------+
| `freia`_             | ✅️           | 2022-04      | ❌️                       |
+----------------------+---------------+--------------+---------------------------+
| `pyro`_              | ✅️           | 2024-06      | ✅️                       |
+----------------------+---------------+--------------+---------------------------+


From which we already decide only to consider `zuko`_ and `flowtorch`_.
We want/need to be able to calculate the following:

1. Have the latent distribution as a `torch.distributions.Distribution`
2. Have the data distribution as a `torch.distributions.Distribution`
3. Library of invertible transformations

We would like to have:

- useful `typing.Protocol`-classes (`Protocol`) and abstract base classes
- `torch.jit.script` support
- `torch.compile` support

1. sample from the latent space
    - sample and also give log-likelihoods
2. sample from the data space (`rsample`)
    - sample and also give log-likelihoods
3. forward/inverse
4. log-determinant of the Jacobian
5. logarithm of the density

References:
    .. [0] https://github.com/janosh/awesome-normalizing-flows
    .. _zuko: https://github.com/probabilists/zuko
    .. _flowtorch: https://github.com/facebookincubator/flowtorch
    .. _nflows: https://github.com/bayesiains/nflows
    .. _normalizing-flows: https://github.com/VincentStimper/normalizing-flows
    .. _freia: https://github.com/vislearn/FrEIA
    .. _pyro: https://github.com/pyro-ppl/pyro
    .. target-notes::
"""

__all__ = [
    # constants
    "DISTRIBUTIONS",
    # types
    # protocols & ABCs
    "Distribution",
    "DistributionBase",
    # classes
    "Dirac",
    "Empirical",
    "MultivariateNormal",
    "Mixture",
    "MultiHeadGaussian",
    "Categorical",
    "MixtureSameFamily",
]

from .base import Distribution, DistributionBase
from .categorical import Categorical
from .empirical import Dirac, Empirical
from .gaussian import MultiHeadGaussian, MultivariateNormal
from .mixture import Mixture, MixtureSameFamily

DISTRIBUTIONS: dict[str, type[Distribution]] = {
    "Dirac"              : Dirac,
    "Empirical"          : Empirical,
    "Categorical"        : Categorical,
    "MultiHeadGaussian"  : MultiHeadGaussian,
    "MixtureSameFamily"  : MixtureSameFamily,
}  # fmt: skip
r"""Dictionary of all available distributions."""
