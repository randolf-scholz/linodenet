r"""Distributions for probabilistic modeling.

Two classes of models are of interest:

1. Normalizing flows.
2. Probabilistic circuits.

We considered the following libraries from the
[normalizing-flow library overview](https://github.com/janosh/awesome-normalizing-flows).
This historical comparison records release dates from 2024 and earlier.

| Library | No extra dependencies | Recorded release | Subclasses `Distribution` |
| --- | --- | --- | --- |
| [zuko](https://github.com/probabilists/zuko) | ✅ | 2024-01 | ✅ |
| [flowtorch](https://github.com/facebookincubator/flowtorch) | ✅ | 2022-04 | ✅ |
| [nflows](https://github.com/bayesiains/nflows) | ❌ | 2020-12 | ✅ |
| [normalizing-flows](https://github.com/VincentStimper/normalizing-flows) | ✅ | 2023-11 | ❌ |
| [FrEIA](https://github.com/vislearn/FrEIA) | ✅ | 2022-04 | ❌ |
| [pyro](https://github.com/pyro-ppl/pyro) | ✅ | 2024-06 | ✅ |


We selected `zuko` and `flowtorch` for further consideration. Our requirements are:

1. Represent the latent distribution as a `torch.distributions.Distribution`.
2. Represent the data distribution as a `torch.distributions.Distribution`.
3. Provide a library of invertible transformations.

Desired features include:

- Useful `typing.Protocol` classes and abstract base classes.
- `torch.compile` support.

The distributions and transformations should support:

1. Sampling from the latent space, optionally returning log-likelihoods.
2. Sampling from the data space, including reparameterized sampling (`rsample`),
   optionally returning log-likelihoods.
3. Forward and inverse transformations.
4. Computing the log determinant of the Jacobian.
5. Evaluating the log density.
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
    "Normal",
    "Uniform",
    "MixtureSameFamily",
    "MarchenkoPastur",
]

from .base import Distribution, DistributionBase
from .categorical import Categorical
from .empirical import Dirac, Empirical
from .gaussian import MultiHeadGaussian, MultivariateNormal
from .marchenko_pastur import MarchenkoPastur
from .mixture import Mixture, MixtureSameFamily
from .uniform import Uniform
from .univariate_normal import Normal

DISTRIBUTIONS: dict[str, type[Distribution]] = {
    "Dirac"             : Dirac,
    "Empirical"         : Empirical,
    "Categorical"       : Categorical,
    "Normal"            : Normal,
    "Uniform"           : Uniform,
    "MultiHeadGaussian" : MultiHeadGaussian,
    "MixtureSameFamily" : MixtureSameFamily,
    "MarchenkoPastur"   : MarchenkoPastur,
}  # fmt: skip
r"""Dictionary of all available distributions."""
