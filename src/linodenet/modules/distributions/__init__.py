"""Distributions for probabilistic modeling.

There are two interesting classes of models:

1. Normalizing Flows.
2. Probabilistic Circuits.

There are several available libraries for normalizing flows, e.g. [1]_.
We considered the following libraries:

- [zuko](https://github.com/probabilists/zuko)
- [nflows](https://github.com/bayesiains/nflows)
- [flowtorch](https://github.com/facebookincubator/flowtorch)
- [normalizing-flows](https://github.com/VincentStimper/normalizing-flows)
- [freia](https://github.com/vislearn/FrEIA)

A first look at these gives:

|                   | extra deps                          | last release | subclass Distribution |
|-------------------|-------------------------------------|--------------|-----------------------|
| zuko              | --                                  | 2024-01      | ✅                     |
| flowtorch         | --                                  | 2022-04      | ✅                     |
| nflows            | tensorboard, tqdm, matplotlib, umnn | 2020-12      | ✅                     |
| normalizing-flows | --                                  | 2023-11      | ❌                     |
| freia             | --                                  | 2022-04      | ❌                     |

From which we already decide only to consider `zuko` and `flowtorch`.

We want/need to be able to calculate the following:

1. Have the latent distribution as a `torch.distributions.Distribution`
2. Have the data distribution as a `torch.distributions.Distribution`
3. Library of invertible transformations

We would like to have:

- useful `Protocol`-classes and abstract base classes
- `torch.jit.script` support
- `torch.compile` support

1. sample from the latent space
    - sample and also give log-likelihoods
2. sample from the data space (`rsample`)
    - sample and also give log-likelihoods
1. forward/inverse
2. log-determinant of the Jacobian
2. logarithm of the density

References:
    ..[1] https://github.com/janosh/awesome-normalizing-flows
"""
