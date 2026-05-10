# LinODEnet — 𝗟𝗶𝗻ear 𝗢rdinary 𝗗ifferential 𝗘quation 𝗡𝗲𝘁work

[INSTALLATION](#installation) | [DOCUMENTATION](#documentation) | [CONTRIBUTING](CONTRIBUTING.md) | [CHANGELOG](CHANGELOG.md) | [LICENSE](LICENSE)

![model-sketch](lssm.png)

## Introduction

LinODEnet is a Python library for building components for state space models, with a particular focus on irregularly sampled time series. It is aimed at forecasting settings where observations do not arrive on a fixed grid and where continuous-time structure matters for modeling latent dynamics, interpolation, and prediction.

The library provides reusable building blocks rather than a monolithic end-to-end forecasting framework. The goal is to make it easier to compose models that combine state space ideas, ordinary differential equations, and neural network components for irregular time series.

## Installation

Install the package into an existing Python environment directly from the git repository:

```bash
python -m pip install "git+https://github.com/randolf-scholz/linodenet.git"
```

Install a development version with `uv`:

```bash
git clone https://github.com/randolf-scholz/linodenet.git
cd linodenet
uv sync
```

## Other libraries

If you are looking for broader time series ecosystems, several other libraries cover adjacent use cases:

- [Darts](https://github.com/unit8co/darts) provides a high-level forecasting API with many classical and deep learning models.
- [GluonTS](https://github.com/awslabs/gluonts) focuses on probabilistic time series modeling and forecasting, with strong support for deep learning workflows.
- [sktime](https://github.com/sktime/sktime) offers a unified framework for forecasting, classification, regression, and related time series tasks.
- [statsmodels](https://github.com/statsmodels/statsmodels) includes many classical statistical time series models such as ARIMA and state space methods.
- [Nixtla](https://github.com/Nixtla) maintains libraries such as `statsforecast`, `mlforecast`, and `neuralforecast` for scalable statistical and machine learning forecasting.
- [PyTorch Forecasting](https://github.com/sktime/pytorch-forecasting) provides deep learning models and dataset utilities built on top of PyTorch.
- [pyrregular](https://github.com/fspinna/pyrregular) focuses on irregular time series classification, providing a unified framework and standardized dataset repository for benchmarking methods on irregular temporal data.

LinODEnet is narrower in scope than these libraries. Its emphasis is on reusable modeling components for continuous-time and irregular-time settings, especially when state space structure is central to the problem.
