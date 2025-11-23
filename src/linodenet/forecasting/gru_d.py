r"""Implementation of GRU-D model for time series forecasting.

Reference:
    - Recurrent Neural Networks for Multivariate Time Series with Missing Values
    Zhengping Che, Sanjay Purushotham, Kyunghyun Cho, David Sontag & Yan Liu
    Nature Scientific Reports
    https://www.nature.com/articles/s41598-018-24271-9
"""

__all__ = ["GRU_D"]

from torch import nn


class GRU_D(nn.Module):
    r"""TODO: Implement GRU-D model for time series forecasting."""
