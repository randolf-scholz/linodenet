r"""Latent State Space Model."""

__all__ = [
    # Classes
    "LSSM",
    "EncoderDecoderLSSM",
]

from collections.abc import Callable
from typing import Final

from torch import Tensor, nn


class LSSM[State](nn.Module):
    r"""Decoder-Only Latent State Space Model."""

    batch_first: Final[bool]
    propagate_state: Callable[[State, Tensor, Tensor], State]
    update_state: Callable[[State, Tensor], State]
    initial_state: State

    def __init__(self, batch_first: bool = True) -> None:
        super().__init__()
        self.batch_first = batch_first

    def forward(
        self,
        *,
        context_values: Tensor,  # (..., $N, D)
        timestamps: Tensor,  # (..., $N)
        initial_state: State | None = None,
        initial_time: Tensor | None = None,
    ) -> list[State]:

        seq_dim = -2 if self.batch_first else -1
        T = timestamps[..., None].movedim(seq_dim, 0).squeeze(-1)  # ($N, ...)
        X = context_values.movedim(seq_dim, 0)  # ($N, ..., D)

        posterior_state: State = (
            initial_state if initial_state is not None else self.initial_state
        )
        t_prev = initial_time if initial_time is not None else T[0]

        prior_states: list[State] = []
        posterior_states: list[State] = []

        for t, x_obs in zip(T, X, strict=True):
            prior_state = self.propagate_state(posterior_state, t_prev, t)
            posterior_state = self.update_state(prior_state, x_obs)

            t_prev = t

            prior_states.append(prior_state)
            posterior_states.append(posterior_state)

        return posterior_states


class EncoderDecoderLSSM[State](nn.Module):
    r"""Latent State Space Model with Encoder-Decoder architecture.

    Contrary to a regular LSSM, this model applies the filter in data-space,
    and obtains the updated latent state from the encoder.

        **Model Sketch**::

            ⟶ [ODE] ⟶ (ẑᵢ)                (ẑᵢ') ⟶ [ODE] ⟶
                       ↓                   ↑
                      [Ψ]                 [Φ]
                       ↓                   ↑
                      (x̂ᵢ) → [ filter ] → (x̂ᵢ')
                                 ↑
                              (tᵢ, xᵢ)

    +---------------------------------------------------+--------------------------------------+
    | Component                                         | Formula                              |
    +===================================================+======================================+
    | Filter  `F` (default: :class:`~torch.nn.GRUCell`) | `\hat x_i' = F(\hat x_i, x_i)`       |
    +---------------------------------------------------+--------------------------------------+
    | Encoder `ϕ` (default: :class:`~iResNet`)          | `\hat z_i' = ϕ(\hat x_i')`           |
    +---------------------------------------------------+--------------------------------------+
    | System  `S` (default: :class:`~LinODECell`)       | `\hat z_{i+1} = S(\hat z_i', Δ t_i)` |
    +---------------------------------------------------+--------------------------------------+
    | Decoder `π` (default: :class:`~iResNet`)          | `\hat x_{i+1}  =  π(\hat z_{i+1})`   |
    +---------------------------------------------------+--------------------------------------+
    """

    batch_first: Final[bool]
    propagate_state: Callable[[State, Tensor, Tensor], State]
    decoder: Callable[[State], Tensor]
    encoder: Callable[[Tensor], State]
    update_prediction: Callable[[Tensor, Tensor], Tensor]

    initial_state: State

    def __init__(self, batch_first: bool = True) -> None:
        super().__init__()
        self.batch_first = batch_first

    def forward(
        self,
        *,
        context_values: Tensor,  # (..., $N, D)
        timestamps: Tensor,  # (..., $N)
        initial_state: State | None = None,
        initial_time: Tensor | None = None,
    ) -> list[State]:

        seq_dim = -2 if self.batch_first else -1
        T = timestamps[..., None].movedim(seq_dim, 0).squeeze(-1)  # ($N, ...)
        X = context_values.movedim(seq_dim, 0)  # ($N, ..., D)

        posterior_state = (
            initial_state if initial_state is not None else self.initial_state
        )
        t_prev = initial_time if initial_time is not None else T[0]

        prior_states: list[State] = []
        posterior_states: list[State] = []

        for t, x_obs in zip(T, X, strict=True):
            prior_state = self.propagate_state(posterior_state, t_prev, t)

            prior_prediction = self.decoder(prior_state)

            posterior_prediction = self.update_prediction(prior_prediction, x_obs)

            posterior_state = self.encoder(posterior_prediction)

            t_prev = t

            prior_states.append(prior_state)
            posterior_states.append(posterior_state)

        return posterior_states
