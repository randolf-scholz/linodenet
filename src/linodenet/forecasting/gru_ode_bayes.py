r"""Implementation of GRU-ODE-Bayes model for time series forecasting.

References:
    - GRU-ODE-Bayes: Continuous modeling of sporadically-observed time series
    Edward De Brouwer, Jaak Simm, Adam Arany, Yves Moreau
    33rd Conference on Neural Information Processing Systems (NeurIPS 2019)
    https://proceedings.neurips.cc/paper/2019/hash/455cb2657aaa59e32fad80cb0b65b9dc-Abstract.html
    - https://github.com/edebrouwer/gru_ode_bayes
"""

__all__ = [
    "FullGRUODECell",
    "FullGRUODECell_Autonomous",
    "GRUODECell",
    "GRUODECell_Autonomous",
    "GRUObservationCell",
    "GRUObservationCellLogvar",
    "GRU_ODE_Bayes",
    # Functions
    "compute_kl_loss",
    "gaussian_kl",
]

import math
from collections.abc import Callable
from typing import Final

import numpy as np
import torch
from torch import Tensor, nn
from torchdiffeq import odeint


class GRUODECell(nn.Module):
    r"""Implements one step of GRU-ODE."""

    input_size: Final[int]
    r"""CONST: The dimensionality of inputs."""
    hidden_size: Final[int]
    r"""CONST: The dimensionality of hidden states."""
    bias: Final[bool]
    r"""CONST: Whether to use bias terms in the linear layers."""

    def __init__(self, input_size: int, hidden_size: int, *, bias: bool = True):
        r"""For p(t) modelling input_size should be 2x the x size."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.lin_xz = nn.Linear(input_size, hidden_size, bias=bias)
        self.lin_xn = nn.Linear(input_size, hidden_size, bias=bias)

        self.lin_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hn = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        r"""Returns a change due to one step of using GRU-ODE for all h.

        The step size is given by delta_t.

        Args:
            x:        input values
            h:        hidden state (current)
            delta_t:  time step

        Returns:
            Updated h
        """
        z = torch.sigmoid(self.lin_xz(x) + self.lin_hz(h))
        n = torch.tanh(self.lin_xn(x) + self.lin_hn(z * h))

        dh = (1 - z) * (n - h)
        return dh


class GRUODECell_Autonomous(nn.Module):
    r"""Implements one step of autonomous GRU-ODE."""

    hidden_size: Final[int]
    r"""CONST: The dimensionality of hidden states."""
    bias: Final[bool]
    r"""CONST: Whether to use bias terms in the linear layers."""

    def __init__(self, hidden_size: int, *, bias: bool = True) -> None:
        r"""For p(t) modelling input_size should be 2x the x size."""
        super().__init__()
        self.hidden_size = hidden_size
        self.bias = bias

        self.lin_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hn = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, t: Tensor, h: Tensor) -> Tensor:  # noqa: ARG002
        """Returns a change due to one step of using GRU-ODE for all h.

        The step size is given by delta_t.

        Args:
            t: time
            h: hidden state (current)

        Returns:
            Updated h
        """
        x = torch.zeros_like(h)
        z = torch.sigmoid(x + self.lin_hz(h))
        n = torch.tanh(x + self.lin_hn(z * h))

        dh = (1 - z) * (n - h)
        return dh


class FullGRUODECell(nn.Module):
    r"""Implements one step of Full GRU-ODE."""

    input_size: Final[int]
    r"""CONST: The dimensionality of inputs."""
    hidden_size: Final[int]
    r"""CONST: The dimensionality of hidden states."""
    bias: Final[bool]
    r"""CONST: Whether to use bias terms in the linear layers."""

    def __init__(self, input_size: int, hidden_size: int, *, bias: bool = True) -> None:
        r"""For p(t) modelling input_size should be 2x the x size."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.lin_x = nn.Linear(input_size, hidden_size * 3, bias=bias)

        self.lin_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hr = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        r"""Executes one step with GRU-ODE for all h.

        The step size is given by delta_t.

        Args:
            x:        input values
            h:        hidden state (current)
            delta_t:  time step

        Returns:
            Updated h
        """
        xr, xz, xh = torch.chunk(self.lin_x(x), 3, dim=1)
        r = torch.sigmoid(xr + self.lin_hr(h))
        z = torch.sigmoid(xz + self.lin_hz(h))
        u = torch.tanh(xh + self.lin_hh(r * h))

        dh = (1 - z) * (u - h)
        return dh


class FullGRUODECell_Autonomous(nn.Module):
    r"""Implements one step of autonomous Full GRU-ODE."""

    hidden_size: Final[int]
    r"""CONST: The dimensionality of hidden states."""
    bias: Final[bool]
    r"""CONST: Whether to use bias terms in the linear layers."""

    def __init__(self, hidden_size: int, *, bias: bool = True) -> None:
        r"""For p(t) modelling input_size should be 2x the x size."""
        super().__init__()
        self.hidden_size = hidden_size
        self.bias = bias

        self.lin_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hr = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, t: Tensor, h: Tensor) -> Tensor:  # noqa: ARG002
        """Executes one step with autonomous GRU-ODE for all h.

        The step size is given by delta_t.

        Args:
            t: time of evaluation
            h: hidden state (current)

        Returns:
            Updated h
        """
        x = torch.zeros_like(h)
        r = torch.sigmoid(x + self.lin_hr(h))
        z = torch.sigmoid(x + self.lin_hz(h))
        u = torch.tanh(x + self.lin_hh(r * h))

        dh = (1 - z) * (u - h)
        return dh


class GRUObservationCellLogvar(nn.Module):
    r"""Implements discrete update based on the received observations."""

    input_size: Final[int]
    r"""CONST: The dimensionality of inputs."""
    hidden_size: Final[int]
    r"""CONST: The dimensionality of hidden states."""
    prep_hidden: Final[int]
    r"""CONST: The dimensionality of the prep layer hidden states."""
    bias: Final[bool]
    r"""CONST: Whether to use bias terms in the linear layers."""

    def __init__(
        self, input_size: int, hidden_size: int, prep_hidden: int, *, bias: bool = True
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.prep_hidden = prep_hidden
        self.bias = bias

        self.gru_d = nn.GRUCell(prep_hidden * input_size, hidden_size, bias=bias)
        self.gru_debug = nn.GRUCell(prep_hidden * input_size, hidden_size, bias=bias)

        ## prep layer and its initialization
        std = math.sqrt(2.0 / (4 + prep_hidden))
        self.w_prep = nn.Parameter(std * torch.randn(input_size, 4, prep_hidden))
        self.bias_prep = nn.Parameter(0.1 + torch.zeros(input_size, prep_hidden))

    def forward(
        self, h: Tensor, p: Tensor, X_obs: Tensor, M_obs: Tensor, i_obs: Tensor
    ) -> tuple[Tensor, Tensor]:
        ## only updating rows that have observations
        p_obs = p[i_obs]

        mean, logvar = torch.chunk(p_obs, 2, dim=1)
        sigma = torch.exp(0.5 * logvar)
        error = (X_obs - mean) / sigma

        ## log normal loss, over all observations
        log_lik_c = np.log(np.sqrt(2 * np.pi))
        losses = 0.5 * ((torch.pow(error, 2) + logvar + 2 * log_lik_c) * M_obs)
        if losses.sum() != losses.sum():
            raise AssertionError

        ## TODO: try removing X_obs (they are included in error)
        gru_input = torch.stack([X_obs, mean, logvar, error], dim=2).unsqueeze(2)
        gru_input = torch.matmul(gru_input, self.w_prep).squeeze(2) + self.bias_prep
        gru_input.relu_()
        ## gru_input is (sample x feature x prep_hidden)
        gru_input = gru_input.permute(2, 0, 1)
        gru_input = (
            (gru_input * M_obs)
            .permute(1, 2, 0)
            .contiguous()
            .view(-1, self.prep_hidden * self.input_size)
        )

        temp = h.clone()
        temp[i_obs] = self.gru_d(gru_input, h[i_obs])
        h = temp

        return h, losses


class GRUObservationCell(nn.Module):
    r"""Implements discrete update based on the received observations."""

    input_size: Final[int]
    r"""CONST: The dimensionality of inputs."""
    hidden_size: Final[int]
    r"""CONST: The dimensionality of hidden states."""
    prep_hidden: Final[int]
    r"""CONST: The dimensionality of the prep layer hidden states."""
    bias: Final[bool]
    r"""CONST: Whether to use bias terms in the linear layers."""
    var_eps: Final[float]
    r"""CONST: Small value added to variance to avoid numerical issues."""

    def __init__(
        self, input_size: int, hidden_size: int, prep_hidden: int, *, bias: bool = True
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.prep_hidden = prep_hidden
        self.var_eps = 1e-6

        self.gru_d = nn.GRUCell(prep_hidden * input_size, hidden_size, bias=bias)
        self.gru_debug = nn.GRUCell(prep_hidden * input_size, hidden_size, bias=bias)

        ## prep layer and its initialization
        std = math.sqrt(2.0 / (4 + prep_hidden))
        self.w_prep = nn.Parameter(std * torch.randn(input_size, 4, prep_hidden))
        self.bias_prep = nn.Parameter(0.1 + torch.zeros(input_size, prep_hidden))

    def forward(
        self, h: Tensor, p: Tensor, X_obs: Tensor, M_obs: Tensor, i_obs: Tensor
    ) -> tuple[Tensor, Tensor]:
        ## only updating rows that have observations
        p_obs = p[i_obs]
        mean, var = torch.chunk(p_obs, 2, dim=1)
        ## making var non-negative and also non-zero (by adding a small value)
        var = torch.abs(var) + self.var_eps
        error = (X_obs - mean) / torch.sqrt(var)

        ## log normal loss, over all observations
        loss = 0.5 * ((torch.pow(error, 2) + torch.log(var)) * M_obs).sum()

        ## TODO: try removing X_obs (they are included in error)
        gru_input = torch.stack([X_obs, mean, var, error], dim=2).unsqueeze(2)
        gru_input = torch.matmul(gru_input, self.w_prep).squeeze(2) + self.bias_prep
        gru_input.relu_()
        ## gru_input is (sample x feature x prep_hidden)
        gru_input = gru_input.permute(2, 0, 1)
        gru_input = (
            (gru_input * M_obs)
            .permute(1, 2, 0)
            .contiguous()
            .view(-1, self.prep_hidden * self.input_size)
        )

        temp = h.clone()
        temp[i_obs] = self.gru_d(gru_input, h[i_obs])
        h = temp

        return h, loss


class GRU_ODE_Bayes(nn.Module):
    r"""Implements the GRU-ODE-Bayes model for time series forecasting."""

    solver: Final[str]
    r"""CONST: The ODE solver to use ('euler', 'midpoint', 'dopri5')."""
    mixing: Final[float]
    r"""CONST: The mixing hyperparameter for loss aggregation."""
    impute: Final[bool]
    r"""CONST: Whether to impute observations into the ODE."""
    input_size: Final[int]
    r"""CONST: The dimensionality of inputs."""
    hidden_size: Final[int]
    r"""CONST: The dimensionality of hidden states."""
    store_hist: Final[bool]
    r"""CONST: Whether to store the history of evaluations for dopri5 solver."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p_hidden: int,
        *,
        prep_hidden: int,
        bias: bool = True,
        cov_size: int = 1,
        cov_hidden: int = 1,
        classification_hidden: int = 1,
        logvar: bool = True,
        mixing: float = 1,
        dropout_rate: float = 0,
        full_gru_ode: bool = False,
        solver: str = "euler",
        impute: bool = True,
        store_hist: bool = False,
    ) -> None:
        r"""Initializes the GRU-ODE-Bayes model.

        The smoother variable computes the classification loss as a weighted average
        of the projection of the latents at each observation.

        Impute feeds the parameters of the distribution to GRU-ODE at each step.
        """
        super().__init__()
        self.impute = impute
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.logvar = logvar
        self.mixing = mixing  # mixing hyperparameter for loss_1 and loss_2 aggregation.
        self.solver = solver
        self.store_hist = store_hist

        self.p_model = nn.Sequential(
            nn.Linear(hidden_size, p_hidden, bias=bias),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(p_hidden, 2 * input_size, bias=bias),
        )

        self.classification_model = nn.Sequential(
            nn.Linear(hidden_size, classification_hidden, bias=bias),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(classification_hidden, 1, bias=bias),
        )

        self.gru_c: Callable[[Tensor, Tensor], Tensor]
        match full_gru_ode, self.impute:
            case True, True:
                self.gru_c = FullGRUODECell(2 * input_size, hidden_size, bias=bias)
            case True, False:
                self.gru_c = FullGRUODECell_Autonomous(hidden_size, bias=bias)
            case False, True:
                self.gru_c = GRUODECell(2 * input_size, hidden_size, bias=bias)
            case False, False:
                self.gru_c = GRUODECell_Autonomous(hidden_size, bias=bias)
            case _:
                raise ValueError

        self.gru_obs = (
            GRUObservationCellLogvar(input_size, hidden_size, prep_hidden, bias=bias)
            if logvar
            else GRUObservationCell(input_size, hidden_size, prep_hidden, bias=bias)
        )

        self.covariates_map = nn.Sequential(
            nn.Linear(cov_size, cov_hidden, bias=bias),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(cov_hidden, hidden_size, bias=bias),
            nn.Tanh(),
        )

        assert solver in ["euler", "midpoint", "dopri5"], (
            "Solver must be either 'euler' or 'midpoint' or 'dopri5'."
        )

        def init_weights(m: nn.Module) -> None:
            if type(m) is nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.05)

        self.apply(init_weights)

    def ode_step(
        self,
        h: Tensor,
        p: Tensor,
        delta_t: float | Tensor,
        current_time: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""Executes a single ODE step."""
        eval_times = torch.tensor([0], device=h.device, dtype=torch.float64)
        eval_ps = torch.tensor([0], device=h.device, dtype=torch.float32)

        if not self.impute:
            p = torch.zeros_like(p)

        if self.solver == "euler":
            h = h + delta_t * self.gru_c(p, h)
            p = self.p_model(h)

        elif self.solver == "midpoint":
            k = h + delta_t / 2 * self.gru_c(p, h)
            pk = self.p_model(k)

            h = h + delta_t * self.gru_c(pk, k)
            p = self.p_model(h)

        elif self.solver == "dopri5":
            assert not self.impute, (
                "Dopri5 solver is only compatible with autonomous ODE."
            )
            solution, eval_times, eval_vals = odeint(
                self.gru_c,
                h,
                torch.tensor([0, delta_t]),
                method=self.solver,
                options={"store_hist": self.store_hist},
            )
            if self.store_hist:
                eval_ps = self.p_model(torch.stack([ev[0] for ev in eval_vals]))
            eval_times = torch.stack(eval_times) + current_time
            h = solution[1, :, :]
            p = self.p_model(h)
        else:
            raise ValueError(f"Unknown solver {self.solver!r}.")

        current_time = current_time + delta_t
        return h, p, current_time, eval_times, eval_ps

    def forward(
        self,
        times: Tensor,
        time_ptr: Tensor,
        X: Tensor,
        M: Tensor,
        obs_idx: Tensor,
        delta_t: float,
        T: float,
        cov: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Executes the GRU-ODE-Bayes over the given time series.

        Args:
            times:      tensor observation times
            time_ptr:   start indices of data for a given time
            X:          data tensor
            M:          mask tensor (1.0 if observed, 0.0 if unobserved)
            obs_idx:    observed patients of each datapoint (indexed within the current minibatch)
            delta_t:    time step for Euler
            T:          total time
            cov:        static covariates for learning the first h0

        Returns:
            h:          hidden state at final time (T)
            loss:       loss of the Gaussian observations
        """
        h = self.covariates_map(cov)
        p = self.p_model(h)

        current_time: Tensor = torch.zeros(())

        loss_1: Tensor = torch.zeros(())  # Pre-jump loss
        loss_2: Tensor = (
            torch.zeros(())
        )  # Post-jump loss (KL between p_updated and the actual sample)

        path_t: list[Tensor] = [torch.zeros(())]
        path_p: list[Tensor] = [p]
        path_h: list[Tensor] = [h]

        assert len(times) + 1 == len(time_ptr)
        assert (len(times) == 0) or (times[-1] <= T)

        eval_times_total: Tensor = torch.tensor(
            [], dtype=torch.float64, device=h.device
        )
        eval_vals_total: Tensor = torch.tensor([], dtype=torch.float32, device=h.device)

        for i, obs_time in enumerate(times):
            ## Propagation of the ODE until next observation
            while current_time < (
                obs_time - 0.001 * delta_t
            ):  # 0.0001 delta_t used for numerical consistency.
                if self.solver == "dopri5":
                    h, p, current_time, eval_times, eval_ps = self.ode_step(
                        h, p, obs_time - current_time, current_time
                    )
                else:
                    h, p, current_time, eval_times, eval_ps = self.ode_step(
                        h, p, delta_t, current_time
                    )
                eval_times_total = torch.cat((eval_times_total, eval_times))
                eval_vals_total = torch.cat((eval_vals_total, eval_ps))

                # Storing the predictions.
                path_t.append(current_time)
                path_p.append(p)
                path_h.append(h)

            ## Reached an observation
            start = time_ptr[i]
            end = time_ptr[i + 1]

            X_obs = X[start:end]
            M_obs = M[start:end]
            i_obs = obs_idx[start:end]

            ## Using GRUObservationCell to update h. Also updating p and loss
            h, losses = self.gru_obs(h, p, X_obs, M_obs, i_obs)

            if not losses.sum().isfinite():
                raise AssertionError

            loss_1 = loss_1 + losses.sum()
            p = self.p_model(h)

            loss_2 = loss_2 + compute_kl_loss(
                p_obs=p[i_obs],
                X_obs=X_obs,
                M_obs=M_obs,
                logvar=self.logvar,
            )

            path_t.append(obs_time)
            path_p.append(p)
            path_h.append(h)

        ## after every observation has been processed, propagating until T
        while current_time < T:
            if self.solver == "dopri5":
                h, p, current_time, eval_times, eval_ps = self.ode_step(
                    h, p, T - current_time, current_time
                )
            else:
                h, p, current_time, eval_times, eval_ps = self.ode_step(
                    h, p, delta_t, current_time
                )
            eval_times_total = torch.cat((eval_times_total, eval_times))
            eval_vals_total = torch.cat((eval_vals_total, eval_ps))

            # Storing the predictions
            path_t.append(current_time)
            path_p.append(p)
            path_h.append(h)

        loss = loss_1 + self.mixing * loss_2

        class_pred = self.classification_model(h)

        torch.stack(path_t)
        torch.stack(path_p)
        torch.stack(path_h)

        return h, loss, class_pred, loss_1


def compute_kl_loss(
    p_obs: Tensor,
    X_obs: Tensor,
    M_obs: Tensor,
    obs_noise_std: float | Tensor = 1e-2,
    logvar: bool = True,
) -> Tensor:
    noise = torch.as_tensor(obs_noise_std)

    if logvar:
        mean, var = torch.chunk(p_obs, 2, dim=1)
        std = torch.exp(0.5 * var)
    else:
        mean, var = torch.chunk(p_obs, 2, dim=1)
        ## making var non-negative and also non-zero (by adding a small value)
        std = torch.pow(torch.abs(var) + 1e-5, 0.5)

    return (gaussian_kl(left=(mean, std), right=(X_obs, noise)) * M_obs).sum()


def gaussian_kl(left: tuple[Tensor, Tensor], right: tuple[Tensor, Tensor]) -> Tensor:
    mu_1, sigma_1 = left
    mu_2, sigma_2 = right
    return (
        torch.log(sigma_2)
        - torch.log(sigma_1)
        + (torch.pow(sigma_1, 2) + torch.pow((mu_1 - mu_2), 2)) / (2 * sigma_2**2)
        - 0.5
    )
