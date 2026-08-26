import pytest
import torch
from torch import Tensor


@pytest.mark.parametrize("method", ["on_gpu", "from_cpu", "from_cache"])
@pytest.mark.benchmark(warmup=True, disable_gc=True)
def test_benchmark_initialization(benchmark, method: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("skip test on CPU")

    GPU = torch.device("cuda:0")
    CPU = torch.device("cpu")
    TARGET_DTYPE = torch.float64

    _P1 = torch.tensor(
        [
            4.05544892305962419923,
            3.15251094599893866154e1,
            5.71628192246421288162e1,
            4.40805073893200834700e1,
            1.46849561928858024014e1,
            2.18663306850790267539,
            -1.40256079171354495875e-1,
            -3.50424626827848203418e-2,
            -8.57456785154685413611e-4,
        ],
        dtype=torch.float64,
        device=CPU,
        pin_memory=True,
    )
    _Q1 = torch.tensor(
        [
            1.57799883256466749731e1,
            4.53907635128879210584e1,
            4.13172038254672030440e1,
            1.50425385692907503408e1,
            2.50464946208309415979,
            -1.42182922854787788574e-1,
            -3.80806407691578277194e-2,
            -9.33259480895457427372e-4,
        ],
        dtype=torch.float64,
        device=CPU,
        pin_memory=True,
    )
    _P2 = torch.tensor(
        [
            3.23774891776946035970,
            6.91522889068984211695,
            3.93881025292474443415,
            1.33303460815807542389,
            2.01485389549179081538e-1,
            1.23716634817820021358e-2,
            3.01581553508235416007e-4,
            2.65806974686737550832e-6,
            6.23974539184983293730e-9,
        ],
        dtype=torch.float64,
        device=CPU,
        pin_memory=True,
    )
    _Q2 = torch.tensor(
        [
            6.02427039364742014255,
            3.67983563856160859403,
            1.37702099489081330271,
            2.16236993594496635890e-1,
            1.34204006088543189037e-2,
            3.28014464682127739104e-4,
            2.89247864745380683936e-6,
            6.79019408009981274425e-9,
        ],
        dtype=torch.float64,
        device=CPU,
        pin_memory=True,
    )

    _COEFF_CACHE: dict[
        tuple[torch.device, torch.dtype],
        tuple[Tensor, Tensor, Tensor, Tensor],
    ] = {}

    def _get_coeffs(
        device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        key = (device, dtype)
        coeffs = _COEFF_CACHE.get(key)
        if coeffs is None:
            coeffs = (
                _P1.detach().clone().to(device=device, dtype=dtype),
                _Q1.detach().clone().to(device=device, dtype=dtype),
                _P2.detach().clone().to(device=device, dtype=dtype),
                _Q2.detach().clone().to(device=device, dtype=dtype),
            )
            _COEFF_CACHE[key] = coeffs
        return coeffs

    def initialize_on_gpu() -> None:
        _p1 = torch.tensor(
            [
                4.05544892305962419923,
                3.15251094599893866154e1,
                5.71628192246421288162e1,
                4.40805073893200834700e1,
                1.46849561928858024014e1,
                2.18663306850790267539,
                -1.40256079171354495875e-1,
                -3.50424626827848203418e-2,
                -8.57456785154685413611e-4,
            ],
            dtype=TARGET_DTYPE,
            device=GPU,
        )
        _q1 = torch.tensor(
            [
                1.57799883256466749731e1,
                4.53907635128879210584e1,
                4.13172038254672030440e1,
                1.50425385692907503408e1,
                2.50464946208309415979,
                -1.42182922854787788574e-1,
                -3.80806407691578277194e-2,
                -9.33259480895457427372e-4,
            ],
            dtype=TARGET_DTYPE,
            device=GPU,
        )
        _p2 = torch.tensor(
            [
                3.23774891776946035970,
                6.91522889068984211695,
                3.93881025292474443415,
                1.33303460815807542389,
                2.01485389549179081538e-1,
                1.23716634817820021358e-2,
                3.01581553508235416007e-4,
                2.65806974686737550832e-6,
                6.23974539184983293730e-9,
            ],
            dtype=TARGET_DTYPE,
            device=GPU,
        )
        _q2 = torch.tensor(
            [
                6.02427039364742014255,
                3.67983563856160859403,
                1.37702099489081330271,
                2.16236993594496635890e-1,
                1.34204006088543189037e-2,
                3.28014464682127739104e-4,
                2.89247864745380683936e-6,
                6.79019408009981274425e-9,
            ],
            dtype=TARGET_DTYPE,
            device=GPU,
        )
        torch.cuda.synchronize()

    def initialize_from_cpu() -> None:
        _p1 = _P1.to(dtype=TARGET_DTYPE, device=GPU)
        _p2 = _P2.to(dtype=TARGET_DTYPE, device=GPU)
        _q1 = _Q1.to(dtype=TARGET_DTYPE, device=GPU)
        _q2 = _Q2.to(dtype=TARGET_DTYPE, device=GPU)
        torch.cuda.synchronize()

    def initialize_from_cache() -> None:
        _p1, _q1, _p2, _q2 = _get_coeffs(device=GPU, dtype=TARGET_DTYPE)
        torch.cuda.synchronize()

    match method:
        case "on_gpu":
            meth = initialize_on_gpu
        case "from_cpu":
            meth = initialize_from_cpu
        case "from_cache":
            meth = initialize_from_cache
        case _:
            raise NotImplementedError

    benchmark.pedantic(meth, rounds=250)
