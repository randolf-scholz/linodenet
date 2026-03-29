import torch


def test_masked_grad() -> None:
    N = 16
    torch.manual_seed(0)

    # create some observations with missing values
    y = torch.randn(N)
    y = torch.where(y > 0.5, y, torch.nan)  # masked target
    x = torch.randn(N, requires_grad=True)

    # variant 1: first where, then square
    # produces NaN-free gradient
    r = x - y
    m = torch.isnan(y)
    r = torch.where(m, 0.0, r) ** 2
    loss = r.sum() / m.sum()
    loss.backward()
    assert x.grad is not None
    assert not x.grad.isnan().any()

    # variant 2: first square, then where
    # produces NaN-gradients
    x.grad = None
    r = (x - y) ** 2
    m = torch.isnan(y)
    r = torch.where(m, 0.0, r)
    loss = r.sum() / m.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.isnan().any()

    # variant 3: first torch.cond, then square
    # produces NaN-free gradient
    x.grad = None

    def masked_residual(
        xi: torch.Tensor, yi: torch.Tensor, mi: torch.Tensor
    ) -> torch.Tensor:
        return torch.cond(
            mi,
            lambda xi, _: torch.zeros_like(xi),
            lambda xi, yi: xi - yi,
            (xi, yi),
        )

    r = torch.vmap(masked_residual)(x, y, m) ** 2
    loss = r.sum() / m.sum()
    loss.backward()
    assert x.grad is not None
    assert not x.grad.isnan().any()

    # variant 4: first square inside torch.cond
    # still produces NaN-gradients
    x.grad = None

    def masked_squared_residual(
        xi: torch.Tensor, yi: torch.Tensor, mi: torch.Tensor
    ) -> torch.Tensor:
        return torch.cond(
            mi,
            lambda xi, _: torch.zeros_like(xi),
            lambda xi, yi: (xi - yi) ** 2,
            (xi, yi),
        )

    r = torch.vmap(masked_squared_residual)(x, y, m)
    loss = r.sum() / m.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.isnan().any()
