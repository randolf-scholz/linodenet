import torch


def test_masked_grad_where() -> None:
    N = 16
    torch.manual_seed(0)

    # create some observations with missing values
    y = torch.randn(N)
    m = y > 0.5
    y = torch.where(m, torch.nan, y)  # masked target
    x = torch.randn(N, requires_grad=True)

    # variant 1: first where, then square
    # produces NaN-free gradient
    r = x - y
    r = torch.where(m, 0.0, r) ** 2
    loss = r.sum() / m.sum()
    loss.backward()
    assert x.grad is not None
    assert not x.grad.isnan().any()

    # variant 2: first square, then where
    # produces NaN-gradients
    x.grad = None
    r = (x - y) ** 2
    r = torch.where(m, 0.0, r)
    loss = r.sum() / m.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.isnan().any()

    # variant 3: sanitize the inactive branch input first, then square
    # produces NaN-free gradient
    x.grad = None
    y_safe = torch.where(m, 0.0, y)
    r = (x - y_safe) ** 2
    r = torch.where(m, 0.0, r)
    loss = r.sum() / m.sum()
    loss.backward()
    assert x.grad is not None
    assert not x.grad.isnan().any()


def test_masked_grad_torch_cond() -> None:
    N = 16
    torch.manual_seed(0)

    # create some observations with missing values
    y = torch.randn(N)
    m = y > 0.5
    y = torch.where(m, torch.nan, y)  # masked target
    x = torch.randn(N, requires_grad=True)

    # variant 1: first torch.cond, then square
    # produces NaN-free gradient
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

    # variant 2: first square inside torch.cond
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

    # variant 3: sanitize the inactive branch input first, then square inside torch.cond
    # produces NaN-free gradient
    x.grad = None
    y_safe = torch.where(m, 0.0, y)

    def masked_squared_residual_safe(
        xi: torch.Tensor, yi: torch.Tensor, mi: torch.Tensor
    ) -> torch.Tensor:
        return torch.cond(
            mi,
            lambda xi, _: torch.zeros_like(xi),
            lambda xi, yi: (xi - yi) ** 2,
            (xi, yi),
        )

    r = torch.vmap(masked_squared_residual_safe)(x, y_safe, m)
    loss = r.sum() / m.sum()
    loss.backward()
    assert x.grad is not None
    assert not x.grad.isnan().any()


def test_masked_grad_gather() -> None:
    N = 16
    torch.manual_seed(0)

    # create some observations with missing values
    y = torch.randn(N)
    m = y > 0.5
    y = torch.where(m, torch.nan, y)  # masked target
    x = torch.randn(N, requires_grad=True)
    selector = (~m).to(torch.int64).unsqueeze(0)

    # variant 1: first gather, then square
    # produces NaN-free gradient
    r = x - y
    choices = torch.stack([torch.zeros_like(r), r], dim=0)
    r = choices.gather(0, selector).squeeze(0)
    r = r**2
    loss = r.sum() / m.sum()
    loss.backward()
    assert x.grad is not None
    assert not x.grad.isnan().any()

    # variant 2: first square, then gather
    # produces NaN-gradients because the NaN-producing op already happened
    x.grad = None
    r = (x - y) ** 2
    choices = torch.stack([torch.zeros_like(r), r], dim=0)
    r = choices.gather(0, selector).squeeze(0)
    loss = r.sum() / m.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.isnan().any()

    # variant 3: sanitize the inactive branch input first, then square inside gather
    # produces NaN-free gradient
    x.grad = None
    y_safe = torch.where(m, 0.0, y)
    r = (x - y_safe) ** 2
    choices = torch.stack([torch.zeros_like(r), r], dim=0)
    r = choices.gather(0, selector).squeeze(0)
    loss = r.sum() / m.sum()
    loss.backward()
    assert x.grad is not None
    assert not x.grad.isnan().any()
