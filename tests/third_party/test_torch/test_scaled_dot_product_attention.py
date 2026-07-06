r"""Tests documenting PyTorch scaled dot-product attention behavior."""

import torch
import torch.nn.functional as F
from torch.testing import assert_close


def test_scaled_dot_product_attention_fully_masked_boolean_row_returns_zero() -> None:
    r"""Show how SDPA handles a fully masked query row with a boolean mask.

    For the first query row, every key is masked out.

    - boolean mask: SDPA returns all zeros
    - float mask with the same fully masked row returns the mean of the value
      vectors instead
    """
    # Three queries, four keys, one head, and two features per tensor.
    # The first query will be fully masked, the second can only attend to key 0,
    # and the third can attend to keys 0 and 1.
    Q = torch.tensor(  # (1, 3, 2)
        [[[1.0, 0.0],
          [0.0, 1.0],
          [1.0, 1.0]]]
    )  # fmt: skip
    K = torch.tensor(  # (1, 4, 2)
        [[[2.0, 0.0],
          [0.0, 3.0],
          [1.0, 1.0],
          [2.0, 2.0]]]
    )  # fmt: skip

    # The values are chosen so their mean is easy to read off:
    # ([1, 0] + [0, 1] + [2, 0] + [0, 2]) / 4 = [0.75, 0.75].
    V = torch.tensor(  # (1, 4, 2)
        [[[1.0, 0.0],
          [0.0, 1.0],
          [2.0, 0.0],
          [0.0, 2.0]]]
    )  # fmt: skip

    # The first query row is fully masked. The other rows keep simple,
    # non-degenerate cases as a sanity check.
    bool_mask = torch.tensor(  # (1, 3, 4)
        [[[False, False, False, False],
          [ True, False, False, False],
          [ True,  True, False, False]]]
    )  # fmt: skip
    bool_output = F.scaled_dot_product_attention(Q, K, V, attn_mask=bool_mask)

    # In the fully masked row, SDPA with a boolean mask returns zeros.
    # The partially masked rows behave as ordinary attention.
    assert_close(
        bool_output,
        torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.3302, 0.6698]]]),
        atol=1e-4,
        rtol=1e-4,
    )

    # This mirrors the legacy GraFITi mask style: valid entries get 0 and
    # masked entries get a large finite negative bias.
    float_mask = Q.new_zeros(bool_mask.shape).masked_fill(~bool_mask, -10e9)
    float_output = F.scaled_dot_product_attention(Q, K, V, attn_mask=float_mask)
    # With a finite additive mask, the fully masked row becomes a uniform
    # average over all value vectors, so we get [0.75, 0.75] instead.
    assert_close(
        float_output,
        torch.tensor([[[0.75, 0.75], [1.0, 0.0], [0.3302, 0.6698]]]),
        atol=1e-4,
        rtol=1e-4,
    )
