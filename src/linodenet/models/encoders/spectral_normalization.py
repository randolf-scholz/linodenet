"""Re-Implementation of Spectral Normalization Layer.

Notes
-----
The default implementation provided by torch is problematic [1]_:

- It only performs a single power iteration, without any convergence test.
- It internally uses pseudo-inverse, which causes a full SVD to be computed.
    - This SVD can fail for ill-conditioned matrices.
- It does not implement any optimized backward pass.

Alternatively, torch.linalg.matrix_norm(A, ord=2) can be used to compute the spectral norm of a matrix A.
But here, again, torch computes the full SVD.

Our implementation addresses these issues:

- We use the analytic formula for the gradient: $∂‖A‖₂/∂A = uvᵀ$,
  where $u$ and $v$ are the left and right singular vectors of $A$.
- We use the power iteration with convergence test.


References
----------
.. [1] https://pytorch.org/docs/stable/_modules/torch/nn/utils/spectral_norm.html#spectral_norm
"""
