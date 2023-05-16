"""Implementation of invertible layers.

Layers:
- Affine: $y = Ax+b$ and $x = A⁻¹(y-b)$
    - A diagonal
    - A triangular
    - A tridiagonal
- Element-wise:
    - Monotonic
- Shears (coupling flows): $y_A = f(x_A, x_B)$ and $y_B = x_B$
    - Example: $y_A = x_A + e^{x_B}$ and $y_B = x_B$
- Residual: $y = x + F(x)$
    - Contractive: $y = F(x)$ with $‖F‖<1$
    - Low Rank Perturbation: $y = x + ABx$
- Continuous Time Flows: $ẋ=f(t, x)$
"""
