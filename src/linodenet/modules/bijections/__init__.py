r"""Diffeomorphisms, i.e. differentiable bijections with differentiable inverse.

We call a module a bijector if it satisfies 3 properties:

1. It has both an `encode` and `decode` method.
2. It is invertible, i.e. `decode(encode(x)) = x` and `encode(decode(y)) = y`
3. Both encode and decode are differentiable.
"""
