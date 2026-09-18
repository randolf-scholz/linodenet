# Motivation

The parametrization utility provided by torch [`torch.nn.utils.parametrize.register_parametrization`](https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrize.register_parametrization.html) has several crucial shortcomings.

1. Incompatibility with `torch.jit.script` and `torch.jit.trace` (see [`issue #60971`](https://github.com/pytorch/pytorch/issues/60971)
2. Context manager `with parametrize.cache()` allows to little control over what is cached.
   - A user might want to cache only a subset of the parametrizations.
   - Instead, we should be able to collect parametrizations we want to cache similar to how optimizers work. (e.g. `with parametrize.cache(model.parametrizations())))`)
3. Recursion problem: Imagine using a model with a parametrized submodule, which we want to cache.
   For example, because this submodule is called often in the forward pass.
   One could consider adding code that caches the values in the forward pass of the outer model.
   Now, imagine the model being added as a submodule to another model. Now, the caching is suboptimal, since the outer model will not cache the values of the inner model.
   Therefore, caching should generally happen at the outermost level, which is not detectable from the inner model.
4. Parametrizations currently provide no ready made schema creating classes that efficiently perform parametrizations. In particular, certain parametrizations like spectral normalization greatly benefit from reusing the singular vectors from previous computations as initial guesses for the next computation.

## Solution

1. We add a utility function to recursively collect all parametrizations of a model.
2. We provide a plug-in replacement `register_parametrization`
3. We provide a plug-in replacement context manager `cache()` to works with JIT and allows to cache only a subset of the parametrizations.
4. Wer provide a `Protocol`-class `ParametrizationProto` and a base class `Parametrization` for implementing parametrizations.
5. We rigorously test the validity of the code against non-parametrized variants both for correctness and performance.
