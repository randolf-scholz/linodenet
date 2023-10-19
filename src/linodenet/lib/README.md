# Some general purpose algorithm schemas

## Fixpoint iteration schema

```python
def fixpoint_iteration_factor(func, cond, encode, decode):
    """General purpose fixpoint iteration schema."""

    def fixpoint_iteration(inputs):
        # initialize state
        state = encode(inputs)

        # perform fixed point iteration
        while True:
            # update state
            state = func(state)
            # check convergence
            if cond(state): break

        # return decoded state
        return decode(state)

    # return function
    return fixpoint_iteration
```

### Noteable examples

- Gradient Descent schemes

## General purpose divide and conquer schema

```python
def divide_and_conquer_factory(cond, conquer_fn, divide_fn, combine_fn, encode, decode):
    """General purpose divide and conquer schema."""

    # define the inner divide and conquer recursive function
    def dac_inner(partition):
        # check base case
        if cond(partition):
            return conquer_fn(partition)

        # perform divide and conquer
        partitions = divide_fn(partition)

        # parallel map
        results = map(dac_inner, partitions)

        # return recombined results
        return combine_fn(results)

    # define divide and conquer function
    def divide_and_conquer(inputs):
        # initialize state
        state = encode(inputs)

        # perform divide and conquer
        result = dac_inner(state)

        # return decoded state
        return decode(result)

    return divide_and_conquer
```

### Notable examples

- merge sort
