# Model Configuration

## Goals

1. We want to be able to configure models with hyperparameters.
2. Each class should provide a default configuration.
3. Support for static typing.
4. There should be a simple way to **export** a configuration from a module instance.

```python
def export_config(model_instance: "Module") -> "JSON_DICT": ...
```

5. There should be a simple way to **import** a configuration into a module instance.
    We do not require this to be a serialization of the full model, but rather to allow instantiation
    of a model with the same hyperparameters, which we can then use `load_state_dict` on.

## Questions.

1. How do we handle nested configurations?
   Example: Model with several subcomponents/layers.


2. How to handle dependent values?
   Example: Model with subcomponents, whose hyperparameters depend on each other.

   => Dependencies should form a Directed Acyclic Graph (DAG).
   => can be solved with topological sorting / lazy evaluation.

## Config Class

We want static typing, so nicer than a plain dictionary would be a class with
attribute access, which allows us to add the type hints to the attributes.

In principle, this would allow us to initialize without using `__init__` directly,
rather using only the `from_config` method.

```python
from collections.abc import Mapping

class Module: ...
class ModuleConfig(Mapping[str, object]): ...
type CFG = Mapping[str, object]

class Foo(Module):
    class Config(ModuleConfig):
        input_size: int
        hidden_size: int
        nested_module: Module | CFG
        use_bias: bool = True

    @classmethod
    def from_config(cls, config: "CFG") -> "Foo": ...
```

## JSON Convertability

JSON offers the following data types:

- `null`
- `bool`
- `int`
- `float`
- `str`
- `list[JSON]`
- `dict[str, JSON]`

We want to support the following types:

- `bool`, `None`, `int`, `float`, `str`
- `shape` -> `list[int]`
- `Tensor[int]` -> ``Nested[`list`, `int`]`` (at least 2D)
- `Tensor[float]` -> ``Nested[`list`, `float`]``
- `Tensor[bool` -> ``Nested[`list`, `bool`]``
- `Module` -> `ConfigDict`
- `Sequence[Module]` -> `list[ConfigDict]`
- `Mapping[str, Module]` -> `dict[str, ConfigDict]`

A dictionary is a `ConfigDict` if it contains at least the following keys:

- `__name__: str` - the name of: the module class
- `__module__: str` - the module name from which the class is imported
- any extra arguments required to initialize the module using the `from_config` method.
  (as a fallback, the `__init__` method of the class can be used, but this is not recommended)
  - for this to work, the `__init__` cannot require any positional-only or vararg arguments.

which allows us to initialize the module.

```python

class Model:
    def __init__(
        self,
        submodule: "Module",
        mandatory_arg: int,
        optional_arg: bool = True,
     ) -> None:
        ...
```
