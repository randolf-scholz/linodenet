# Conventions

We distinguish between two types of models:

1. **Basic Components** - these are building blocks used to construct larger models.
   - Generally do not have any submodules. (attributes that are `nn.Module`s)
   - Examples: `nn.Linear`, `nn.Conv2d`, `nn.BatchNorm2d`, `nn.ReLU`, etc.
   - Convention: `__init__` takes `hyperparameters` as arguments.
2. Complex Models - these are the models that are built from the basic components
   - Generally have several submodules. (attributes that are `nn.Module`s)
   - Examples: `nn.Sequential`, `ResNet`
   - Conventions:
     - `__init__` takes submodules as input, and hyperparameters only as necessary.
     - provide a `from_hyperparameters` method that allows to initialize from plain JSON.
       Here, submodules can be initialized using the dictionaries that provide the hyperparameters.
       NOTE: `from_hyperparameters` should be a `@classmethod`, and **always** return `cls(*args, **kwargs)`,
       i.e. it needs to go through the regular `__init__` method.
   - These take `hyperparameters` as arguments to the constructor `__init__`.

## Specifying Hyperparameters

1. Every model **type** should have an associated dictionary-like structure that contains
   default hyperparameters. Candidates:
   - plain dictionary
   - typed dictionary (e.g. `TypedDict`).
   - `dataclasses.dataclass`
   - `pydantic.BaseModel`

   That is, `module_type` should have a class attribute that produces such a thing.

2. Every model **instance** should have a JSON-serializable dictionary that contains
   the actual hyperparameters. This dictionary should be sufficient to reproducibly
   initialize the model via the `from_hyperparameters` classmethod.

   In particular, this dictionary should contain, recursively, the hyperparameters
   of all submodules.

## Hyperparameter Dictionary

Let `model` be a model instance. Then, `model.hparams` should be a dictionary
with the following contents:

1. Entries `__module__: str` and `__name__: str` which can be used to dynamically import the
   model class.
2. For any **direct**, **named** submodule the entry `<name>: HP_DICT = <submodule>.hparams`
   should be present.
3. For any **direct**, **unnamed** submodules we consider 2 cases:
   1. submodules organized in a `nn.ModuleList` or `nn.Sequential`.
      In this case, we have the entry `<name>: list[HP_DICT] = [sub.hparams for sub in <modules>]`
   2. submodules organized in a `nn.ModuleDict`.
      In this case, we have the entry
      `<name>: dict[str, HP_DICT] = {name: sub.hparams for <name>, sub in <modules>.items()}`

Sometimes, models have attributes that are derived from other attributes. If we represent the
hyperparameters as a pydantic model, we would use `computed` fields for this. When creating
the hyperparameter dictionary, such fields are not included.

We could consider a "extended" hyperparameter dictionary that includes computed fields.

## Initialization from Hyperparameters

Let `model` be a model instance. Then, `model.from_hyperparameters(model.hparams)` should
reproduce the model, up to training state, and potentially up to random initialization.
Getting random initialization right is hard, and will be left for later.

## Synthesizing Hyperparameter Dictionary

Take for example `nn.Linear`. The relevant hyperparameters are:
`in_features: int`, `out_features: int`, `bias: bool`, and potentially `device: str`, `dtype: str`.
However, the module does not provide an obvious way how to get these hyperparameters from an instance.

Therefore, we need a function `get_hparams(module: nn.Module) -> HP_DICT`
that takes a model instance and returns a dictionary, roughly as follows:

```python
def get_hparams(module: nn.Module) -> dict:
    if hasattr(module, "hparams"):
        return module.hparams

    # else: dispatch on type(module)
    cls = type(module)
    known_types: dict[type[nn.Module], Callable[[nn.Module], dict]] = {
        nn.Linear: __get_hparams_linear,
        nn.Conv2d: __get_hparams_conv2d,
        ...
    }
    try:
        getter = known_types[cls]
    except KeyError as exc:
        raise NotImplementedError(f"Cannot synthesize hyperparameters for {cls}") from exc
    return getter(module)
```

## Serialization to JSON

- Simply recursively query the `hparams` attribute of the model instance.
- If a model does not have a `hparams` attribute, we need a fallback method that attempts to synthesize
  the hyperparameters from the instance.
- NOTE: `torch.jit.script` does not support `@property` decorators. Hence, the hparams attribute
  needs to be created at initialization time.
