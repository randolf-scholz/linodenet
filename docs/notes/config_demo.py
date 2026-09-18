from collections.abc import Mapping


class Module: ...


class ModuleConfig(Mapping[str, object]): ...


type CFG = Mapping[str, object]

# class Foo(Module):
#     class Config(ModuleConfig):
#         input_size: int
#         hidden_size: int
#         nested_module: Module | CFG
#         use_bias: bool = True
#
#     @classmethod
#     def from_config(cls, config: "CFG") -> "Foo": ...


class Foo:
    class CFG:
        __name__ = __qualname__
        __module__ = __module__


class Bar:
    CFG = {
        "__name__": __qualname__,
        "__module__": __name__,
    }


print(f"{Foo.CFG.__name__=}", f"{Foo.CFG.__module__=}", sep="\n")
print(f"{Bar.CFG['__name__']=}", f"{Bar.CFG['__module__']=}", sep="\n")
print(f"{Foo.CFG.__qualname__=}")
