import torch
from torch import Tensor, jit, nn


class Bar(nn.Module):
    @jit.export
    def exported_method(self) -> int:
        return 7


class Baz(Bar):
    pass


class Foo(nn.Module):
    weight: Tensor

    def __init__(self, m: int, n: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(m, n))
        self.submodule = Baz()

    @jit.export
    def exported_method(self) -> int:
        return 42


model = Foo(2, 3)

print(model.exported_method())
print(model.submodule.exported_method())

model = jit.script(model)  # scripting the model

print(model.exported_method())
print(model.submodule.exported_method())

jit.save(model, "model.pt")  # saving and reloading the model
model = jit.load("model.pt")

print(model.exported_method())  # works! ✓
print(model.submodule.exported_method())
