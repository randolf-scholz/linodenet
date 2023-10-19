#!/usr/bin/env python
# FIXME: https://github.com/pytorch/pytorch/issues/110959
from collections.abc import Mapping

from torch import Tensor, nn


class Foo(nn.Module, Mapping):
    tensors: dict[str, Tensor]

    def __len__(self):
        return len(self.tensors)

    def __iter__(self):
        return iter(self.tensors)

    def __getitem__(self, key):
        return self.tensors[key]

    def __init__(self, tensors: dict[str, Tensor]) -> None:
        super().__init__()
        self.tensors = tensors


m = Foo({})
print(dict(m.named_buffers()))
