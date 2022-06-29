from typing import Optional

from torch.nn import Module, ReLU, LeakyReLU, ELU, ReLU6, Hardswish, SiLU, Tanh, Sigmoid

from tha3.module.module_factory import ModuleFactory


class ReLUFactory(ModuleFactory):
    def __init__(self, inplace: bool = False):
        self.inplace = inplace

    def create(self) -> Module:
        return ReLU(self.inplace)


class LeakyReLUFactory(ModuleFactory):
    def __init__(self, inplace: bool = False, negative_slope: float = 1e-2):
        self.negative_slope = negative_slope
        self.inplace = inplace

    def create(self) -> Module:
        return LeakyReLU(inplace=self.inplace, negative_slope=self.negative_slope)


class ELUFactory(ModuleFactory):
    def __init__(self, inplace: bool = False, alpha: float = 1.0):
        self.alpha = alpha
        self.inplace = inplace

    def create(self) -> Module:
        return ELU(inplace=self.inplace, alpha=self.alpha)


class ReLU6Factory(ModuleFactory):
    def __init__(self, inplace: bool = False):
        self.inplace = inplace

    def create(self) -> Module:
        return ReLU6(inplace=self.inplace)


class SiLUFactory(ModuleFactory):
    def __init__(self, inplace: bool = False):
        self.inplace = inplace

    def create(self) -> Module:
        return SiLU(inplace=self.inplace)


class HardswishFactory(ModuleFactory):
    def __init__(self, inplace: bool = False):
        self.inplace = inplace

    def create(self) -> Module:
        return Hardswish(inplace=self.inplace)


class TanhFactory(ModuleFactory):
    def create(self) -> Module:
        return Tanh()


class SigmoidFactory(ModuleFactory):
    def create(self) -> Module:
        return Sigmoid()


def resolve_nonlinearity_factory(nonlinearity_fatory: Optional[ModuleFactory]) -> ModuleFactory:
    if nonlinearity_fatory is None:
        return ReLUFactory(inplace=False)
    else:
        return nonlinearity_fatory
