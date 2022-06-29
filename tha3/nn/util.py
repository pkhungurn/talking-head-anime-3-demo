from typing import Optional, Callable, Union

from torch.nn import Module

from tha3.module.module_factory import ModuleFactory
from tha3.nn.init_function import create_init_function
from tha3.nn.nonlinearity_factory import resolve_nonlinearity_factory
from tha3.nn.normalization import NormalizationLayerFactory
from tha3.nn.spectral_norm import apply_spectral_norm


def wrap_conv_or_linear_module(module: Module,
                               initialization_method: Union[str, Callable[[Module], Module]],
                               use_spectral_norm: bool):
    if isinstance(initialization_method, str):
        init = create_init_function(initialization_method)
    else:
        init = initialization_method
    return apply_spectral_norm(init(module), use_spectral_norm)


class BlockArgs:
    def __init__(self,
                 initialization_method: Union[str, Callable[[Module], Module]] = 'he',
                 use_spectral_norm: bool = False,
                 normalization_layer_factory: Optional[NormalizationLayerFactory] = None,
                 nonlinearity_factory: Optional[ModuleFactory] = None):
        self.nonlinearity_factory = resolve_nonlinearity_factory(nonlinearity_factory)
        self.normalization_layer_factory = normalization_layer_factory
        self.use_spectral_norm = use_spectral_norm
        self.initialization_method = initialization_method

    def wrap_module(self, module: Module) -> Module:
        return wrap_conv_or_linear_module(module, self.get_init_func(), self.use_spectral_norm)

    def get_init_func(self) -> Callable[[Module], Module]:
        if isinstance(self.initialization_method, str):
            return create_init_function(self.initialization_method)
        else:
            return self.initialization_method
