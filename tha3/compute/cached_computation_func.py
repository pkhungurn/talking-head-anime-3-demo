from typing import Callable, Dict, List

from torch import Tensor
from torch.nn import Module

TensorCachedComputationFunc = Callable[
    [Dict[str, Module], List[Tensor], Dict[str, List[Tensor]]], Tensor]
TensorListCachedComputationFunc = Callable[
    [Dict[str, Module], List[Tensor], Dict[str, List[Tensor]]], List[Tensor]]
