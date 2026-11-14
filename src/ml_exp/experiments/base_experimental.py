from abc import ABC, abstractmethod
from enum import Enum
from typing import Generic, Self, Type, TypedDict, TypeVar

from torch.nn import Module

# Constrain the type variable so only Enum subclasses are accepted
VarEnum = TypeVar("VarEnum", bound=Enum)


class ParameterBreakdown(TypedDict):
    total: int
    layers: dict[str, int]


class BaseExperimentalModel[VarEnum](Module, ABC):
    # Subclasses must define their own Enum class here
    variables: VarEnum

    @abstractmethod
    def copy_scaled_to_fixed_p(self, var: VarEnum, pcount: int) -> Self:
        """Return a new model instance scaled along `var` to match `pcount`."""
        pass

    def count_parameters(self) -> ParameterBreakdown:
        """Return the total trainable parameter count."""
        d = {}
        for name, param in self.named_parameters():
            d[name] = param.numel()

        return {"total": sum(d.values()), "layers": d}
