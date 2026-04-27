"""Unit-aware tensors for SciMLx using Pint."""

import torch
import pint
from typing import Any, Union

# Create a shared unit registry
ureg = pint.UnitRegistry()

class SciMLTensor:
    """A wrapper for torch.Tensor that maintains physical units."""
    
    def __init__(self, data: torch.Tensor, units: Union[str, pint.Unit]):
        if isinstance(units, str):
            self.units = ureg(units).units
        else:
            self.units = units
        self.data = data

    def __repr__(self):
        return f"SciMLTensor({self.data}, units={self.units})"

    def to(self, new_units: str):
        """Convert to new units."""
        factor = ureg.convert(1.0, self.units, new_units)
        return SciMLTensor(self.data * factor, new_units)

    def __add__(self, other):
        if not isinstance(other, SciMLTensor):
            raise TypeError("Can only add SciMLTensor to SciMLTensor")
        if self.units != other.units:
            # Try to convert other to self.units
            other = other.to(str(self.units))
        return SciMLTensor(self.data + other.data, self.units)

    def __sub__(self, other):
        if not isinstance(other, SciMLTensor):
            raise TypeError("Can only subtract SciMLTensor from SciMLTensor")
        if self.units != other.units:
            other = other.to(str(self.units))
        return SciMLTensor(self.data - other.data, self.units)

    def __mul__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            return SciMLTensor(self.data * other, self.units)
        if isinstance(other, SciMLTensor):
            new_units = self.units * other.units
            return SciMLTensor(self.data * other.data, new_units)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            return SciMLTensor(self.data / other, self.units)
        if isinstance(other, SciMLTensor):
            new_units = self.units / other.units
            return SciMLTensor(self.data / other.data, new_units)
        return NotImplemented

def check_consistency(a: SciMLTensor, b: SciMLTensor):
    """Check if two tensors have compatible units."""
    return a.units.dimensionality == b.units.dimensionality
