"""Unit-aware tensors for SciMLx using Pint."""

import pint
from typing import Any, Union
from core.device import FRAMEWORK, to_array

# Import frameworks conditionally for type checking and isinstance
try:
    import torch
except ImportError:
    torch = None

try:
    import mlx.core as mx
except ImportError:
    mx = None

# Create a shared unit registry
ureg = pint.UnitRegistry()

class BackendGeneric:
    """
    A generic wrapper that delegates operations to the underlying 
    framework-specific tensor (Torch or MLX).
    """
    def __init__(self, data: Any):
        self.data = to_array(data)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.data})"

    def __getattr__(self, name):
        """Delegate missing attributes/methods to the underlying tensor."""
        return getattr(self.data, name)

    # Basic arithmetic delegation
    def __add__(self, other):
        other_data = other.data if isinstance(other, BackendGeneric) else other
        return self.__class__(self.data + other_data)

    def __sub__(self, other):
        other_data = other.data if isinstance(other, BackendGeneric) else other
        return self.__class__(self.data - other_data)

    def __mul__(self, other):
        other_data = other.data if isinstance(other, BackendGeneric) else other
        return self.__class__(self.data * other_data)

    def __truediv__(self, other):
        other_data = other.data if isinstance(other, BackendGeneric) else other
        return self.__class__(self.data / other_data)

class SciMLTensor(BackendGeneric):
    """A wrapper for framework-native tensors that maintains physical units."""
    
    def __init__(self, data: Any, units: Union[str, pint.Unit]):
        super().__init__(data)
        if isinstance(units, str):
            self.units = ureg(units).units
        else:
            self.units = units

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
            other = other.to(str(self.units))
        return SciMLTensor(self.data + other.data, self.units)

    def __sub__(self, other):
        if not isinstance(other, SciMLTensor):
            raise TypeError("Can only subtract SciMLTensor from SciMLTensor")
        if self.units != other.units:
            other = other.to(str(self.units))
        return SciMLTensor(self.data - other.data, self.units)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return SciMLTensor(self.data * other, self.units)
        
        if isinstance(other, SciMLTensor):
            new_units = self.units * other.units
            return SciMLTensor(self.data * other.data, new_units)
            
        # Fallback to BackendGeneric multiplication (e.g. with raw native tensors)
        res = super().__mul__(other)
        return SciMLTensor(res.data, self.units)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return SciMLTensor(self.data / other, self.units)
            
        if isinstance(other, SciMLTensor):
            new_units = self.units / other.units
            return SciMLTensor(self.data / other.data, new_units)

        # Fallback to BackendGeneric division
        res = super().__truediv__(other)
        return SciMLTensor(res.data, self.units)

def check_consistency(a: SciMLTensor, b: SciMLTensor):
    """Check if two tensors have compatible units."""
    return a.units.dimensionality == b.units.dimensionality
