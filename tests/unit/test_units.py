import pytest
import numpy as np
from core.units import SciMLTensor, ureg
from core.device import FRAMEWORK

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

def test_scimltensor_init():
    data = np.array([1.0, 2.0, 3.0])
    tensor = SciMLTensor(data, "meter")
    assert tensor.units == ureg("meter").units
    assert np.allclose(np.array(tensor.data), data)

def test_scimltensor_conversion():
    data = np.array([100.0])
    tensor = SciMLTensor(data, "cm")
    converted = tensor.to("meter")
    assert converted.units == ureg("meter").units
    assert np.allclose(np.array(converted.data), np.array([1.0]))

def test_scimltensor_addition():
    t1 = SciMLTensor([1.0], "meter")
    t2 = SciMLTensor([50.0], "cm")
    res = t1 + t2
    assert res.units == ureg("meter").units
    assert np.allclose(np.array(res.data), np.array([1.5]))

def test_scimltensor_multiplication():
    t1 = SciMLTensor([2.0], "meter")
    t2 = SciMLTensor([3.0], "second")
    res = t1 * t2
    assert res.units == ureg("meter * second").units
    assert np.allclose(np.array(res.data), np.array([6.0]))

def test_scimltensor_native_ops():
    data = np.array([1.0, 2.0])
    t = SciMLTensor(data, "kg")
    
    # Multiplication by scalar
    res = t * 2.0
    assert np.allclose(np.array(res.data), data * 2.0)
    
    # Multiplication by native array/tensor
    if FRAMEWORK == "torch" and HAS_TORCH:
        native = torch.tensor([2.0, 3.0])
        res = t * native
        assert torch.is_tensor(res.data)
    elif FRAMEWORK == "mlx" and HAS_MLX:
        native = mx.array([2.0, 3.0])
        res = t * native
        assert isinstance(res.data, mx.array)
