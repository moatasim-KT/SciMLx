# All SciMLx Models - PyTorch/CUDA Optimized

from .fno import FNO1d, FNO2d, SpectralConv1d, SpectralConv2d, RFNO1d, RFNO2d, UNO1d, UNO2d
from .afno import AFNO1d, FFNO1d
from .deeponet import DeepONet, PODDeepONet
from .wno import WNO1d, WNO2d, WNO_GNOT
from .s4d import S4NO1d
from .gnot import GNOT1d, GNOT2d
from .pinn import PINO1d, PINN, ModalPINN
from .tfno import TFNO1d, TFNO2d
from .transolver import Transolver1d, Transolver2d
from .time_deeponet import TimeDeepONet1d, DualBranchDeepONet1d
from .hnn import HamiltonianNO1d
from .neural_ode import NeuralODE1d, UniversalDE1d
from .mamba_no import MambaNO1d
from .sar import SARModel2d
from .pacmann import PACMANN
from .vsmno import VSMNO2d
from .ssno import SSNO1d
from .mem_no import MemNO1d
from .kan import KAN_FNO
from .chebyshev_kan import cPIKAN_FNO

__all__ = [
    "FNO1d", "FNO2d", "SpectralConv1d", "SpectralConv2d", "RFNO1d", "RFNO2d", "UNO1d", "UNO2d",
    "AFNO1d", "FFNO1d",
    "DeepONet", "PODDeepONet",
    "WNO1d", "WNO2d", "WNO_GNOT",
    "S4NO1d",
    "GNOT1d", "GNOT2d",
    "PINO1d", "PINN", "ModalPINN",
    "TFNO1d", "TFNO2d",
    "Transolver1d", "Transolver2d",
    "TimeDeepONet1d", "DualBranchDeepONet1d",
    "HamiltonianNO1d",
    "NeuralODE1d", "UniversalDE1d",
    "MambaNO1d",
    "SARModel2d",
    "PACMANN",
    "VSMNO2d",
    "SSNO1d",
    "MemNO1d",
    "KAN_FNO",
    "cPIKAN_FNO"
]
