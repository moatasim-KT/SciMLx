from .fno          import FNO1d, FNO2d, FNO1dMC, UNO1d, RFNO1d, SpectralConv1d, SpectralConv2d
from .afno         import AFNO1d, AFNOBlock1d, AdaptiveSpectralMixer1d, FFNO1d, FFNOBlock1d, DiagSpectralConv1d
from .deeponet     import DeepONet, PODDeepONet
from .wno          import WNO1d, WaveletConv1d
from .s4d          import S4NO1d, S4DLayer
from .gnot         import GNOT1d, GNOT2d
from .pinn         import PINO1d, PINN
from .tfno         import (TFNO1d, RTFNO1d, CPFNO1d, TFNO2d,
                            TuckerSpectralConv1d, CPSpectralConv1d, TuckerSpectralConv2d)
from .transolver   import Transolver1d, Transolver2d
from .time_deeponet import TimeDeepONet1d, DualBranchDeepONet1d
from .hnn          import HamiltonianNO1d, HamiltonianNet1d, EnergyConservingFNO1d
from .neural_ode   import NeuralODE1d, UniversalDE1d, LatentODE1d
from .ssno         import SSNO1d

__all__ = [
    # FNO family
    "FNO1d", "FNO2d", "FNO1dMC",
    "UNO1d",
    "RFNO1d",
    "SpectralConv1d", "SpectralConv2d",
    # AFNO/FFNO family
    "AFNO1d", "AFNOBlock1d", "AdaptiveSpectralMixer1d",
    "FFNO1d", "FFNOBlock1d", "DiagSpectralConv1d",
    # DeepONet family
    "DeepONet", "PODDeepONet",
    # WNO family
    "WNO1d", "WaveletConv1d",
    # State Space Models
    "S4NO1d", "S4DLayer",
    # GNOT family
    "GNOT1d", "GNOT2d",
    # PINN family
    "PINO1d", "PINN",
    # TFNO family (Tucker/CP factorized FNO) — from PhysicsNeMo
    "TFNO1d", "RTFNO1d", "CPFNO1d", "TFNO2d",
    "TuckerSpectralConv1d", "CPSpectralConv1d", "TuckerSpectralConv2d",
    # Transolver (Physics Attention Transformer) — from PhysicsNeMo/NeurIPS 2024
    "Transolver1d", "Transolver2d",
    # Time-Marching DeepONet — from FE-NO coupling paper (CMAME 2025)
    "TimeDeepONet1d", "DualBranchDeepONet1d",
    # Hamiltonian Neural Networks — from Greydanus et al. NeurIPS 2019 / MathWorks examples
    "HamiltonianNO1d", "HamiltonianNet1d", "EnergyConservingFNO1d",
    # Neural ODEs & Universal Differential Equations — from Chen et al. / Rackauckas et al.
    "NeuralODE1d", "UniversalDE1d", "LatentODE1d",
    # State-Space Neural Operator (SS-NO) — adaptive S4D + spectral conv dual-branch
    "SSNO1d",
]
