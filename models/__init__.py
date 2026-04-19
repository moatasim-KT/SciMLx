from .fno import FNO1d, FNO2d, FNO1dMC, UNO1d, RFNO1d, RFNO2d, SpectralConv1d, SpectralConv2d
from .afno import AFNO1d, AFNOBlock1d, AdaptiveSpectralMixer1d, FFNO1d, FFNOBlock1d, DiagSpectralConv1d
from .deeponet import DeepONet, PODDeepONet
from .wno import WNO1d, WaveletConv1d
from .s4d import S4NO1d, S4DLayer
from .gnot import GNOT1d, GNOT2d
from .pinn import PINO1d, PINN
from .tfno import TFNO1d, RTFNO1d, CPFNO1d, TFNO2d, TuckerSpectralConv1d, CPSpectralConv1d, TuckerSpectralConv2d
from .transolver import Transolver1d, Transolver2d
from .time_deeponet import TimeDeepONet1d, DualBranchDeepONet1d
from .hnn import HamiltonianNO1d, HamiltonianNet1d, EnergyConservingFNO1d
from .neural_ode import NeuralODE1d, UniversalDE1d, LatentODE1d
from .ssno import SSNO1d

from .pacmann import PACMANN
from .vsmno import VSMNO2d

from .mamba_no  import MambaNO1d
__all__ = [
	"PACMANN",
	"VSMNO2d",
	"PINN",
	# AFNO/FFNO family
	"AFNO1d",
	"AFNOBlock1d",
	"AdaptiveSpectralMixer1d",
	"CPFNO1d",
	"CPSpectralConv1d",
	# DeepONet family
	"DeepONet",
	"DiagSpectralConv1d",
	"DualBranchDeepONet1d",
	"EnergyConservingFNO1d",
	"FFNO1d",
	"FFNOBlock1d",
	# FNO family
	"FNO1d",
	"FNO1dMC",
	"FNO2d",
	# GNOT family
	"GNOT1d",
	"GNOT2d",
	# Hamiltonian Neural Networks — from Greydanus et al. NeurIPS 2019 / MathWorks examples
	"HamiltonianNO1d",
	"HamiltonianNet1d",
	"LatentODE1d",
	# Neural ODEs & Universal Differential Equations — from Chen et al. / Rackauckas et al.
	"NeuralODE1d",
	# PINN family
	"PINO1d",
	"PODDeepONet",
	"RFNO1d",
	"RFNO2d",
	"RTFNO1d",
	"S4DLayer",
	# State Space Models
	"S4NO1d",
	# State-Space Neural Operator (SS-NO) — adaptive S4D + spectral conv dual-branch
	"SSNO1d",
	"SpectralConv1d",
	"SpectralConv2d",
	# TFNO family (Tucker/CP factorized FNO) — from PhysicsNeMo
	"TFNO1d",
	"TFNO2d",
	# Time-Marching DeepONet — from FE-NO coupling paper (CMAME 2025)
	"TimeDeepONet1d",
	# Transolver (Physics Attention Transformer) — from PhysicsNeMo/NeurIPS 2024
	"Transolver1d",
	"Transolver2d",
	"TuckerSpectralConv1d",
	"TuckerSpectralConv2d",
	"UNO1d",
	"UniversalDE1d",
	# WNO family
	"WNO1d",
	"WaveletConv1d",
    "MambaNO1d",
]
