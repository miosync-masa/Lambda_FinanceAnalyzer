# lambda3/core/__init__.py
"""
Lambda³ Core Modules

核心機能の段階的インポート
"""

from .config import L3BaseConfig, L3ComprehensiveConfig
from .structural_tensor import StructuralTensorFeatures

try:
    from .jit_functions import test_jit_functions_fixed as test_jit_functions
    JIT_AVAILABLE = True
except ImportError:
    JIT_AVAILABLE = False

__all__ = ['L3BaseConfig', 'L3ComprehensiveConfig', 'StructuralTensorFeatures']

if JIT_AVAILABLE:
    __all__.append('test_jit_functions')
