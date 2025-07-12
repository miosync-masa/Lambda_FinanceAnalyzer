# lambda3/visualization/__init__.py
"""
Lambda³ Visualization Modules

可視化機能の段階的インポート
"""

try:
    from .base import Lambda3BaseVisualizer, apply_lambda3_style
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

__all__ = []

if VISUALIZATION_AVAILABLE:
    __all__.extend(['Lambda3BaseVisualizer', 'apply_lambda3_style'])
