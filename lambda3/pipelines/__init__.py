# lambda3/pipelines/__init__.py  
"""
Lambda³ Pipeline Modules

パイプライン機能の段階的インポート
"""

try:
    from .comprehensive import Lambda3ComprehensivePipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

__all__ = []

if PIPELINE_AVAILABLE:
    __all__.append('Lambda3ComprehensivePipeline')
