# lambda3/analysis/__init__.py

try:
    from .hierarchical import HierarchicalAnalyzer
    HIERARCHICAL_AVAILABLE = True
except ImportError:
    HIERARCHICAL_AVAILABLE = False

try:
    from .pairwise import PairwiseAnalyzer  
    PAIRWISE_AVAILABLE = True
except ImportError:
    PAIRWISE_AVAILABLE = False

__all__ = []

if HIERARCHICAL_AVAILABLE:
    __all__.append('HierarchicalAnalyzer')
    
if PAIRWISE_AVAILABLE:
    __all__.append('PairwiseAnalyzer')
