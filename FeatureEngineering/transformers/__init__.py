"""
Transformers Module

Contains all custom sklearn transformers for feature engineering:
- BaseTransformer: Common functionality for all transformers
- CraftedFeaturesTransformer: Domain-specific financial features
- PolynomialTransformer: Polynomial feature creation
- DateTransformer: Date feature extraction
- DtypeOptimizer: Data type optimization
- EncodingTransformer: Categorical encoding

Each transformer follows sklearn's BaseEstimator and TransformerMixin pattern.
"""

from .base_transformer import BaseTransformer
from .crafted_features_transformer import CraftedFeaturesTransformer
from .polynomial_transformer import PolynomialTransformer
from .date_transformer import DateTransformer
from .dtype_optimizer import DtypeOptimizer
from .encoding_transformer import EncodingTransformer

__all__ = [
    "BaseTransformer",
    "CraftedFeaturesTransformer", 
    "PolynomialTransformer",
    "DateTransformer",
    "DtypeOptimizer",
    "EncodingTransformer"
]
