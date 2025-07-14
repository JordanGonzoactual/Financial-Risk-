import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import List, Dict, Any, Optional, Union

# Import custom transformers
import logging
from .transformers.base_transformer import BaseTransformer
from .transformers.date_transformer import DateTransformer
from .transformers.crafted_features_transformer import CraftedFeaturesTransformer
from .transformers.dtype_optimizer import DtypeOptimizer
from .transformers.polynomial_transformer import PolynomialTransformer
from .transformers.encoding_transformer import EncodingTransformer
from .transformers.column_dropper import ColumnDropper

# Configure logger for this module
logger = logging.getLogger(__name__)


def identify_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Identify numerical, categorical, date and other columns in the dataframe.

    Args:
        df: Input dataframe

    Returns:
        Dictionary with column types as keys and lists of column names as values
    """
    column_types = {
        'numerical': [],
        'categorical': [],
        'date': [],
        'other': []
    }

    for col in df.columns:
        col_dtype = df[col].dtype

        # 1. Date Identification: Check for datetime dtype or object type with a date-like name.
        if pd.api.types.is_datetime64_any_dtype(col_dtype) or \
           (pd.api.types.is_object_dtype(col_dtype) and ('date' in col.lower() or col.lower() == 'applicationdate')):
            column_types['date'].append(col)

        # 2. Numerical Identification
        elif pd.api.types.is_numeric_dtype(col_dtype):
            if df[col].nunique() < 10:  # Treat low-cardinality numerics as categorical
                column_types['categorical'].append(col)
            else:
                column_types['numerical'].append(col)

        # 3. Categorical Identification for remaining columns
        elif df[col].nunique() < df.shape[0] * 0.05:
            column_types['categorical'].append(col)

        # 4. Fallback for other types (e.g., high-cardinality strings)
        else:
            column_types['other'].append(col)

    logger.info(f"Identified column types: {column_types}")
    return column_types


def create_preprocessing_pipeline(
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    date_features: Optional[List[str]] = None,
    columns_to_drop: Optional[List[str]] = None,
    poly_degree: int = 2,
    encoding_method: str = 'onehot',
    verbose: bool = False
) -> Pipeline:
    """
    Create a complete preprocessing pipeline.
    
    Args:
        numerical_features: List of numerical column names
        categorical_features: List of categorical column names
        date_features: List of date column names
        columns_to_drop: List of columns to be dropped
        poly_degree: Degree for polynomial features
        encoding_method: Encoding method ('onehot', 'label', etc.)
        
    Returns:
        Configured sklearn Pipeline object
    """
    if not any([numerical_features, categorical_features, date_features]):
        raise ValueError("At least one feature type must be specified")

    pipeline_steps = []

    # Step 1: Conditionally drop specified columns first
    if columns_to_drop:
        pipeline_steps.append(('column_dropper', ColumnDropper(
            columns_to_drop=columns_to_drop,
            verbose=verbose
        )))

    # Step 2: Always process date features, defaulting to 'ApplicationDate'
    pipeline_steps.append(('date_features', DateTransformer(
        date_column='ApplicationDate',
        verbose=verbose
    )))

    # Step 3: Create crafted features, handling division by zero by returning 0
    pipeline_steps.append(('crafted_features', CraftedFeaturesTransformer(
        zero_division_mode="zero",
        verbose=verbose
    )))



    # Step 5: Generate polynomial features for all available numerical columns
    # The transformer will automatically identify numerical columns at runtime.
    pipeline_steps.append(('polynomial_features', PolynomialTransformer(
        degree=poly_degree,
        verbose=verbose
    )))

    # Step 6: Encode categorical features
    if categorical_features:
        pipeline_steps.append(('encoding_features', EncodingTransformer(verbose=verbose)))

    # Step 7: Final data type optimization after all transformations
    pipeline_steps.append(('final_dtype_optimizer', DtypeOptimizer(verbose=verbose)))

    # Create the final pipeline object
    pipeline = Pipeline(pipeline_steps)

    # --- Pipeline Validation ---
    # Ensure all core transformers are present.
    pipeline_step_names = [step[0] for step in pipeline.steps]
    required_steps = {
        'date_features',
        'crafted_features',
        'polynomial_features',
        'final_dtype_optimizer'
    }

    missing_steps = required_steps - set(pipeline_step_names)
    if missing_steps:
        error_msg = f"Pipeline validation failed. Missing required steps: {', '.join(missing_steps)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("Successfully built and validated preprocessing pipeline.")
    logger.info(f"Pipeline steps: {pipeline_step_names}")

    return pipeline


def build_pipeline_from_dataframe(
    df: pd.DataFrame, 
    columns_to_drop: Optional[List[str]] = None,
    verbose: bool = False, 
    **kwargs
) -> Pipeline:
    """
    Build a preprocessing pipeline by automatically identifying column types from a dataframe.
    
    Args:
        df: Input dataframe
        columns_to_drop: List of columns to be dropped
        **kwargs: Additional arguments to pass to create_preprocessing_pipeline
        
    Returns:
        Configured sklearn Pipeline object
    """
    try:
        # Exclude columns that will be dropped from column type identification
        df_filtered = df.drop(columns=columns_to_drop or [], errors='ignore')
        column_types = identify_column_types(df_filtered)
        
        return create_preprocessing_pipeline(
            numerical_features=column_types.get('numerical'),
            categorical_features=column_types.get('categorical'),
            date_features=column_types.get('date'),
            columns_to_drop=columns_to_drop,
            verbose=verbose,
            **kwargs
        )
    except Exception as e:
        raise RuntimeError(f"Failed to build pipeline: {str(e)}")