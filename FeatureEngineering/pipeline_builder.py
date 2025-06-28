import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import List, Dict, Any, Optional, Union

# Import custom transformers
from transformers.date_transformer import DateTransformer
from transformers.crafted_features_transformer import CraftedFeaturesTransformer
from transformers.dtype_optimizer import DtypeOptimizer
from transformers.polynomial_transformer import PolynomialTransformer
from transformers.encoding_transformer import EncodingTransformer
from transformers.column_dropper import ColumnDropper


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
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() < 10:  # Small number of unique values suggests categorical
                column_types['categorical'].append(col)
            else:
                column_types['numerical'].append(col)
        elif pd.api.types.is_datetime64_dtype(df[col]):
            column_types['date'].append(col)
        elif df[col].nunique() < df.shape[0] * 0.05:  # Less than 5% unique values
            column_types['categorical'].append(col)
        else:
            column_types['other'].append(col)
            
    return column_types


def create_preprocessing_pipeline(
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    date_features: Optional[List[str]] = None,
    columns_to_drop: Optional[List[str]] = None,
    poly_degree: int = 2,
    encoding_method: str = 'onehot'
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
        pipeline_steps.append(('column_dropper', ColumnDropper(columns_to_drop=columns_to_drop)))

    # Step 2: Extract features from date columns, with a fallback to a default column name
    pipeline_steps.append(('date_features', DateTransformer(date_column=date_features[0] if date_features else 'ApplicationDate')))

    # Step 3: Create crafted features from existing ones
    pipeline_steps.append(('crafted_features', CraftedFeaturesTransformer()))

    # Step 4: Optimize data types for memory efficiency before transformations
    pipeline_steps.append(('first_dtype_optimizer', DtypeOptimizer()))

    # Step 5: Generate polynomial features for all available numerical columns
    # The transformer will automatically identify numerical columns at runtime.
    pipeline_steps.append(('polynomial_features', PolynomialTransformer(
        target_features=['AnnualIncome', 'CreditScore', 'LoanAmount', 'TotalAssets', 'TotalLiabilities', 'NetWorth'],
        degree=poly_degree
    )))

    # Step 6: Encode categorical features
    if categorical_features:
        pipeline_steps.append(('encoding_features', EncodingTransformer()))

    # Step 7: Final data type optimization after all transformations
    pipeline_steps.append(('final_dtype_optimizer', DtypeOptimizer()))

    # Create and return the final pipeline object
    pipeline = Pipeline(pipeline_steps)
    
    return pipeline


def build_pipeline_from_dataframe(
    df: pd.DataFrame, 
    columns_to_drop: Optional[List[str]] = None, 
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
            **kwargs
        )
    except Exception as e:
        raise RuntimeError(f"Failed to build pipeline: {str(e)}")