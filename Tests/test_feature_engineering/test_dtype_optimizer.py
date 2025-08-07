#!/usr/bin/env python3
"""
Test script to verify the DtypeOptimizer handles categorical fields correctly.
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from FeatureEngineering.transformers.dtype_optimizer import DtypeOptimizer

def test_dtype_optimizer():
    """Test the DtypeOptimizer with categorical fields."""
    
    # Create sample data with the categorical fields
    sample_data = {
        'BankruptcyHistory': ['No', 'Yes', 'No', 'Yes'],
        'PreviousLoanDefaults': ['None', 'One', 'Multiple', 'None'],
        'PaymentHistory': [1, 2, 3, 1],
        'UtilityBillsPaymentHistory': [0.8, 0.9, 0.7, 0.95],
        'LoanPurpose': ['Home', 'Auto', 'Personal', 'Home'],
        'SomeNumericField': [100, 200, 300, 400],
        'SomeFloatField': [1.5, 2.5, 3.5, 4.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame dtypes:")
    print(df.dtypes)
    print("\nOriginal DataFrame:")
    print(df)
    
    # Initialize and fit the DtypeOptimizer
    optimizer = DtypeOptimizer(verbose=True)
    optimizer.fit(df)
    
    print(f"\nOptimizations determined: {optimizer.optimizations_}")
    
    # Transform the data
    df_optimized = optimizer.transform(df)
    
    print("\nOptimized DataFrame dtypes:")
    print(df_optimized.dtypes)
    print("\nOptimized DataFrame:")
    print(df_optimized)
    
    # Verify the expected dtypes
    expected_dtypes = {
        'BankruptcyHistory': 'float64',
        'PreviousLoanDefaults': 'category',
        'PaymentHistory': 'int64',
        'UtilityBillsPaymentHistory': 'float64',
        'LoanPurpose': 'category'
    }
    
    print("\nVerifying expected dtypes:")
    for col, expected_dtype in expected_dtypes.items():
        actual_dtype = str(df_optimized[col].dtype)
        if expected_dtype == 'category':
            is_correct = df_optimized[col].dtype.name == 'category'
        else:
            is_correct = actual_dtype == expected_dtype
        
        status = "✓" if is_correct else "✗"
        print(f"{status} {col}: expected {expected_dtype}, got {actual_dtype}")
    
    # Test XGBoost compatibility - check for no object dtypes
    object_columns = df_optimized.select_dtypes(include=['object']).columns.tolist()
    if object_columns:
        print(f"\n⚠️  Warning: Found object dtypes in columns: {object_columns}")
        return False
    else:
        print("\n✓ No object dtypes found - XGBoost compatible!")
        return True

if __name__ == "__main__":
    success = test_dtype_optimizer()
    if success:
        print("\n🎉 DtypeOptimizer test passed!")
    else:
        print("\n❌ DtypeOptimizer test failed!")
