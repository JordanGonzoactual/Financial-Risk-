#!/usr/bin/env python3
"""
Test script to verify the EncodingTransformer handles enum fields correctly.
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from FeatureEngineering.transformers.encoding_transformer import EncodingTransformer

def test_encoding_transformer():
    """
    Test the EncodingTransformer with:
    - Binary integer fields (should pass through unchanged)
    - Categorical string fields
    """
    
    # Create sample data with the enum fields
    sample_data = {
        'BankruptcyHistory': [1, 0, 1, 0],
        'PreviousLoanDefaults': [1, 0, 1, 0],
        'PaymentHistory': [4, 3, 2, 1],  # Numeric values (no encoding needed)
        'UtilityBillsPaymentHistory': [4, 3, 2, 1],  # Numeric values (no encoding needed)
        'EducationLevel': ['High School', 'Bachelor', 'Master', 'Doctorate'],
        'HomeOwnershipStatus': ['Own', 'Rent', 'Mortgage', 'Own'],
        'EmploymentStatus': ['Employed', 'Unemployed', 'Self_Employed', 'Employed'],
        'LoanPurpose': ['Home', 'Auto', 'Education', 'Home'],
        'SomeNumericField': [100, 200, 300, 400]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nOriginal DataFrame dtypes:")
    print(df.dtypes)
    
    # Initialize and fit the EncodingTransformer
    encoder = EncodingTransformer(verbose=True)
    encoder.fit(df)
    
    # Transform the data
    df_encoded = encoder.transform(df)
    
    print("\nEncoded DataFrame:")
    print(df_encoded)
    print("\nEncoded DataFrame dtypes:")
    print(df_encoded.dtypes)
    
    # Verify the expected transformations
    expected_enum_values = {
        'BankruptcyHistory': [1, 0, 1, 0],
        'PreviousLoanDefaults': [1, 0, 1, 0],
        'PaymentHistory': [4, 3, 2, 1],
        'UtilityBillsPaymentHistory': [4, 3, 2, 1]
    }
    
    print("\nVerifying enum transformations:")
    all_correct = True
    for col, expected_values in expected_enum_values.items():
        if col in df_encoded.columns:
            actual_values = df_encoded[col].tolist()
            is_correct = actual_values == expected_values
            status = "✓" if is_correct else "✗"
            print(f"{status} {col}: expected {expected_values}, got {actual_values}")
            if not is_correct:
                all_correct = False
        else:
            print(f"✗ {col}: column not found in encoded data")
            all_correct = False
    
    # Check for object dtypes (should only be from one-hot encoding)
    object_columns = df_encoded.select_dtypes(include=['object']).columns.tolist()
    if object_columns:
        print(f"\nObject columns remaining: {object_columns}")
        expected_object_cols = [col for col in df_encoded.columns 
                              if any(base in col for base in ['HomeOwnershipStatus', 'EmploymentStatus', 'LoanPurpose'])]
        unexpected_object_cols = [col for col in object_columns if col not in expected_object_cols]
        if unexpected_object_cols:
            print(f"⚠️  Unexpected object columns: {unexpected_object_cols}")
            all_correct = False
        else:
            print("✓ Object columns are expected (one-hot encoded)")
    else:
        print("✓ No object columns found")
    
    return all_correct

if __name__ == "__main__":
    success = test_encoding_transformer()
    if success:
        print("\n🎉 EncodingTransformer test passed!")
    else:
        print("\n❌ EncodingTransformer test failed!")
