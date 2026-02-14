import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Added OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os

def load_data(filepath):
    """Loads the dataset from a CSV file."""
    return pd.read_csv(filepath)

def create_preprocessing_pipeline(numeric_features):
    """Creates a scikit-learn preprocessing pipeline."""
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # No categorical features in the current generator, but good to have the structure
    # If we added 'Education' or 'State', we would use OneHotEncoder here.
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ],
        remainder='passthrough' # Keep target if passed, or other columns
    )
    
    return preprocessor

def preprocess_and_save(data_path, output_dir="models"):
    """Loads data, preprocesses it, splits it, and saves the pipeline."""
    
    df = load_data(data_path)
    
    target = 'Default'
    X = df.drop(columns=[target])
    y = df[target]
    
    # Identify numeric features
    numeric_features = ['Age', 'Income', 'LoanAmount', 'CreditScore', 
                        'YearsEmployed', 'NumCreditLines', 'InterestRate', 
                        'LoanTerm', 'DTIRatio']
    
    # Create pipeline
    pipeline = create_preprocessing_pipeline(numeric_features)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit pipeline on training data
    # Note: We fit on X_train, then transform both. 
    # To save the pipeline for inference, we fit it on X_train.
    
    pipeline.fit(X_train)
    
    # Save the pipeline
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(output_dir, 'preprocessing_pipeline.pkl'))
    
    print(f"Preprocessing pipeline saved to {output_dir}")
    
    return X_train, X_test, y_train, y_test, pipeline

if __name__ == "__main__":
    data_path = "data/loan_data.csv"
    if os.path.exists(data_path):
        preprocess_and_save(data_path)
    else:
        print(f"Error: {data_path} not found. Run data_generator.py first.")
