import pandas as pd
import joblib
import os
import argparse

def load_artifacts(model_dir="models"):
    """Loads the trained model and preprocessing pipeline."""
    model_path = os.path.join(model_dir, "best_model.pkl")
    pipeline_path = os.path.join(model_dir, "preprocessing_pipeline.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(pipeline_path):
        raise FileNotFoundError("Model or pipeline not found. Run train.py first.")
        
    model = joblib.load(model_path)
    pipeline = joblib.load(pipeline_path)
    
    return model, pipeline

def predict(input_data):
    """Makes a prediction for the input data."""
    model, pipeline = load_artifacts()
    
    # Ensure input is a DataFrame with correct columns
    # We expect a dictionary or list of dictionaries
    if isinstance(input_data, dict):
        input_data = [input_data]
        
    df = pd.DataFrame(input_data)
    
    # Function to ensure columns match expected numeric features (and others if we had them)
    # The pipeline handles scaling, so we just need to pass the raw data in the right structure
    
    try:
        data_processed = pipeline.transform(df)
        prediction = model.predict(data_processed)
        probability = model.predict_proba(data_processed)[:, 1]
        
        return prediction, probability
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

if __name__ == "__main__":
    # Example usage
    sample_input = {
        'Age': 30,
        'Income': 45000,
        'LoanAmount': 10000,
        'CreditScore': 650,
        'YearsEmployed': 5,
        'NumCreditLines': 2,
        'InterestRate': 12.5,
        'LoanTerm': 36,
        'DTIRatio': 15.0
    }
    
    print("Running inference on sample data:")
    print(sample_input)
    
    pred, prob = predict(sample_input)
    
    if pred is not None:
        result = "Default" if pred[0] == 1 else "Repay"
        print(f"\nPrediction: {result}")
        print(f"Probability of Default: {prob[0]:.2%}")
