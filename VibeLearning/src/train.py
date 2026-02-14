import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from src.preprocessing import preprocess_and_save

def train_models():
    """Trains multiple models and selects the best one."""
    
    data_path = "data/loan_data.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    print("Preprocessing data...")
    X_train_raw, X_test_raw, y_train, y_test, pipeline = preprocess_and_save(data_path)
    
    # Transform data for training using the fitted pipeline
    X_train_processed = pipeline.transform(X_train_raw)
    X_test_processed = pipeline.transform(X_test_raw)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    print("\nTraining models...")
    for name, model in models.items():
        model.fit(X_train_processed, y_train)
        y_pred = model.predict(X_test_processed)
        y_prob = model.predict_proba(X_test_processed)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        print(f"\n{name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  ROC AUC:  {roc_auc:.4f}")
        
        if roc_auc > best_score:
            best_score = roc_auc
            best_model = model
            best_name = name
            
    print(f"\nBest Model: {best_name} with ROC AUC: {best_score:.4f}")
    
    # Save best model
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "best_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"Best model saved to {model_path}")
    
    # Also save the column names for inference to map correctly
    # Note: ColumnTransformer output order is transformers list order + remaining
    # Here, 'num' transformer was applied to numeric_features, and no other columns were passed through (since we dropped target)
    # If we had passthrough columns, we'd need to be careful.
    
    return best_model

if __name__ == "__main__":
    train_models()
