import pandas as pd
import numpy as np
import os

def generate_loan_data(n_samples=10000, random_state=42):
    """Generates synthetic loan data for default prediction."""
    np.random.seed(random_state)
    
    # Generate features
    age = np.random.randint(21, 70, n_samples)
    income = np.random.normal(50000, 15000, n_samples).astype(int)
    income = np.maximum(income, 15000) # Ensure no negative/too low income
    
    loan_amount = np.random.randint(1000, 50000, n_samples)
    credit_score = np.random.randint(300, 850, n_samples)
    years_employed = np.random.randint(0, 40, n_samples)
    num_credit_lines = np.random.randint(0, 15, n_samples)
    interest_rate = np.random.uniform(3.5, 25.0, n_samples)
    loan_term = np.random.choice([36, 60], n_samples)
    dti_ratio = np.random.uniform(0, 50, n_samples) # Debt-to-Income ratio
    
    # Introduce some correlation for the target variable 'Default'
    # Default probability increases with lower credit score, higher DTI, higher loan amount relative to income
    
    logit = (
        -0.02 * (credit_score - 600) + 
        0.05 * dti_ratio + 
        0.0001 * (loan_amount - 15000) - 
        0.00005 * (income - 50000) +
        0.1 * (interest_rate - 10)
    )
    
    default_prob = 1 / (1 + np.exp(-logit))
    default = np.random.binomial(1, default_prob)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Age': age,
        'Income': income,
        'LoanAmount': loan_amount,
        'CreditScore': credit_score,
        'YearsEmployed': years_employed,
        'NumCreditLines': num_credit_lines,
        'InterestRate': interest_rate,
        'LoanTerm': loan_term,
        'DTIRatio': dti_ratio,
        'Default': default
    })
    
    return df

if __name__ == "__main__":
    print("Generating synthetic loan data...")
    df = generate_loan_data()
    
    output_path = os.path.join("data", "loan_data.csv")
    df.to_csv(output_path, index=False)
    
    print(f"Data generated and saved to {output_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Default rate: {df['Default'].mean():.2%}")
