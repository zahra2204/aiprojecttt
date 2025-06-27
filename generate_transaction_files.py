import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from sklearn.datasets import make_classification

def generate_transactions(n_samples, fraud_ratio, is_fraudulent=False):
    """
    Generate transaction data
    
    Parameters:
    -----------
    n_samples : int
        Number of transactions to generate
    fraud_ratio : float
        Ratio of fraudulent transactions
    is_fraudulent : bool
        Whether to generate fraudulent transactions
    """
    # Generate base features
    X, y = make_classification(
        n_samples=n_samples,
        n_features=28,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=2,
        weights=[1-fraud_ratio, fraud_ratio],
        random_state=42
    )
    
    # Create DataFrame with V1-V28 features
    df = pd.DataFrame(X, columns=[f'V{i}' for i in range(1, 29)])
    
    # Add Amount feature (different distributions for fraud/non-fraud)
    if is_fraudulent:
        # Fraudulent transactions tend to be larger
        df['Amount'] = np.random.exponential(scale=200, size=n_samples)
    else:
        # Legitimate transactions follow a more normal distribution
        df['Amount'] = np.random.normal(loc=100, scale=50, size=n_samples)
        df['Amount'] = np.abs(df['Amount'])  # Ensure positive amounts
    
    # Add Time feature
    start_time = datetime(2023, 1, 1)
    times = [start_time + timedelta(seconds=int(random.uniform(0, 86400))) 
             for _ in range(n_samples)]
    df['Time'] = [t.hour * 3600 + t.minute * 60 + t.second for t in times]
    
    # Add Class feature
    df['Class'] = 1 if is_fraudulent else 0
    
    # Add some realistic patterns
    if is_fraudulent:
        # Modify features for fraudulent transactions
        df['V1'] *= 1.5
        df['V2'] *= 1.3
        df['V3'] *= 1.2
        # Add some unusual patterns
        df['V4'] = np.random.uniform(-5, 5, n_samples)
        df['V5'] = np.random.uniform(-3, 3, n_samples)
    
    return df

def main():
    # Generate legitimate transactions
    print("Generating legitimate transactions...")
    legit_df = generate_transactions(n_samples=9500, fraud_ratio=0.01, is_fraudulent=False)
    legit_df.to_csv('legitimate_transactions.csv', index=False)
    print(f"Generated {len(legit_df)} legitimate transactions")
    
    # Generate fraudulent transactions
    print("\nGenerating fraudulent transactions...")
    fraud_df = generate_transactions(n_samples=500, fraud_ratio=0.99, is_fraudulent=True)
    fraud_df.to_csv('fraudulent_transactions.csv', index=False)
    print(f"Generated {len(fraud_df)} fraudulent transactions")
    
    # Print some statistics
    print("\nTransaction Statistics:")
    print("\nLegitimate Transactions:")
    print(f"Average Amount: ${legit_df['Amount'].mean():.2f}")
    print(f"Min Amount: ${legit_df['Amount'].min():.2f}")
    print(f"Max Amount: ${legit_df['Amount'].max():.2f}")
    
    print("\nFraudulent Transactions:")
    print(f"Average Amount: ${fraud_df['Amount'].mean():.2f}")
    print(f"Min Amount: ${fraud_df['Amount'].min():.2f}")
    print(f"Max Amount: ${fraud_df['Amount'].max():.2f}")

if __name__ == "__main__":
    main() 