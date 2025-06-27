import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from datetime import datetime, timedelta
import random

def generate_sample_data(n_samples=10000, fraud_ratio=0.01):
    """
    Generate a sample credit card transaction dataset
    
    Parameters:
    -----------
    n_samples : int
        Total number of transactions to generate
    fraud_ratio : float
        Ratio of fraudulent transactions (0.01 = 1%)
    
    Returns:
    --------
    pandas.DataFrame
        Generated dataset with transaction features
    """
    # Generate base features using make_classification
    X, y = make_classification(
        n_samples=n_samples,
        n_features=28,  # V1-V28 features like in real credit card data
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=2,
        weights=[1-fraud_ratio, fraud_ratio],
        random_state=42
    )
    
    # Create DataFrame with V1-V28 features
    df = pd.DataFrame(X, columns=[f'V{i}' for i in range(1, 29)])
    
    # Add Amount feature (skewed distribution)
    df['Amount'] = np.random.exponential(scale=100, size=n_samples)
    
    # Add Time feature (in seconds from start of day)
    start_time = datetime(2023, 1, 1)
    times = [start_time + timedelta(seconds=int(random.uniform(0, 86400))) 
             for _ in range(n_samples)]
    df['Time'] = [t.hour * 3600 + t.minute * 60 + t.second for t in times]
    
    # Add Class feature (0 for legitimate, 1 for fraud)
    df['Class'] = y
    
    # Add some realistic patterns to fraudulent transactions
    fraud_indices = df[df['Class'] == 1].index
    df.loc[fraud_indices, 'Amount'] *= 2  # Fraudulent transactions tend to be larger
    df.loc[fraud_indices, 'V1'] *= 1.5    # Modify some features for fraud
    df.loc[fraud_indices, 'V2'] *= 1.3
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def main():
    # Generate sample data
    print("Generating sample dataset...")
    df = generate_sample_data(n_samples=10000, fraud_ratio=0.01)
    
    # Save to CSV
    output_file = 'creditcard.csv'
    df.to_csv(output_file, index=False)
    print(f"Sample dataset saved to {output_file}")
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total transactions: {len(df)}")
    print(f"Legitimate transactions: {len(df[df['Class'] == 0])}")
    print(f"Fraudulent transactions: {len(df[df['Class'] == 1])}")
    print(f"Fraud ratio: {len(df[df['Class'] == 1])/len(df):.2%}")
    print("\nFeature statistics:")
    print(df.describe())

if __name__ == "__main__":
    main() 