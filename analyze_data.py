import pandas as pd
import numpy as np
from fraud_detection import FraudDetectionSystem
import matplotlib.pyplot as plt

def analyze_transactions():
    # Initialize fraud detection system
    fraud_system = FraudDetectionSystem()
    try:
        fraud_system.load_model('fraud_detection_model.joblib')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    # Load and analyze legitimate transactions
    print("\n=== Analyzing Legitimate Transactions ===")
    legit_df = pd.read_csv('legitimate_transactions.csv')
    print(f"Total legitimate transactions: {len(legit_df):,}")
    
    # Load and analyze fraudulent transactions
    print("\n=== Analyzing Fraudulent Transactions ===")
    fraud_df = pd.read_csv('fraudulent_transactions.csv')
    print(f"Total fraudulent transactions: {len(fraud_df):,}")
    
    # Combine datasets for testing
    test_df = pd.concat([legit_df, fraud_df])
    
    # Make predictions
    print("\n=== Fraud Detection Analysis ===")
    numeric_data = test_df.select_dtypes(include=[np.number])
    if 'Class' in numeric_data.columns:
        numeric_data = numeric_data.drop('Class', axis=1)
    
    predictions, probabilities = fraud_system.predict(numeric_data)
    
    # Calculate detection metrics
    true_positives = sum((predictions == 1) & (test_df['Class'] == 1))
    false_positives = sum((predictions == 1) & (test_df['Class'] == 0))
    true_negatives = sum((predictions == 0) & (test_df['Class'] == 0))
    false_negatives = sum((predictions == 0) & (test_df['Class'] == 1))
    
    print("\n=== Fraud Detection Performance ===")
    print(f"Correctly identified fraudulent transactions: {true_positives:,}")
    print(f"Correctly identified legitimate transactions: {true_negatives:,}")
    print(f"False alarms (legitimate marked as fraud): {false_positives:,}")
    print(f"Missed fraudulent transactions: {false_negatives:,}")
    
    # Calculate protection metrics
    detection_rate = true_positives / len(fraud_df) * 100
    false_alarm_rate = false_positives / len(legit_df) * 100
    
    print("\n=== Protection Statistics ===")
    print(f"Fraud detection rate: {detection_rate:.2f}%")
    print(f"False alarm rate: {false_alarm_rate:.2f}%")
    
    # Amount saved from fraud
    detected_fraud_amount = test_df.loc[(predictions == 1) & (test_df['Class'] == 1), 'Amount'].sum()
    print(f"\nAmount protected from fraud: ${detected_fraud_amount:,.2f}")
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Detection Performance
    plt.subplot(1, 2, 1)
    performance = [true_positives, false_positives, true_negatives, false_negatives]
    labels = ['True Positives', 'False Positives', 'True Negatives', 'False Negatives']
    colors = ['green', 'red', 'blue', 'orange']
    plt.pie(performance, labels=labels, colors=colors, autopct='%1.1f%%')
    plt.title('Fraud Detection Performance')
    
    # Plot 2: Amount Distribution
    plt.subplot(1, 2, 2)
    plt.hist(test_df.loc[test_df['Class'] == 1, 'Amount'], 
             bins=50, alpha=0.5, label='Actual Fraud', density=True)
    plt.hist(test_df.loc[predictions == 1, 'Amount'], 
             bins=50, alpha=0.5, label='Detected Fraud', density=True)
    plt.xlabel('Transaction Amount ($)')
    plt.ylabel('Normalized Frequency')
    plt.title('Fraudulent Transaction Amount Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('fraud_protection_analysis.png')
    print("\nProtection analysis visualization saved as 'fraud_protection_analysis.png'")

if __name__ == "__main__":
    analyze_transactions() 