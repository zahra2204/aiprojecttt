import pandas as pd
import numpy as np
from fraud_detection import FraudDetectionSystem
import matplotlib.pyplot as plt
from datetime import datetime

def analyze_current_fraud():
    # Initialize fraud detection system
    fraud_system = FraudDetectionSystem()
    try:
        fraud_system.load_model('fraud_detection_model.joblib')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    # Load current datasets
    print("\n=== Loading Current Transactions ===")
    legit_df = pd.read_csv('current_legitimate_transactions.csv')
    fraud_df = pd.read_csv('current_fraudulent_transactions.csv')
    
    # Basic statistics
    print("\n=== Current Dataset Statistics ===")
    print(f"Total legitimate transactions: {len(legit_df):,}")
    print(f"Total fraudulent transactions: {len(fraud_df):,}")
    print(f"Fraud rate: {(len(fraud_df)/(len(legit_df) + len(fraud_df))*100):.2f}%")
    
    # Combine datasets for testing
    test_df = pd.concat([legit_df, fraud_df])
    
    # Make predictions
    print("\n=== Real-time Fraud Detection Analysis ===")
    numeric_data = test_df.select_dtypes(include=[np.number])
    if 'Class' in numeric_data.columns:
        numeric_data = numeric_data.drop('Class', axis=1)
    
    predictions, probabilities = fraud_system.predict(numeric_data)
    
    # Calculate detection metrics
    true_positives = sum((predictions == 1) & (test_df['Class'] == 1))
    false_positives = sum((predictions == 1) & (test_df['Class'] == 0))
    true_negatives = sum((predictions == 0) & (test_df['Class'] == 0))
    false_negatives = sum((predictions == 0) & (test_df['Class'] == 1))
    
    print("\n=== Protection Performance ===")
    print(f"Successfully blocked fraudulent transactions: {true_positives:,}")
    print(f"Correctly approved legitimate transactions: {true_negatives:,}")
    print(f"False alarms (legitimate transactions flagged): {false_positives:,}")
    print(f"Missed fraudulent transactions: {false_negatives:,}")
    
    # Calculate protection metrics
    detection_rate = true_positives / len(fraud_df) * 100
    accuracy = (true_positives + true_negatives) / len(test_df) * 100
    false_alarm_rate = false_positives / len(legit_df) * 100
    
    print("\n=== Protection Effectiveness ===")
    print(f"Fraud detection success rate: {detection_rate:.2f}%")
    print(f"Overall accuracy: {accuracy:.2f}%")
    print(f"False alarm rate: {false_alarm_rate:.2f}%")
    
    # Amount analysis
    detected_fraud_amount = test_df.loc[(predictions == 1) & (test_df['Class'] == 1), 'Amount'].sum()
    total_fraud_amount = test_df.loc[test_df['Class'] == 1, 'Amount'].sum()
    
    print("\n=== Financial Protection ===")
    print(f"Total amount of attempted fraud: ${total_fraud_amount:,.2f}")
    print(f"Amount protected from fraud: ${detected_fraud_amount:,.2f}")
    print(f"Protection rate: {(detected_fraud_amount/total_fraud_amount*100):.2f}%")
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Protection Performance
    plt.subplot(1, 3, 1)
    performance = [true_positives, false_positives, true_negatives, false_negatives]
    labels = ['Blocked Fraud', 'False Alarms', 'Approved Legitimate', 'Missed Fraud']
    colors = ['green', 'orange', 'blue', 'red']
    plt.pie(performance, labels=labels, colors=colors, autopct='%1.1f%%')
    plt.title('Fraud Detection Performance')
    
    # Plot 2: Amount Distribution
    plt.subplot(1, 3, 2)
    plt.hist(test_df.loc[test_df['Class'] == 0, 'Amount'], 
             bins=50, alpha=0.5, label='Legitimate', density=True)
    plt.hist(test_df.loc[test_df['Class'] == 1, 'Amount'], 
             bins=50, alpha=0.5, label='Fraudulent', density=True)
    plt.xlabel('Transaction Amount ($)')
    plt.ylabel('Frequency')
    plt.title('Transaction Amount Distribution')
    plt.legend()
    
    # Plot 3: Time Distribution
    plt.subplot(1, 3, 3)
    hours = test_df['Time'] // 3600
    fraud_hours = hours[test_df['Class'] == 1]
    legit_hours = hours[test_df['Class'] == 0]
    plt.hist(legit_hours, bins=24, alpha=0.5, label='Legitimate', density=True)
    plt.hist(fraud_hours, bins=24, alpha=0.5, label='Fraudulent', density=True)
    plt.xlabel('Hour of Day')
    plt.ylabel('Frequency')
    plt.title('Transaction Time Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('current_fraud_analysis.png', dpi=300, bbox_inches='tight')
    print("\nCurrent fraud analysis visualization saved as 'current_fraud_analysis.png'")

if __name__ == "__main__":
    analyze_current_fraud() 