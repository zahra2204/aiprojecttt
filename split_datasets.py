import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def split_datasets():
    # Load the original datasets
    print("Loading original datasets...")
    legit_df = pd.read_csv('legitimate_transactions.csv')
    fraud_df = pd.read_csv('fraudulent_transactions.csv')
    
    # Add a timestamp column based on the Time feature
    current_time = datetime.now()
    legit_df['timestamp'] = pd.to_datetime(current_time - pd.to_timedelta(legit_df['Time'], unit='s'))
    fraud_df['timestamp'] = pd.to_datetime(current_time - pd.to_timedelta(fraud_df['Time'], unit='s'))
    
    # Split into current (last 30 days) and past datasets
    cutoff_date = current_time - timedelta(days=30)
    
    # Split legitimate transactions
    legit_current = legit_df[legit_df['timestamp'] >= cutoff_date]
    legit_past = legit_df[legit_df['timestamp'] < cutoff_date]
    
    # Split fraudulent transactions
    fraud_current = fraud_df[fraud_df['timestamp'] >= cutoff_date]
    fraud_past = fraud_df[fraud_df['timestamp'] < cutoff_date]
    
    # Save the split datasets
    print("\nSaving split datasets...")
    
    # Current datasets
    legit_current.to_csv('current_legitimate_transactions.csv', index=False)
    fraud_current.to_csv('current_fraudulent_transactions.csv', index=False)
    
    # Past datasets
    legit_past.to_csv('past_legitimate_transactions.csv', index=False)
    fraud_past.to_csv('past_fraudulent_transactions.csv', index=False)
    
    # Print statistics
    print("\n=== Dataset Statistics ===")
    print("Current Datasets (Last 30 days):")
    print(f"  Legitimate transactions: {len(legit_current):,}")
    print(f"  Fraudulent transactions: {len(fraud_current):,}")
    print(f"  Total current transactions: {len(legit_current) + len(fraud_current):,}")
    
    print("\nPast Datasets (Older than 30 days):")
    print(f"  Legitimate transactions: {len(legit_past):,}")
    print(f"  Fraudulent transactions: {len(fraud_past):,}")
    print(f"  Total past transactions: {len(legit_past) + len(fraud_past):,}")
    
    # Calculate fraud rates with error handling
    try:
        current_fraud_rate = len(fraud_current) / (len(legit_current) + len(fraud_current)) * 100
        print(f"\nCurrent fraud rate: {current_fraud_rate:.2f}%")
    except ZeroDivisionError:
        print("\nCurrent fraud rate: No transactions in current period")
    
    try:
        past_fraud_rate = len(fraud_past) / (len(legit_past) + len(fraud_past)) * 100
        print(f"Past fraud rate: {past_fraud_rate:.2f}%")
    except ZeroDivisionError:
        print("Past fraud rate: No transactions in past period")
    
    # Calculate average amounts with error handling
    print("\n=== Average Transaction Amounts ===")
    print("Current transactions:")
    try:
        print(f"  Legitimate: ${legit_current['Amount'].mean():.2f}")
    except:
        print("  Legitimate: No transactions")
    try:
        print(f"  Fraudulent: ${fraud_current['Amount'].mean():.2f}")
    except:
        print("  Fraudulent: No transactions")
    
    print("\nPast transactions:")
    try:
        print(f"  Legitimate: ${legit_past['Amount'].mean():.2f}")
    except:
        print("  Legitimate: No transactions")
    try:
        print(f"  Fraudulent: ${fraud_past['Amount'].mean():.2f}")
    except:
        print("  Fraudulent: No transactions")

if __name__ == "__main__":
    split_datasets() 