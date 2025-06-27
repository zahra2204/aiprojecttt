import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    USE_SEABORN = True
except ImportError:
    USE_SEABORN = False
    print("Seaborn not available, using matplotlib for visualization")
try:
    from imblearn.over_sampling import SMOTE
    USE_SMOTE = True
except ImportError:
    USE_SMOTE = False
    print("SMOTE not available, using class weights for imbalance handling")
import joblib

class FraudDetectionSystem:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced' if not USE_SMOTE else None
        )
        self.scaler = StandardScaler()
        self.data = None
        
    def load_data(self, file_path):
        """Load and preprocess the dataset"""
        try:
            self.data = pd.read_csv(file_path)
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
            
    def show_dataset_comparison(self):
        """Display comparison statistics between legitimate and fraudulent transactions"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
            
        # Basic statistics
        legit = self.data[self.data['Class'] == 0]
        fraud = self.data[self.data['Class'] == 1]
        
        print("\n=== Dataset Overview ===")
        print(f"Total transactions: {len(self.data):,}")
        print(f"Legitimate transactions: {len(legit):,} ({len(legit)/len(self.data):.2%})")
        print(f"Fraudulent transactions: {len(fraud):,} ({len(fraud)/len(self.data):.2%})")
        
        # Amount comparison with more statistics
        print("\n=== Transaction Amount Analysis ===")
        print("Legitimate transactions:")
        print(f"  Mean amount: ${legit['Amount'].mean():.2f}")
        print(f"  Median amount: ${legit['Amount'].median():.2f}")
        print(f"  Standard deviation: ${legit['Amount'].std():.2f}")
        print(f"  Min amount: ${legit['Amount'].min():.2f}")
        print(f"  Max amount: ${legit['Amount'].max():.2f}")
        print(f"  Total volume: ${legit['Amount'].sum():,.2f}")
        
        print("\nFraudulent transactions:")
        print(f"  Mean amount: ${fraud['Amount'].mean():.2f}")
        print(f"  Median amount: ${fraud['Amount'].median():.2f}")
        print(f"  Standard deviation: ${fraud['Amount'].std():.2f}")
        print(f"  Min amount: ${fraud['Amount'].min():.2f}")
        print(f"  Max amount: ${fraud['Amount'].max():.2f}")
        print(f"  Total volume: ${fraud['Amount'].sum():,.2f}")
        
        # Time distribution comparison with more granular analysis
        print("\n=== Time Distribution Analysis ===")
        legit_hours = legit['Time'] // 3600
        fraud_hours = fraud['Time'] // 3600
        
        # Create a figure with 3 subplots
        plt.figure(figsize=(18, 6))
        
        # Subplot 1: Hourly distribution
        plt.subplot(1, 3, 1)
        plt.hist(legit_hours, bins=24, alpha=0.5, label='Legitimate', density=True)
        plt.hist(fraud_hours, bins=24, alpha=0.5, label='Fraudulent', density=True)
        plt.xlabel('Hour of Day')
        plt.ylabel('Normalized Frequency')
        plt.title('Transaction Distribution by Hour')
        plt.legend()
        
        # Subplot 2: Amount distribution
        plt.subplot(1, 3, 2)
        plt.hist(legit['Amount'], bins=50, alpha=0.5, label='Legitimate', density=True)
        plt.hist(fraud['Amount'], bins=50, alpha=0.5, label='Fraudulent', density=True)
        plt.xlabel('Transaction Amount ($)')
        plt.ylabel('Normalized Frequency')
        plt.title('Transaction Amount Distribution')
        plt.legend()
        plt.xlim(0, min(legit['Amount'].max(), fraud['Amount'].max()))
        
        # Feature comparison with statistical significance
        print("\n=== Feature Analysis ===")
        print("Top 5 most significant features (based on mean difference and standard deviation):")
        
        # Calculate feature statistics
        feature_stats = {}
        for col in self.data.columns:
            if col not in ['Time', 'Amount', 'Class']:
                legit_mean = legit[col].mean()
                fraud_mean = fraud[col].mean()
                legit_std = legit[col].std()
                fraud_std = fraud[col].std()
                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt((legit_std**2 + fraud_std**2) / 2)
                effect_size = abs(legit_mean - fraud_mean) / pooled_std
                feature_stats[col] = {
                    'effect_size': effect_size,
                    'legit_mean': legit_mean,
                    'fraud_mean': fraud_mean,
                    'legit_std': legit_std,
                    'fraud_std': fraud_std
                }
        
        # Sort by effect size
        sorted_features = sorted(feature_stats.items(), key=lambda x: x[1]['effect_size'], reverse=True)
        
        # Display top 5 features
        for feature, stats in sorted_features[:5]:
            print(f"\n{feature}:")
            print(f"  Effect size (Cohen's d): {stats['effect_size']:.4f}")
            print(f"  Legitimate: mean={stats['legit_mean']:.4f}, std={stats['legit_std']:.4f}")
            print(f"  Fraudulent: mean={stats['fraud_mean']:.4f}, std={stats['fraud_std']:.4f}")
        
        # Subplot 3: Feature comparison
        plt.subplot(1, 3, 3)
        features = [f[0] for f in sorted_features[:5]]
        effect_sizes = [f[1]['effect_size'] for f in sorted_features[:5]]
        
        plt.barh(features, effect_sizes)
        plt.xlabel("Effect Size (Cohen's d)")
        plt.title('Top 5 Most Significant Features')
        
        plt.tight_layout()
        plt.savefig('dataset_comparison.png', dpi=300, bbox_inches='tight')
        print("\nDetailed comparison visualization saved as 'dataset_comparison.png'")
        
        # Additional insights
        print("\n=== Additional Insights ===")
        print(f"Average time between transactions (legitimate): {legit['Time'].diff().mean():.2f} seconds")
        print(f"Average time between transactions (fraudulent): {fraud['Time'].diff().mean():.2f} seconds")
        print(f"Most common hour for legitimate transactions: {legit_hours.mode()[0]}:00")
        print(f"Most common hour for fraudulent transactions: {fraud_hours.mode()[0]}:00")
        
    def preprocess_data(self, data):
        """Preprocess the data for training"""
        # Drop any non-numeric columns for simplicity
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Handle missing values
        numeric_data = numeric_data.fillna(numeric_data.mean())
        
        # Separate features and target
        X = numeric_data.drop('Class', axis=1)
        y = numeric_data['Class']
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
        
    def handle_imbalance(self, X, y):
        """Handle class imbalance using SMOTE or class weights"""
        if USE_SMOTE:
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            return X_resampled, y_resampled
        else:
            # If SMOTE is not available, return original data
            # Class weights are already set in the model
            return X, y
        
    def train_model(self, X, y):
        """Train the fraud detection model"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.model.fit(X_train, y_train)
        return X_test, y_test
        
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model's performance"""
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        print("\nROC AUC Score:", roc_auc_score(y_test, y_prob))
        
        # Plot feature importance
        self.plot_feature_importance()
        
    def plot_feature_importance(self):
        """Plot feature importance"""
        feature_importance = pd.DataFrame({
            'feature': range(self.model.n_features_in_),
            'importance': self.model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        if USE_SEABORN:
            sns.barplot(x='importance', y='feature', data=feature_importance)
        else:
            plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        
    def save_model(self, model_path='fraud_detection_model.joblib'):
        """Save the trained model and scaler"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler
        }
        joblib.dump(model_data, model_path)
        print(f"Model and scaler saved to {model_path}")
        
    def load_model(self, model_path='fraud_detection_model.joblib'):
        """Load a trained model and scaler"""
        try:
            # Load the model data
            model_data = joblib.load(model_path)
            
            # Check if model_data is a dictionary
            if isinstance(model_data, dict):
                self.model = model_data['model']
                self.scaler = model_data['scaler']
            else:
                # If it's not a dictionary, assume it's just the model
                self.model = model_data
                # Initialize a new scaler
                self.scaler = StandardScaler()
                # Fit the scaler with some sample data
                sample_data = pd.read_csv('creditcard.csv')
                numeric_data = sample_data.select_dtypes(include=[np.number])
                if 'Class' in numeric_data.columns:
                    numeric_data = numeric_data.drop('Class', axis=1)
                self.scaler.fit(numeric_data)
            
            print(f"Model and scaler loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise e
        
    def predict(self, X):
        """Make predictions on new data"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled), self.model.predict_proba(X_scaled)[:, 1]

def main():
    # Initialize the fraud detection system
    fraud_system = FraudDetectionSystem()
    
    # Load and preprocess the data
    data = fraud_system.load_data('creditcard.csv')  # Replace with your dataset
    if data is not None:
        # Show dataset comparison
        fraud_system.show_dataset_comparison()
        
        # Preprocess the data
        X, y = fraud_system.preprocess_data(data)
        
        # Handle class imbalance
        X_resampled, y_resampled = fraud_system.handle_imbalance(X, y)
        
        # Train the model
        X_test, y_test = fraud_system.train_model(X_resampled, y_resampled)
        
        # Evaluate the model
        fraud_system.evaluate_model(X_test, y_test)
        
        # Save the model
        fraud_system.save_model()

if __name__ == "__main__":
    main() 