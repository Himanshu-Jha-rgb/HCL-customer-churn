"""
Customer Churn Prediction System
Complete ML solution with imbalanced data handling, evaluation metrics, and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ChurnPredictionModel:
    """Complete Customer Churn Prediction Pipeline"""
    
    def __init__(self, csv_path='customer_churn_data.csv'):
        """Initialize the model with data path"""
        self.csv_path = csv_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """Load and display basic information about the dataset"""
        print("="*70)
        print("STEP 1: DATA LOADING")
        print("="*70)
        
        self.df = pd.read_csv(self.csv_path)
        print(f"✓ Dataset loaded successfully!")
        print(f"  Shape: {self.df.shape}")
        print(f"\n📊 First few rows:")
        print(self.df.head())
        print(f"\n📈 Dataset Info:")
        print(self.df.info())
        print(f"\n📉 Statistical Summary:")
        print(self.df.describe())
        
        return self.df
    
    def check_class_imbalance(self):
        """Analyze and visualize class imbalance"""
        print("\n" + "="*70)
        print("STEP 2: CLASS IMBALANCE ANALYSIS")
        print("="*70)
        
        churn_counts = self.df['Churn'].value_counts()
        churn_percentages = self.df['Churn'].value_counts(normalize=True) * 100
        
        print(f"\n📊 Churn Distribution:")
        print(f"  No Churn (0): {churn_counts[0]} ({churn_percentages[0]:.2f}%)")
        print(f"  Churn (1):    {churn_counts[1]} ({churn_percentages[1]:.2f}%)")
        print(f"\n⚠️  Imbalance Ratio: {churn_counts[0]/churn_counts[1]:.2f}:1")
        
        # Visualize class distribution
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Count plot
        churn_counts.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
        axes[0].set_title('Customer Churn Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Churn Status')
        axes[0].set_ylabel('Count')
        axes[0].set_xticklabels(['No Churn', 'Churn'], rotation=0)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Pie chart
        axes[1].pie(churn_counts, labels=['No Churn', 'Churn'], autopct='%1.1f%%',
                   colors=['#2ecc71', '#e74c3c'], startangle=90)
        axes[1].set_title('Churn Rate Proportion', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('01_class_imbalance.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved: 01_class_imbalance.png")
        plt.show()
        
    def preprocess_data(self):
        """Clean and preprocess the data"""
        print("\n" + "="*70)
        print("STEP 3: DATA PREPROCESSING")
        print("="*70)
        
        # Check for missing values
        missing = self.df.isnull().sum()
        print(f"\n🔍 Missing Values (Before Handling):")
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("  No missing values found!")
        
        # Handle missing values
        print(f"\n🔧 Handling Missing Values:")
        
        # Fill numerical columns with median
        numerical_cols = ['Age', 'MonthlyUsageHours']
        for col in numerical_cols:
            if self.df[col].isnull().sum() > 0:
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                print(f"  ✓ {col}: Filled {missing[col]} missing values with median ({median_val:.2f})")
        
        # Verify no missing values remain
        remaining_missing = self.df.isnull().sum().sum()
        print(f"\n✓ Total missing values after handling: {remaining_missing}")
        
        # Clean Gender column (handle inconsistent entries)
        print(f"\n🧹 Cleaning Gender column:")
        print(f"  Original unique values: {self.df['Gender'].unique()}")
        # Standardize gender values
        self.df['Gender'] = self.df['Gender'].str.lower().str.strip()
        self.df['Gender'] = self.df['Gender'].map({'male': 'Male', 'm': 'Male', 'female': 'Female', 'f': 'Female'})
        print(f"  Cleaned unique values: {self.df['Gender'].unique()}")
        
        # Encode categorical variables
        categorical_cols = ['Gender', 'SubscriptionType']
        print(f"\n🔄 Encoding categorical variables: {categorical_cols}")
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col + '_Encoded'] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
            print(f"  ✓ {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        # Prepare features and target
        feature_cols = ['Age', 'Gender_Encoded', 'MonthlyUsageHours', 
                       'NumTransactions', 'SubscriptionType_Encoded', 'Complaints']
        
        X = self.df[feature_cols]
        y = self.df['Churn']
        
        print(f"\n📋 Features selected: {feature_cols}")
        print(f"🎯 Target variable: Churn")
        
        # Data quality check - handle outliers in Age
        age_outliers = (X['Age'] > 100).sum()
        if age_outliers > 0:
            print(f"\n⚠️  Found {age_outliers} age outliers (>100 years)")
            print(f"  Capping ages to reasonable range (18-100)")
            X['Age'] = X['Age'].clip(upper=100)
        
        # Final verification - no NaN values
        if X.isnull().sum().sum() > 0:
            print(f"\n❌ Warning: Still have missing values in features!")
            print(X.isnull().sum()[X.isnull().sum() > 0])
        else:
            print(f"\n✓ All features verified: No missing values")
        
        return X, y
    
    def split_and_scale_data(self, X, y):
        """Split data and apply scaling"""
        print("\n" + "="*70)
        print("STEP 4: TRAIN-TEST SPLIT & SCALING")
        print("="*70)
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📊 Data Split:")
        print(f"  Training set: {self.X_train.shape[0]} samples")
        print(f"  Testing set:  {self.X_test.shape[0]} samples")
        print(f"  Train churn rate: {self.y_train.mean():.2%}")
        print(f"  Test churn rate:  {self.y_test.mean():.2%}")
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"\n✓ Feature scaling applied (StandardScaler)")
        
        return self.X_train_scaled, self.X_test_scaled
    
    def handle_imbalance_smote(self, X_train, y_train):
        """Handle class imbalance using SMOTE"""
        print("\n" + "="*70)
        print("STEP 5: HANDLING IMBALANCED DATA (SMOTE)")
        print("="*70)
        
        print(f"\n📊 Before SMOTE:")
        print(f"  Class 0: {(y_train == 0).sum()} samples")
        print(f"  Class 1: {(y_train == 1).sum()} samples")
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"\n📊 After SMOTE:")
        print(f"  Class 0: {(y_resampled == 0).sum()} samples")
        print(f"  Class 1: {(y_resampled == 1).sum()} samples")
        print(f"\n✓ Dataset balanced successfully!")
        
        # Visualize before and after
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Before SMOTE
        pd.Series(y_train).value_counts().plot(kind='bar', ax=axes[0], 
                                                color=['#3498db', '#e74c3c'])
        axes[0].set_title('Before SMOTE', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Churn Status')
        axes[0].set_ylabel('Count')
        axes[0].set_xticklabels(['No Churn', 'Churn'], rotation=0)
        
        # After SMOTE
        pd.Series(y_resampled).value_counts().plot(kind='bar', ax=axes[1],
                                                     color=['#3498db', '#e74c3c'])
        axes[1].set_title('After SMOTE', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Churn Status')
        axes[1].set_ylabel('Count')
        axes[1].set_xticklabels(['No Churn', 'Churn'], rotation=0)
        
        plt.tight_layout()
        plt.savefig('02_smote_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved: 02_smote_comparison.png")
        plt.show()
        
        return X_resampled, y_resampled
    
    def train_model(self, X_train, y_train):
        """Train the classification model"""
        print("\n" + "="*70)
        print("STEP 6: MODEL TRAINING")
        print("="*70)
        
        print(f"\n🤖 Training Random Forest Classifier...")
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        print(f"✓ Model trained successfully!")
        print(f"  Algorithm: Random Forest")
        print(f"  Trees: 100")
        print(f"  Max Depth: 10")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation"""
        print("\n" + "="*70)
        print("STEP 7: MODEL EVALUATION")
        print("="*70)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n📊 EVALUATION METRICS:")
        print(f"  {'='*50}")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
        print(f"  ROC-AUC:   {roc_auc:.4f} ({roc_auc*100:.2f}%)")
        print(f"  {'='*50}")
        
        # Detailed classification report
        print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['No Churn', 'Churn']))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def plot_confusion_matrix(self, y_test, y_pred):
        """Plot confusion matrix with detailed explanation"""
        print("\n" + "="*70)
        print("STEP 8: CONFUSION MATRIX")
        print("="*70)
        
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\n🔢 Confusion Matrix Values:")
        print(f"  True Negatives (TN):  {cm[0,0]} - Correctly predicted No Churn")
        print(f"  False Positives (FP): {cm[0,1]} - Incorrectly predicted Churn")
        print(f"  False Negatives (FN): {cm[1,0]} - Missed actual Churn")
        print(f"  True Positives (TP):  {cm[1,1]} - Correctly predicted Churn")
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'],
                   cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix\n', fontsize=16, fontweight='bold')
        plt.ylabel('Actual Class', fontsize=12)
        plt.xlabel('Predicted Class', fontsize=12)
        
        # Add text annotations for clarity
        plt.text(0.5, -0.15, f'TN={cm[0,0]}', ha='center', transform=plt.gca().transAxes)
        plt.text(1.5, -0.15, f'FP={cm[0,1]}', ha='center', transform=plt.gca().transAxes)
        plt.text(0.5, -0.22, f'FN={cm[1,0]}', ha='center', transform=plt.gca().transAxes)
        plt.text(1.5, -0.22, f'TP={cm[1,1]}', ha='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.savefig('03_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved: 03_confusion_matrix.png")
        plt.show()
        
        return cm
    
    def plot_roc_curve(self, y_test, y_pred_proba, roc_auc):
        """Plot ROC curve with explanation"""
        print("\n" + "="*70)
        print("STEP 9: ROC CURVE ANALYSIS")
        print("="*70)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        
        print(f"\n📈 ROC Curve Analysis:")
        print(f"  ROC-AUC Score: {roc_auc:.4f}")
        
        if roc_auc >= 0.9:
            interpretation = "Excellent"
        elif roc_auc >= 0.8:
            interpretation = "Very Good"
        elif roc_auc >= 0.7:
            interpretation = "Good"
        elif roc_auc >= 0.6:
            interpretation = "Fair"
        else:
            interpretation = "Poor"
        
        print(f"  Interpretation: {interpretation} discrimination ability")
        print(f"\n💡 Explanation:")
        print(f"  The ROC curve shows the trade-off between True Positive Rate")
        print(f"  and False Positive Rate at different classification thresholds.")
        print(f"  AUC = {roc_auc:.3f} means the model has {interpretation.lower()} predictive power.")
        
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        
        plt.plot(fpr, tpr, color='darkorange', lw=3, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier (AUC = 0.5)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve\n', 
                 fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('04_roc_curve.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved: 04_roc_curve.png")
        plt.show()
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        print("\n" + "="*70)
        print("STEP 10: FEATURE IMPORTANCE")
        print("="*70)
        
        # Get feature importance
        feature_names = ['Age', 'Gender', 'MonthlyUsageHours', 
                        'NumTransactions', 'SubscriptionType', 'Complaints']
        importances = self.model.feature_importances_
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        print(f"\n🎯 Feature Importance Ranking:")
        for i, idx in enumerate(indices, 1):
            print(f"  {i}. {feature_names[idx]}: {importances[idx]:.4f}")
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices], color='steelblue')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], 
                  rotation=45, ha='right')
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importance Score', fontsize=12)
        plt.title('Feature Importance for Churn Prediction\n', 
                 fontsize=16, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('05_feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved: 05_feature_importance.png")
        plt.show()
    
    def run_complete_pipeline(self):
        """Execute the complete ML pipeline"""
        print("\n" + "🎯"*35)
        print("CUSTOMER CHURN PREDICTION - COMPLETE ML PIPELINE")
        print("🎯"*35 + "\n")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Check imbalance
        self.check_class_imbalance()
        
        # Step 3: Preprocess
        X, y = self.preprocess_data()
        
        # Step 4: Split and scale
        X_train_scaled, X_test_scaled = self.split_and_scale_data(X, y)
        
        # Step 5: Handle imbalance with SMOTE
        X_resampled, y_resampled = self.handle_imbalance_smote(
            X_train_scaled, self.y_train
        )
        
        # Step 6: Train model
        self.train_model(X_resampled, y_resampled)
        
        # Step 7: Evaluate
        results = self.evaluate_model(X_test_scaled, self.y_test)
        
        # Step 8: Confusion Matrix
        self.plot_confusion_matrix(self.y_test, results['y_pred'])
        
        # Step 9: ROC Curve
        self.plot_roc_curve(self.y_test, results['y_pred_proba'], 
                           results['roc_auc'])
        
        # Step 10: Feature Importance
        self.plot_feature_importance()
        
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\n📁 Generated Files:")
        print(f"  • 01_class_imbalance.png")
        print(f"  • 02_smote_comparison.png")
        print(f"  • 03_confusion_matrix.png")
        print(f"  • 04_roc_curve.png")
        print(f"  • 05_feature_importance.png")
        print(f"\n🎉 Ready for your hackathon presentation!")

"""
BONUS FEATURES - Add this to the END of your churn_prediction.py file
Just before the "if __name__ == '__main__':" section
"""

def compare_algorithms_bonus(self, X_train, X_test, y_train, y_test):
    """Compare multiple algorithms - BONUS FEATURE"""
    print("\n" + "="*70)
    print("BONUS: COMPARING MULTIPLE ALGORITHMS")
    print("="*70)
    
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    
    algorithms = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, 
                                               random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, 
                                                        random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, 
                                                 class_weight='balanced', 
                                                 random_state=42)
    }
    
    print(f"\n🤖 Training {len(algorithms)} algorithms...\n")
    
    results_data = []
    
    for name, model in algorithms.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results_data.append({
            'Algorithm': name,
            'Accuracy': f'{accuracy:.3f}',
            'ROC-AUC': f'{roc_auc:.3f}'
        })
        
        print(f"  ✓ {name}: Accuracy={accuracy:.3f}, ROC-AUC={roc_auc:.3f}")
    
    results_df = pd.DataFrame(results_data)
    print(f"\n📊 COMPARISON TABLE:")
    print(results_df.to_string(index=False))
    
    return results_df


# Add this method to your ChurnPredictionModel class
# Then modify run_complete_pipeline() to call it at the end:

def run_complete_pipeline_with_bonus(self):
    """Execute the complete ML pipeline WITH BONUS"""
    print("\n" + "🎯"*35)
    print("CUSTOMER CHURN PREDICTION - COMPLETE ML PIPELINE")
    print("🎯"*35 + "\n")
    
    # All your existing steps
    self.load_data()
    self.check_class_imbalance()
    X, y = self.preprocess_data()
    X_train_scaled, X_test_scaled = self.split_and_scale_data(X, y)
    X_resampled, y_resampled = self.handle_imbalance_smote(X_train_scaled, self.y_train)
    self.train_model(X_resampled, y_resampled)
    results = self.evaluate_model(X_test_scaled, self.y_test)
    self.plot_confusion_matrix(self.y_test, results['y_pred'])
    self.plot_roc_curve(self.y_test, results['y_pred_proba'], results['roc_auc'])
    self.plot_feature_importance()
    
    # NEW: Add algorithm comparison
    self.compare_algorithms_bonus(X_resampled, X_test_scaled, 
                                  y_resampled, self.y_test)
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n🎉 Ready for your hackathon presentation!")



if __name__ == "__main__":
    # CHANGE THE FILENAME HERE 👇
    model = ChurnPredictionModel('customer_churn_data_enhanced.csv')
    model.run_complete_pipeline()