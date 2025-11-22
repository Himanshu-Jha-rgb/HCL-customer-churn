import pandas as pd
import numpy as np

def generate_high_accuracy_data():
    print("Loading dataset...")
    df = pd.read_csv('customer_churn_data.csv')
    
    # Initialize probability
    np.random.seed(42)
    df['Churn_Prob'] = 0.5 

    # --- RULE 1: COMPLAINTS ARE KING ---
    # Low Complaints (0-3) = Almost guaranteed to STAY
    df.loc[df['Complaints'] <= 3, 'Churn_Prob'] = 0.05
    
    # High Complaints (7-10) = Almost guaranteed to CHURN
    df.loc[df['Complaints'] >= 7, 'Churn_Prob'] = 0.95

    # --- RULE 2: THE TIE-BREAKER (For middle complaints 4-6) ---
    # If a customer has "average" complaints, we look at Usage to decide.
    middle_mask = (df['Complaints'] > 3) & (df['Complaints'] < 7)

    # High Usage (> 100 hours) means they are addicted/dependent -> THEY STAY
    df.loc[middle_mask & (df['MonthlyUsageHours'] >= 100), 'Churn_Prob'] = 0.15
    
    # Low Usage (< 100 hours) means they are disengaged -> THEY CHURN
    df.loc[middle_mask & (df['MonthlyUsageHours'] < 100), 'Churn_Prob'] = 0.85
    
    # --- Final Generation ---
    # Convert probabilities to 0/1 labels
    df['Churn'] = np.random.binomial(1, df['Churn_Prob'])
    
    # Cleanup
    df.drop(columns=['Churn_Prob'], inplace=True)
    output_file = 'customer_churn_data_final.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✅ Created '{output_file}'")
    print("Expected Accuracy: ~88-90%")
    print("Expected ROC AUC: ~0.92+")

if __name__ == "__main__":
    generate_high_accuracy_data()