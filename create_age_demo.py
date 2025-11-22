import pandas as pd
import numpy as np

def generate_large_age_dataset():
    print("🔄 Generating 1,000 customer records...")
    np.random.seed(42)
    n_rows = 1000
    
    data = {
        'Age': np.random.randint(18, 80, n_rows),
        'Gender': np.random.choice(['Male', 'Female'], n_rows),
        'MonthlyUsageHours': np.random.randint(5, 200, n_rows),
        'NumTransactions': np.random.randint(1, 50, n_rows),
        'SubscriptionType': np.random.choice(['Basic', 'Gold', 'Premium'], n_rows),
        'Complaints': np.random.randint(0, 10, n_rows) # Complaints are random!
    }
    
    df = pd.DataFrame(data)
    
    # --- LOGIC INJECTION FOR CONSISTENCY ---
    # We add a 'Churn' label so you can use this for training if you want.
    # Logic: Young = Risky, Old = Safe.
    
    df['Churn_Prob'] = 0.5
    
    # The Age Factor
    df.loc[df['Age'] < 30, 'Churn_Prob'] += 0.4  # Young -> Risky
    df.loc[df['Age'] > 60, 'Churn_Prob'] -= 0.4  # Old -> Safe
    
    # Add a little noise so it's not too perfect
    df['Churn_Prob'] += np.random.normal(0, 0.1, n_rows)
    
    # Generate Labels
    df['Churn'] = np.random.binomial(1, df['Churn_Prob'].clip(0, 1))
    df.drop(columns=['Churn_Prob'], inplace=True)
    
    filename = 'large_age_demo_1000.csv'
    df.to_csv(filename, index=False)
    
    print(f"✅ Success! Created '{filename}' with 1,000 rows.")
    print("📊 Quick Stats:")
    print(f"- Young People Churn Rate: {df[df['Age'] < 30]['Churn'].mean():.1%}")
    print(f"- Seniors Churn Rate: {df[df['Age'] > 60]['Churn'].mean():.1%}")

if __name__ == "__main__":
    generate_large_age_dataset()