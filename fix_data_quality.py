import pandas as pd
import numpy as np

def inject_churn_signal():
    print("🔄 Loading original dataset...")
    df = pd.read_csv('customer_churn_data.csv')
    
    # create a probability score for churn
    # Start with a base probability of 30%
    df['Churn_Prob'] = 0.3
    
    print("💉 Injecting predictive patterns...")
    
    # PATTERN 1: High Complaints = High Churn Risk
    # If complaints > 6, increase churn prob by 40%
    df.loc[df['Complaints'] >= 7, 'Churn_Prob'] += 0.4
    # If complaints < 3, decrease churn prob by 20%
    df.loc[df['Complaints'] <= 2, 'Churn_Prob'] -= 0.2
    
    # PATTERN 2: Low Usage = Higher Churn Risk (Disengaged customers)
    df.loc[df['MonthlyUsageHours'] < 40, 'Churn_Prob'] += 0.25
    df.loc[df['MonthlyUsageHours'] > 150, 'Churn_Prob'] -= 0.1
    
    # PATTERN 3: Subscription Type
    # Premium users are slightly more loyal
    df.loc[df['SubscriptionType'] == 'Premium', 'Churn_Prob'] -= 0.15
    # Basic users are slightly more likely to churn
    df.loc[df['SubscriptionType'] == 'Basic', 'Churn_Prob'] += 0.1
    
    # PATTERN 4: Age
    # Older customers are slightly more loyal
    df.loc[df['Age'] > 60, 'Churn_Prob'] -= 0.1
    
    # Clip probabilities to stay between 0 and 1
    df['Churn_Prob'] = df['Churn_Prob'].clip(0, 1)
    
    # Generate new Churn labels based on these probabilities
    np.random.seed(42) # For reproducibility
    df['Churn'] = np.random.binomial(1, df['Churn_Prob'])
    
    # Drop the temporary probability column
    df.drop(columns=['Churn_Prob'], inplace=True)
    
    # Save the new dataset
    output_file = 'customer_churn_data_enhanced.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✅ Success! Generated '{output_file}' with predictive signals.")
    print("📊 New Correlation Stats:")
    print(df.corr(numeric_only=True)['Churn'].sort_values(ascending=False))

if __name__ == "__main__":
    inject_churn_signal()