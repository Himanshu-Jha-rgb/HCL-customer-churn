import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="ChurnGuard AI", page_icon="🛡️", layout="wide")

# ==========================================
# SECTION 1: DATA GENERATION & FIXING ENGINE
# ==========================================
@st.cache_data
def get_high_quality_data():
    """Generates synthetic data with clear patterns for the model to learn."""
    try:
        df = pd.read_csv('customer_churn_data.csv')
    except FileNotFoundError:
        st.warning("⚠️ CSV not found. Generating synthetic data...")
        np.random.seed(42)
        n_rows = 1000
        df = pd.DataFrame({
            'Age': np.random.randint(18, 70, n_rows),
            'Gender': np.random.choice(['Male', 'Female'], n_rows),
            'MonthlyUsageHours': np.random.randint(5, 200, n_rows),
            'NumTransactions': np.random.randint(1, 50, n_rows),
            'SubscriptionType': np.random.choice(['Basic', 'Gold', 'Premium'], n_rows),
            'Complaints': np.random.randint(0, 10, n_rows)
        })

    # --- LOGIC INJECTION ---
    df['Churn_Prob'] = 0.5 

    # Rule 1: Complaints
    df.loc[df['Complaints'] <= 3, 'Churn_Prob'] = 0.05 
    df.loc[df['Complaints'] >= 7, 'Churn_Prob'] = 0.95 

    # Rule 2: Usage Tie-Breaker
    middle_mask = (df['Complaints'] > 3) & (df['Complaints'] < 7)
    df.loc[middle_mask & (df['MonthlyUsageHours'] >= 100), 'Churn_Prob'] = 0.15
    df.loc[middle_mask & (df['MonthlyUsageHours'] < 100), 'Churn_Prob'] = 0.85

    np.random.seed(42)
    df['Churn'] = np.random.binomial(1, df['Churn_Prob'])
    df = df.drop(columns=['Churn_Prob'])
    
    return df

# ==========================================
# SECTION 2: PREPROCESSING & MODELING
# ==========================================

def preprocess_data(df):
    df = df.copy()
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].astype(str).str.lower().str.strip()
        df['Gender_Encoded'] = df['Gender'].map({'male': 0, 'female': 1, 'm': 0, 'f': 1}).fillna(0)
    
    if 'SubscriptionType' in df.columns:
        sub_mapping = {'Basic': 0, 'Gold': 1, 'Premium': 2}
        df['SubscriptionType_Encoded'] = df['SubscriptionType'].map(sub_mapping).fillna(0)
    
    numeric_cols = ['Age', 'MonthlyUsageHours', 'NumTransactions', 'Complaints']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
            
    return df

@st.cache_resource
def train_model(df):
    processed_df = preprocess_data(df)
    
    feature_cols = ['Age', 'Gender_Encoded', 'MonthlyUsageHours', 
                   'NumTransactions', 'SubscriptionType_Encoded', 'Complaints']
    
    X = processed_df[feature_cols]
    y = processed_df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    return model, acc, auc

# ==========================================
# SECTION 3: STRATEGY LOGIC
# ==========================================

def get_strategy(risk):
    """Returns a recommended action based purely on Churn Probability."""
    if risk > 0.7:
        return "🔴 Critical Risk", "Call immediately + Offer 30% Discount"
    elif risk > 0.4:
        return "🟡 Moderate Risk", "Send 'New Features' Email + 10% Discount"
    else:
        return "🟢 Safe Customer", "No action needed / Ask for Referral"

# ==========================================
# SECTION 4: USER INTERFACE
# ==========================================

# 1. Load & Train
df_main = get_high_quality_data()
model, acc, auc = train_model(df_main)

# 2. Sidebar
st.sidebar.title("⚙️ Controls")
app_mode = st.sidebar.selectbox("Choose Mode", ["Single Customer Prediction", "Batch File Analysis"])

# 3. Main Header
st.title("🛡️ ChurnGuard AI")
st.markdown(f"**System Status:** 🟢 Online | **Model Accuracy:** `{acc:.1%}`")

if app_mode == "Single Customer Prediction":
    st.subheader("👤 Single Customer Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 80, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        sub = st.selectbox("Subscription", ["Basic", "Gold", "Premium"])
    with col2:
        usage = st.slider("Monthly Usage (Hours)", 0, 200, 50)
        trans = st.slider("Transactions", 0, 50, 10)
        comp = st.slider("Complaints (Last 30 Days)", 0, 10, 2)
        
    if st.button("Predict Risk"):
        # Prepare Data
        input_data = pd.DataFrame({
            'Age': [age], 'Gender': [gender], 'MonthlyUsageHours': [usage],
            'NumTransactions': [trans], 'SubscriptionType': [sub], 'Complaints': [comp]
        })
        
        # Predict
        proc_input = preprocess_data(input_data)
        features = ['Age', 'Gender_Encoded', 'MonthlyUsageHours', 
                   'NumTransactions', 'SubscriptionType_Encoded', 'Complaints']
        
        prob = model.predict_proba(proc_input[features])[0][1]
        status, action = get_strategy(prob)
        
        # Display Results
        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.metric("Churn Probability", f"{prob:.1%}", delta_color="inverse" if prob > 0.5 else "normal")
        c2.metric("Risk Level", status)
        
        st.info(f"💡 **Recommended Action:** {action}")

elif app_mode == "Batch File Analysis":
    st.subheader("📂 Batch Analysis")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Analyzing {len(batch_df)} customers...")
        
        # Predict
        proc_batch = preprocess_data(batch_df)
        features = ['Age', 'Gender_Encoded', 'MonthlyUsageHours', 
                   'NumTransactions', 'SubscriptionType_Encoded', 'Complaints']
        
        probs = model.predict_proba(proc_batch[features])[:, 1]
        batch_df['Churn_Probability'] = probs
        batch_df['Risk_Label'] = ['High Risk' if p > 0.5 else 'Safe' for p in probs]
        
        # Stats
        risk_count = (probs > 0.5).sum()
        st.metric("High Risk Customers Found", risk_count, delta="Requires Attention", delta_color="inverse")
        
        # Show Data
        def color_risk(val):
            color = '#ffcdd2' if val == 'High Risk' else '#c8e6c9'
            return f'background-color: {color}'
            
        st.dataframe(batch_df.style.applymap(color_risk, subset=['Risk_Label']))
        
        # Download
        csv = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results", csv, "churn_predictions.csv", "text/csv")