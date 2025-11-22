import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="ChurnGuard AI", page_icon="🛡️", layout="wide")

# ==========================================
# SECTION 1: DATA GENERATION (The Engine)
# ==========================================
@st.cache_data
def get_high_quality_data():
    """Generates synthetic data where EVERY column matters."""
    try:
        df = pd.read_csv('customer_churn_data.csv')
    except FileNotFoundError:
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

    # --- INJECT LOGIC ---
    df['Churn_Prob'] = 0.5 

    # 1. Age (Young = Risky)
    df.loc[df['Age'] < 30, 'Churn_Prob'] += 0.25
    df.loc[df['Age'] > 55, 'Churn_Prob'] -= 0.20

    # 2. Gender (Male slightly riskier pattern)
    df.loc[df['Gender'] == 'Male', 'Churn_Prob'] += 0.05

    # 3. Subscription (Premium = Safe)
    df.loc[df['SubscriptionType'] == 'Premium', 'Churn_Prob'] -= 0.15

    # 4. Usage (High usage = Safe)
    df.loc[df['MonthlyUsageHours'] > 150, 'Churn_Prob'] -= 0.20
    df.loc[df['MonthlyUsageHours'] < 20, 'Churn_Prob'] += 0.20

    # 5. Complaints (The Big Driver)
    df.loc[df['Complaints'] >= 7, 'Churn_Prob'] += 0.40
    df.loc[df['Complaints'] <= 1, 'Churn_Prob'] -= 0.30

    # 6. Transactions (Active = Safe)
    df.loc[df['NumTransactions'] > 30, 'Churn_Prob'] -= 0.10

    np.random.seed(42)
    df['Churn'] = np.random.binomial(1, df['Churn_Prob'].clip(0,1))
    df = df.drop(columns=['Churn_Prob'])
    return df

# ==========================================
# SECTION 2: MODELING
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
    return model, acc, auc, feature_cols

# ==========================================
# SECTION 3: UI & DASHBOARD
# ==========================================

def get_strategy(risk):
    if risk > 0.7: return "🔴 Critical", "Call + 30% Off"
    elif risk > 0.4: return "🟡 Moderate", "Email + 10% Off"
    else: return "🟢 Safe", "Ask for Referral"

# Load Logic
df_main = get_high_quality_data()
model, acc, auc, feature_names = train_model(df_main)

# Header
st.title("🛡️ ChurnGuard AI: 360° Analytics")
st.markdown(f"**System Status:** 🟢 Online | **Accuracy:** `{acc:.1%}`")

# Sidebar
app_mode = st.sidebar.selectbox("Choose Mode", ["Single Customer Prediction", "Batch File Analysis"])

if app_mode == "Single Customer Prediction":
    st.subheader("👤 Single Customer Analysis")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 80, 25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        sub = st.selectbox("Subscription", ["Basic", "Gold", "Premium"])
    with col2:
        usage = st.slider("Monthly Usage (Hours)", 0, 200, 50)
        trans = st.slider("Transactions", 0, 50, 10)
        comp = st.slider("Complaints (Last 30 Days)", 0, 10, 2)
        
    if st.button("Predict Risk"):
        input_data = pd.DataFrame({
            'Age': [age], 'Gender': [gender], 'MonthlyUsageHours': [usage],
            'NumTransactions': [trans], 'SubscriptionType': [sub], 'Complaints': [comp]
        })
        proc_input = preprocess_data(input_data)
        features = ['Age', 'Gender_Encoded', 'MonthlyUsageHours', 
                   'NumTransactions', 'SubscriptionType_Encoded', 'Complaints']
        prob = model.predict_proba(proc_input[features])[0][1]
        status, action = get_strategy(prob)
        
        c1, c2 = st.columns(2)
        c1.metric("Churn Risk", f"{prob:.1%}", delta_color="inverse" if prob > 0.5 else "normal")
        c2.metric("Status", status)
        st.info(f"**Recommended Action:** {action}")
        
        # Mini Feature Importance
        importances = model.feature_importances_
        feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        st.bar_chart(feat_df.set_index('Feature'))

elif app_mode == "Batch File Analysis":
    st.subheader("📂 Full Population Analytics")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(batch_df)} records.")
        
        # Predict
        proc_batch = preprocess_data(batch_df)
        features = ['Age', 'Gender_Encoded', 'MonthlyUsageHours', 
                   'NumTransactions', 'SubscriptionType_Encoded', 'Complaints']
        probs = model.predict_proba(proc_batch[features])[:, 1]
        batch_df['Churn_Probability'] = probs
        batch_df['Risk_Label'] = ['High Risk' if p > 0.5 else 'Safe' for p in probs]
        
        # --- DASHBOARD TABS ---
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔍 Input Data Overview", 
            "🧠 Churn Risk Analysis", 
            "📊 Feature Importance", 
            "📋 Prediction Table"
        ])
        
        # TAB 1: UPLOADED DATA ANALYSIS (The New Feature)
        with tab1:
            st.markdown("### 📊 Analysis of Uploaded Dataset")
            st.write("Overview of the customer demographics before prediction.")
            
            # Row 1: Categorical
            c1, c2 = st.columns(2)
            with c1:
                if 'Gender' in batch_df.columns:
                    st.write("**Gender Distribution**")
                    fig, ax = plt.subplots(figsize=(5,3))
                    batch_df['Gender'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#66b3ff','#ff9999'], ax=ax)
                    ax.set_ylabel('')
                    st.pyplot(fig)
            with c2:
                if 'SubscriptionType' in batch_df.columns:
                    st.write("**Subscription Distribution**")
                    fig, ax = plt.subplots(figsize=(5,3))
                    sns.countplot(x='SubscriptionType', data=batch_df, palette='viridis', ax=ax)
                    st.pyplot(fig)
            
            # Row 2: Numerical
            st.markdown("---")
            st.write("**Numerical Distributions**")
            c3, c4 = st.columns(2)
            with c3:
                st.write("Age Distribution")
                st.bar_chart(batch_df['Age'].value_counts().sort_index())
            with c4:
                st.write("Monthly Usage Distribution")
                st.line_chart(batch_df['MonthlyUsageHours'])
                
            # Summary Stats
            with st.expander("View Detailed Statistics"):
                st.write(batch_df.describe())

        # TAB 2: RISK ANALYSIS
        with tab2:
            st.markdown("### 🧠 What is driving the Risk?")
            
            # Age vs Risk
            st.write("**1. Churn Risk by Age Group**")
            batch_df['Age Group'] = pd.cut(batch_df['Age'], bins=[18, 30, 50, 80], labels=['Young (18-30)', 'Mid (31-50)', 'Senior (50+)'])
            age_risk = batch_df.groupby('Age Group')['Churn_Probability'].mean()
            st.bar_chart(age_risk)
            
            # Complaints vs Risk
            col1, col2 = st.columns(2)
            with col1:
                st.write("**2. Impact of Complaints**")
                fig_c, ax_c = plt.subplots()
                sns.barplot(x='Complaints', y='Churn_Probability', data=batch_df, palette='Reds', ax=ax_c)
                st.pyplot(fig_c)
            with col2:
                st.write("**3. Impact of Usage**")
                fig_u, ax_u = plt.subplots()
                sns.scatterplot(x='MonthlyUsageHours', y='Churn_Probability', hue='Risk_Label', data=batch_df, ax=ax_u)
                st.pyplot(fig_u)

        # TAB 3: FEATURE IMPORTANCE
        with tab3:
            st.markdown("### 🔑 Global Feature Importance")
            importances = model.feature_importances_
            feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x='Importance', y='Feature', data=feat_df, palette='magma', ax=ax)
            st.pyplot(fig)

        # TAB 4: DATA TABLE
        with tab4:
            def highlight_risk(val):
                return f'background-color: {"#ffcdd2" if val == "High Risk" else "#c8e6c9"}'
            st.dataframe(batch_df.style.applymap(highlight_risk, subset=['Risk_Label']))
            csv = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results", csv, "predictions.csv", "text/csv")