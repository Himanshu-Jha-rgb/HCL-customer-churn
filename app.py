import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="ChurnGuard AI", page_icon="🛡️", layout="wide")

# --- TITLE & DESCRIPTION ---
st.title("🛡️ ChurnGuard AI: Customer Retention System")
st.markdown("""
This application predicts customer churn using a Random Forest algorithm. 
Adjust the customer profile in the sidebar to see real-time predictions.
""")

# --- DATA HANDLING ---
@st.cache_data
def load_and_prep_data():
    # Try to load the "High Accuracy" file first, else fallback
    try:
        df = pd.read_csv('customer_churn_data_final.csv')
    except FileNotFoundError:
        try:
            df = pd.read_csv('customer_churn_data_enhanced.csv')
        except FileNotFoundError:
            st.error("⚠️ Data file not found! Please run the data fixer script first.")
            st.stop()
    
    # Simple Preprocessing for the model
    # We map text to numbers manually to keep it simple for the App
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    
    # Handle Subscription Type mapping
    sub_mapping = {'Basic': 0, 'Gold': 1, 'Premium': 2}
    df['SubscriptionType'] = df['SubscriptionType'].map(sub_mapping)
    
    return df

df = load_and_prep_data()

# --- SIDEBAR: USER INPUTS ---
st.sidebar.header("User Input Features")

def user_input_features():
    age = st.sidebar.slider("Age", 18, 70, 45)
    
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    gender_val = 0 if gender == "Male" else 1
    
    usage = st.sidebar.slider("Monthly Usage (Hours)", 5, 200, 100)
    transactions = st.sidebar.slider("Num Transactions", 1, 50, 25)
    
    sub_type = st.sidebar.selectbox("Subscription Type", ("Basic", "Gold", "Premium"))
    sub_val = {'Basic': 0, 'Gold': 1, 'Premium': 2}[sub_type]
    
    complaints = st.sidebar.slider("Complaints (Last 30 Days)", 0, 10, 2)
    
    data = {
        'Age': age,
        'Gender_Encoded': gender_val,
        'MonthlyUsageHours': usage,
        'NumTransactions': transactions,
        'SubscriptionType_Encoded': sub_val,
        'Complaints': complaints
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- MODEL TRAINING ---
# We train the model 'live' but cache it so it doesn't reload every click
@st.cache_resource
def train_model(data):
    # Prepare X and y
    # Note: Columns must match the input_df order exactly
    feature_cols = ['Age', 'Gender', 'MonthlyUsageHours', 'NumTransactions', 'SubscriptionType', 'Complaints']
    X = data[feature_cols]
    # Rename columns to match input_df for safety
    X.columns = ['Age', 'Gender_Encoded', 'MonthlyUsageHours', 'NumTransactions', 'SubscriptionType_Encoded', 'Complaints']
    y = data['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    return model, acc, auc, X_test, y_test

model, acc, auc, X_test, y_test = train_test_split_model = train_model(df)

# --- MAIN DASHBOARD ---

# Row 1: Prediction Result
st.markdown("---")
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 Prediction Result")
    
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    if prediction[0] == 1:
        st.error(f"🚨 CHURN RISK DETECTED (Probability: {prediction_proba[0][1]:.2f})")
        st.write("Recommendation: Offer immediate discount or schedule support call.")
    else:
        st.success(f"✅ CUSTOMER IS SAFE (Probability: {prediction_proba[0][0]:.2f})")
        st.write("Recommendation: Keep engaging with standard newsletters.")

with col2:
    st.subheader("📊 Model Stats")
    st.metric(label="Model Accuracy", value=f"{acc:.1%}")
    st.metric(label="ROC AUC Score", value=f"{auc:.2f}")

# Row 2: Visualization Tabs
st.markdown("---")
st.header("📈 Model Performance & Insights")

tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Raw Data"])

with tab1:
    st.write("Which features drive customer churn?")
    # Calculate feature importance
    importances = model.feature_importances_
    feature_names = input_df.columns
    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_df = feat_df.sort_values(by='Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x='Importance', y='Feature', data=feat_df, palette='viridis', ax=ax)
    st.pyplot(fig)

with tab2:
    st.write("How often is the model right vs wrong?")
    cm = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)

with tab3:
    st.dataframe(df.head(20))