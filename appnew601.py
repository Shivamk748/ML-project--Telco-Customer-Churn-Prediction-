
import joblib                 
import numpy as np
import streamlit as st
import pandas as pd

# Load the trained model and preprocessing objects
model = joblib.load('Churn Prediction.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoders.pkl')

# Streamlit app title
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict the likelihood of churn.")


def user_input_features():
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"], key="gender")
    SeniorCitizen = st.sidebar.selectbox("SeniorCitizen", [0, 1], key="Senior_citizen", format_func=lambda x: "Yes" if x == 1 else "No")
    Partner = st.sidebar.radio("Partner", ["Yes", "No"], key="partner")
    Dependents = st.sidebar.radio("Dependents", ["Yes", "No"], key="dependents")
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 24, key="tenure")
    PhoneService = st.sidebar.radio("Phone Service", ["Yes", "No"], key="phone_service")
    MultipleLines = st.sidebar.selectbox("Multiple Lines", ["No phone service", "No", "Yes"], key="multiple_lines")
    InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "No", "Fiber optic"], key="internet_service")
    OnlineSecurity = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"], key="online_security")
    OnlineBackup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"], key="online_backup")
    DeviceProtection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"], key="device_protection")
    TechSupport = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"], key="tech_support")
    StreamingTV = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"], key="streaming_tv")
    StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], key="streaming_movies")
    Contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], key="contract")
    PaperlessBilling = st.sidebar.radio("Paperless Billing", ["Yes", "No"], key="paperless_billing")
    PaymentMethod = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], key="payment_method")
    MonthlyCharges = st.sidebar.slider("Monthly Charges", 10, 120, 50, key="monthly_charges")
    TotalCharges = st.sidebar.slider("TotalCharges", 0, 8000, 2000, key="total_charges")

    # Store user inputs in a DataFrame
    data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }
    
    return pd.DataFrame([data])


# Get user input
input_data = user_input_features()

# Show user input
st.subheader("Customer Data Preview")
st.write(input_data)

# Data preprocessing before prediction
def preprocess_data(df):
    """Ensure features match the trained model's expected format."""
    
    # Features used during model training
    expected_features = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", 
                         "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", 
                         "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", 
                         "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", 
                         "MonthlyCharges","TotalCharges"]  
    
    # Ensuring only the expected features are included
    df = df[expected_features]
    
    # Apply scaling to numerical columns
    numerical_features = ["SeniorCitizen","tenure", "MonthlyCharges","TotalCharges"]
    
    # Ensure all numerical features exist before scaling
    if set(numerical_features).issubset(df.columns):
        df[numerical_features] = scaler.transform(df[numerical_features])

    # Encode categorical features
    categorical_features = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", 
                            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
                            "TechSupport", "StreamingTV", "StreamingMovies", "Contract", 
                            "PaperlessBilling", "PaymentMethod"]
    
    df_categorical = encoder.transform(df[categorical_features])
    
    # Convert encoded categories to DataFrame
    df_categorical = pd.DataFrame(df_categorical, columns=encoder.get_feature_names_out(categorical_features))
    
    # Merge processed numerical and categorical data
    df_processed = pd.concat([df[['SeniorCitizen', 'tenure', 'MonthlyCharges',"TotalCharges"]], df_categorical], axis=1)
    
    return df_processed

# Predict and display result
if st.button("Predict Churn"):
    processed_data = preprocess_data(input_data)
    prediction = model.predict(processed_data)
    st.write("customer is churned :", prediction[0] )
    # result = "Churn" if prediction[0] == 1 else "No Churn"
    
  
