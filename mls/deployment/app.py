import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="shrikantpillay/MLOPS", filename="best_tourism_model_v1.joblib.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package
st.title("Tourism Package Prediction")
st.write("""
predicting the likelihood of purchasing the Wellness Tourism Package. .
""")


# User input
age = st.number_input("Age", min_value=18, max_value=100, value=30)
typeofcontact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
citytier = st.selectbox("City Tier", ["1", "2", "3"])
gender = st.selectbox("Gender", ["Male", "Female"])
occupation = st.selectbox("Occupation", ["Freelancer", "Large Business","Small Business","Salaried"])
Numberofpersonvisiting = st.number_input("Number of People Visiting", min_value=1, max_value=5, value=2)
preferredpropertystar = st.selectbox("Preferred Property Star", ["3", "4", "5"])
MaritalStatus = st.selectbox("Marital Status", ["Married", "Single"])
numberoftrips = st.number_input("Number of Trips", min_value=1, max_value=22, value=3)
passport = st.selectbox("Passport", ["Yes", "No"])
owncar = st.selectbox("Own Car", ["Yes", "No"])
Numberofchildrenvisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=3, value=0)
Designation= st.selectbox("Designation", ["Executive", "AVP","Manager","Senior Manager","VP"])
monthlyincome = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=50000)
Productpitched = st.selectbox("Product Pitched", ["Basic", "Delux", "King","Standard","Super Delux"])

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'typeofcontact': typeofcontact,
    'citytier': citytier,
    'gender': gender,
    'occupation': occupation,
    'Numberofpersonvisiting': Numberofpersonvisiting,
    'preferredpropertystar': preferredpropertystar,
    'MaritalStatus': MaritalStatus,
    'numberoftrips': numberoftrips,
    'passport': passport,
    'owncar': owncar,
    'Numberofchildrenvisiting': Numberofchildrenvisiting,
    'Designation': Designation,
    'monthlyincome': monthlyincome,
    'ProductPitched': Productpitched
}])


if st.button("Predict Purchasing"):
    prediction = model.predict(input_data)[0]
    result = "Predict Purchasing" if prediction == 1 else "Customer will purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
