import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="shrikantpillay/MLOPS", filename="best_tourism_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package
st.title("Tourism Package Prediction")
st.write("""
predicting the likelihood of purchasing the Wellness Tourism Package. .
""")


# User input
age = st.number_input("Age", min_value=18, max_value=100, value=30)
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
CityTier = st.selectbox("City Tier", ["1", "2", "3"])
Gender = st.selectbox("Gender", ["Male", "Female"])
Occupation = st.selectbox("Occupation", ["Freelancer", "Large Business","Small Business","Salaried"])
NumberOfPersonVisiting = st.number_input("Number of People Visiting", min_value=1, max_value=5, value=2)
NumberOfFollowups=st.number_input("No of Followups",min_value=1, max_value=10,value=2)
PreferredPropertyStar = st.selectbox("Preferred Property Star", ["3", "4", "5"])
MaritalStatus = st.selectbox("Marital Status", ["Married", "Single"])
NumberOfTrips = st.number_input("Number of Trips", min_value=1, max_value=22, value=3)
Passport = st.selectbox("Passport", ["Yes", "No"])
OwnCar = st.selectbox("Own Car", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=3, value=0)
Designation= st.selectbox("Designation", ["Executive", "AVP","Manager","Senior Manager","VP"])
MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=50000)
DurationOfPitch =st.number_input("Pitch Duration", min_value=5 , max_value=200,value=5)
Productpitched = st.selectbox("Product Pitched", ["Basic", "Delux", "King","Standard","Super Delux"])
PitchSatisfactionScore=st.selectbox("PitchSatisfactionScore",["1","2","3","4","5"])
# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'Gender': Gender,
    'Occupation': Occupation,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups' : NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': monthlyincome,
    'ProductPitched': Productpitched,
    'DurationOfPitch':DurationOfPitch,
    'PitchSatisfactionScore': PitchSatisfactionScore
}])


if st.button("Predict Purchasing"):
    prediction = model.predict(input_data)[0]
    result = "Predict Purchasing" if prediction == 1 else "Customer will purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
