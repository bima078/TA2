import os
import pickle
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide")

# Get the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load the saved models
diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# Function to handle batch predictions
def batch_predict(model, input_df):
    predictions = model.predict(input_df)
    return ["Positive" if pred == 1 else "Negative" for pred in predictions]

# Function to download template
def download_template(columns, filename):
    template = pd.DataFrame(columns=columns)
    csv = template.to_csv(index=False)
    st.download_button(f"Download {filename} Template", csv, f"{filename}.csv", "text/csv")

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using SVM Method')

    # Add option for single or batch prediction
    prediction_type = st.radio("Select Prediction Type", ('Single Prediction', 'Batch Prediction'))

    if prediction_type == 'Single Prediction':
        col1, col2, col3 = st.columns(3)

        with col1:
            Pregnancies = st.text_input('Number of Pregnancies')

        with col2:
            Glucose = st.text_input('Glucose Level')

        with col3:
            BloodPressure = st.text_input('Blood Pressure value')

        with col1:
            SkinThickness = st.text_input('Skin Thickness value')

        with col2:
            Insulin = st.text_input('Insulin Level')

        with col3:
            BMI = st.text_input('BMI value')

        with col1:
            DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

        with col2:
            Age = st.text_input('Age of the Person')

        # Prediction button
        if st.button('Diabetes Test Result'):
            try:
                user_input = [float(x) for x in [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
                diab_prediction = diabetes_model.predict([user_input])

                if diab_prediction[0] == 1:
                    st.success('The person is diabetic')
                else:
                    st.success('The person is not diabetic')
            except ValueError:
                st.error("Please provide valid numerical inputs.")

    elif prediction_type == 'Batch Prediction':
        required_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        download_template(required_columns, "Diabetes Prediction")

        uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type=['csv'])

        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.write("Uploaded Data:", data.head())

                # Ensure the data has the correct columns
                if all(col in data.columns for col in required_columns):
                    predictions = batch_predict(diabetes_model, data[required_columns])
                    data['Prediction'] = predictions
                    st.write("Predictions:", data)
                    st.download_button("Download Predictions", data.to_csv(index=False), "predictions.csv", "text/csv")
                else:
                    st.error(f"CSV file must contain the following columns: {', '.join(required_columns)}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Repeat similar logic for Heart Disease and Parkinson's Prediction Pages
# The implementation is omitted for brevity but follows the same pattern as the Diabetes Prediction Page.
