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

# Function to download training dataset
def download_training_data(file_path, display_name):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            st.download_button(f"Download {display_name} Training Data", file, file_name=display_name, mime="text/csv")
    else:
        st.error(f"Training dataset for {display_name} not found.")

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using SVM Method')

    # Download training data
    download_training_data(f'{working_dir}/dataset/diabetes.csv', 'Diabetes Training Data')

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

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using SVM Method')

    # Download training data
    download_training_data(f'{working_dir}/dataset/heart.csv', 'Heart Disease Training Data')

    prediction_type = st.radio("Select Prediction Type", ('Single Prediction', 'Batch Prediction'))

    if prediction_type == 'Single Prediction':
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.text_input('Age')

        with col2:
            sex = st.text_input('Sex')

        with col3:
            cp = st.text_input('Chest Pain types')

        with col1:
            trestbps = st.text_input('Resting Blood Pressure')

        with col2:
            chol = st.text_input('Serum Cholestoral in mg/dl')

        with col3:
            fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

        with col1:
            restecg = st.text_input('Resting Electrocardiographic results')

        with col2:
            thalach = st.text_input('Maximum Heart Rate achieved')

        with col3:
            exang = st.text_input('Exercise Induced Angina')

        with col1:
            oldpeak = st.text_input('ST depression induced by exercise')

        with col2:
            slope = st.text_input('Slope of the peak exercise ST segment')

        with col3:
            ca = st.text_input('Major vessels colored by flourosopy')

        with col1:
            thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

        # Prediction button
        if st.button('Heart Disease Test Result'):
            try:
                user_input = [float(x) for x in [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
                heart_prediction = heart_disease_model.predict([user_input])

                if heart_prediction[0] == 1:
                    st.success('The person is having heart disease')
                else:
                    st.success('The person does not have any heart disease')
            except ValueError:
                st.error("Please provide valid numerical inputs.")

    elif prediction_type == 'Batch Prediction':
        required_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        download_template(required_columns, "Heart Disease Prediction")

        uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type=['csv'])

        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.write("Uploaded Data:", data.head())

                # Ensure the data has the correct columns
                if all(col in data.columns for col in required_columns):
                    predictions = batch_predict(heart_disease_model, data[required_columns])
                    data['Prediction'] = predictions
                    st.write("Predictions:", data)
                    st.download_button("Download Predictions", data.to_csv(index=False), "predictions.csv", "text/csv")
                else:
                    st.error(f"CSV file must contain the following columns: {', '.join(required_columns)}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Parkinson's Prediction Page
if selected == 'Parkinsons Prediction':
    st.title("Parkinson's Disease Prediction using SVM Method")

    # Download training data
    download_training_data(f'{working_dir}/dataset/parkinsons.csv', "Parkinson's Training Data")

    prediction_type = st.radio("Select Prediction Type", ('Single Prediction', 'Batch Prediction'))

    if prediction_type == 'Single Prediction':
        col1, col2, col3 = st.columns(3)

        with col1:
            MDVP_Fo = st.text_input('MDVP:Fo (Hz)')

        with col2:
            MDVP_Fhi = st.text_input('MDVP:Fhi (Hz)')

        with col3:
            MDVP_Flo = st.text_input('MDVP:Flo (Hz)')

        with col1:
            MDVP_Jitter_percent = st.text_input('MDVP:Jitter (%)')

        with col2:
            MDVP_Jitter_Abs = st.text_input('MDVP:Jitter (Abs)')

        with col3:
            MDVP_RAP = st.text_input('MDVP:RAP')

        with col1:
            MDVP_PPQ = st.text_input('MDVP:PPQ')

        with col2:
            Jitter_DDP = st.text_input('Jitter:DDP')

        with col3:
            MDVP_Shimmer = st.text_input('MDVP:Shimmer')

        with col1:
            MDVP_Shimmer_dB = st.text_input('MDVP:Shimmer (dB)')

        with col2:
            Shimmer_APQ3 = st.text_input('Shimmer:APQ3')

        with col3:
            Shimmer_APQ5 = st.text_input('Shimmer:APQ5')

        with col1:
            MDVP_APQ = st.text_input('MDVP:APQ')

        with col2:
            Shimmer_DDA = st.text_input('Shimmer:DDA')

        with col3:
            NHR = st.text_input('NHR')

        with col1:
            HNR = st.text_input('HNR')

        with col2:
            RPDE = st.text_input('RPDE')

        with col3:
            DFA = st.text_input('DFA')

        with col1:
            spread1 = st.text_input('Spread1')

        with col2:
            spread2 = st.text_input('Spread2')

        with col3:
            D2 = st.text_input('D2')

        with col1:
            PPE = st.text_input('PPE')

        # Prediction button
        if st.button("Parkinson's Test Result"):
            try:
                user_input = [float(x) for x in [MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitter_percent, MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]]
                parkinsons_prediction = parkinsons_model.predict([user_input])

                if parkinsons_prediction[0] == 1:
                    st.success("The person has Parkinson's disease")
                else:
                    st.success("The person does not have Parkinson's disease")
            except ValueError:
                st.error("Please provide valid numerical inputs.")

    elif prediction_type == 'Batch Prediction':
        required_columns = ['MDVP:Fo (Hz)', 'MDVP:Fhi (Hz)', 'MDVP:Flo (Hz)', 'MDVP:Jitter (%)', 'MDVP:Jitter (Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer (dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'Spread1', 'Spread2', 'D2', 'PPE']
        download_template(required_columns, "Parkinson's Prediction")

        uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type=['csv'])

        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.write("Uploaded Data:", data.head())

                # Ensure the data has the correct columns
                if all(col in data.columns for col in required_columns):
                    predictions = batch_predict(parkinsons_model, data[required_columns])
                    data['Prediction'] = predictions
                    st.write("Predictions:", data)
                    st.download_button("Download Predictions", data.to_csv(index=False), "predictions.csv", "text/csv")
                else:
                    st.error(f"CSV file must contain the following columns: {', '.join(required_columns)}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
