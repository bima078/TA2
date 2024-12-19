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

# Function to display and download predictions
def display_and_download_predictions(data, predictions_column="Prediction"):
    st.write("Predictions:")
    st.dataframe(data)
    st.download_button("Download Predictions", data.to_csv(index=False), "predictions.csv", "text/csv")

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using SVM Method')
    st.write("Accuracy Model Untuk Data Training : **78%** ")
    st.write("Accuracy Model Untuk Data Uji : **77%** ")
  
    # Download training data
    download_training_data(f'{working_dir}/dataset/diabetes.csv', 'Diabetes Training Data')

    # Add option for single or batch prediction
    prediction_type = st.radio("Select Prediction Type", ('Single Prediction', 'Batch Prediction'))

    if prediction_type == 'Single Prediction':
        col1, col2, col3 = st.columns(3)

        with col1:
            Pregnancies = st.text_input('Number of Pregnancies', placeholder='Enter number of pregnancies here')

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

                result = 'Positive' if diab_prediction[0] == 1 else 'Negative'
                st.success(f'The person is {result} for diabetes.')

                # Display result in a single-row table
                result_df = pd.DataFrame([user_input + [result]], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Prediction'])
                st.dataframe(result_df)
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
                    display_and_download_predictions(data)
                else:
                    st.error(f"CSV file must contain the following columns: {', '.join(required_columns)}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using SVM Method')

    # Download training data
    download_training_data(f'{working_dir}/dataset/heart.csv', 'Heart Disease Training Data')

    # Add option for single or batch prediction
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

                result = 'Positive' if heart_prediction[0] == 1 else 'Negative'
                st.success(f'The person is {result} for heart disease.')

                # Display result in a single-row table
                result_df = pd.DataFrame([user_input + [result]], columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'Prediction'])
                st.dataframe(result_df)
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
                    display_and_download_predictions(data)
                else:
                    st.error(f"CSV file must contain the following columns: {', '.join(required_columns)}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Parkinson's Disease Prediction Page
if selected == 'Parkinsons Prediction':
    st.title("Parkinson's Disease Prediction using SVM Method")

    # Download training data
    download_training_data(f'{working_dir}/dataset/parkinsons.csv', "Parkinson's Training Data")

    # Add option for single or batch prediction
    prediction_type = st.radio("Select Prediction Type", ('Single Prediction', 'Batch Prediction'))

    if prediction_type == 'Single Prediction':
        col1, col2, col3 = st.columns(3)

        with col1:
            MDVP_Fo = st.text_input("MDVP:Fo(Hz) - Average vocal fundamental frequency")

        with col2:
            MDVP_Fhi = st.text_input("MDVP:Fhi(Hz) - Maximum vocal fundamental frequency")

        with col3:
            MDVP_Flo = st.text_input("MDVP:Flo(Hz) - Minimum vocal fundamental frequency")

        with col1:
            MDVP_Jitter = st.text_input("MDVP:Jitter(%)")

        with col2:
            MDVP_Shimmer = st.text_input("MDVP:Shimmer")

        with col3:
            NHR = st.text_input("NHR - Noise-to-Harmonics ratio")

        with col1:
            HNR = st.text_input("HNR - Harmonics-to-Noise ratio")

        with col2:
            RPDE = st.text_input("RPDE - Recurrence period density entropy")

        with col3:
            DFA = st.text_input("DFA - Signal fractal scaling exponent")

        with col1:
            spread1 = st.text_input("Spread1 - Nonlinear measures of fundamental frequency")

        with col2:
            spread2 = st.text_input("Spread2 - Nonlinear measures of fundamental frequency")

        with col3:
            D2 = st.text_input("D2 - Dynamical complexity measures")

        with col1:
            PPE = st.text_input("PPE - Pitch period entropy")

        # Prediction button
        if st.button("Parkinson's Disease Test Result"):
            try:
                user_input = [float(x) for x in [MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitter, MDVP_Shimmer, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]]
                parkinsons_prediction = parkinsons_model.predict([user_input])

                result = 'Positive' if parkinsons_prediction[0] == 1 else 'Negative'
                st.success(f'The person is {result} for Parkinson\'s disease.')

                # Display result in a single-row table
                result_df = pd.DataFrame([user_input + [result]], columns=['MDVP:Fo', 'MDVP:Fhi', 'MDVP:Flo', 'MDVP:Jitter', 'MDVP:Shimmer', 'NHR', 'HNR', 'RPDE', 'DFA', 'Spread1', 'Spread2', 'D2', 'PPE', 'Prediction'])
                st.dataframe(result_df)
            except ValueError:
                st.error("Please provide valid numerical inputs.")

    elif prediction_type == 'Batch Prediction':
        required_columns = ['MDVP:Fo', 'MDVP:Fhi', 'MDVP:Flo', 'MDVP:Jitter', 'MDVP:Shimmer', 'NHR', 'HNR', 'RPDE', 'DFA', 'Spread1', 'Spread2', 'D2', 'PPE']
        download_template(required_columns, "Parkinson's Disease Prediction")

        uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type=['csv'])

        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.write("Uploaded Data:", data.head())

                # Ensure the data has the correct columns
                if all(col in data.columns for col in required_columns):
                    predictions = batch_predict(parkinsons_model, data[required_columns])
                    data['Prediction'] = predictions
                    display_and_download_predictions(data)
                else:
                    st.error(f"CSV file must contain the following columns: {', '.join(required_columns)}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
