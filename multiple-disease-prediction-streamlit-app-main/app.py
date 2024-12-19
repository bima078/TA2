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
            Glucose = st.text_input('Glucose Level', placeholder='mg/dL')

        with col3:
            BloodPressure = st.text_input('Blood Pressure value', placeholder='mm Hg')

        with col1:
            SkinThickness = st.text_input('Skin Thickness value', placeholder='mm')

        with col2:
            Insulin = st.text_input('Insulin Level', placeholder='µU/ml')

        with col3:
            BMI = st.text_input('BMI value', placeholder='kg/m^2')

        with col1:
            DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value', placeholder='Input Here')

        with col2:
            Age = st.text_input('Age of the Person', placeholder='years old')

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
    st.title('Heart Disease Prediction using Linear Regression Method')
    st.write("Accuracy Model Untuk Data Training : **85%** ")
    st.write("Accuracy Model Untuk Data Uji : **82%** ")

    # Download training data
    download_training_data(f'{working_dir}/dataset/heart.csv', 'Heart Disease Training Data')

    # Add option for single or batch prediction
    prediction_type = st.radio("Select Prediction Type", ('Single Prediction', 'Batch Prediction'))

    if prediction_type == 'Single Prediction':
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            age = st.text_input('Age', placeholder='Years old')

        with col2:
            sex = st.text_input('Sex', placeholder="Enter Value (0-1)")
            st.caption("""
            - **0**: Female  
            - **1**: Male 
            """)

        with col3:
            cp = st.text_input('Chest Pain types', placeholder="Enter value (1-4)")
            st.caption("""
            - **1**: Typical angina  
            - **2**: Atypical angina  
            - **3**: Non-anginal pain  
            - **4**: Asymptomatic
            """)

        with col1:
            trestbps = st.text_input('Resting Blood Pressure', placeholder="Enter value (mm Hg)")

        with col2:
            chol = st.text_input('Serum Cholestoral', placeholder="Enter value (mg/dl)")

        with col3:
            fbs = st.text_input('Fasting Blood Sugar', placeholder="Enter Value (0-1)")
            st.caption("""
              - **0**: Blood sugar level <= 120 mg/dl  
              - **1**: Blood Sugar Level > 120 mg/dl
              """)

        with col1:
            restecg = st.text_input('Resting Electrocardiographic results', placeholder="Enter Value (0-2)")
            st.caption("""
            - **0**: Normal  
            - **1**: ST-T wave abnormality  
            - **2**: Hipertropi in left ventricular
            """)

        with col2:
            thalach = st.text_input('Maximum Heart Rate achieved', placeholder="Enter Value (bpm)")

        with col3:
            exang = st.text_input('Exercise Induced Angina', placeholder="Enter Value (0-1)")
            st.caption("""
              - **0**: No Chest pain when doing exercise  
              - **1**: Had Chest pain when doing exercise  
              """)

        with col1:
            oldpeak = st.text_input('ST depression induced by exercise', placeholder="Enter Value")

        with col2:
            slope = st.text_input('Slope of the peak exercise ST segment', placeholder="Enter Value (1-3)")
            st.caption("""
              - **1**: Upsloping  
              - **2**: Flat 
              - **3**: Downsloping 
              """)

        with col3:
            ca = st.text_input('Major vessels colored by flourosopy', placeholder="Enter Value")

        with col1:
            thal = st.text_input('thal', placeholder="Enter Value (0-2)")
            st.caption("""
                - **0**: Normal  
                - **1**: Fixed Defect 
                - **2**: Reversal Defect
                """)

        # Adjust layout alignment by adding empty captions where necessary
        with col2:
            st.caption("")

        with col3:
            st.caption("")

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
    st.write("Accuracy Model Untuk Data Training : **87%** ")
    st.write("Accuracy Model Untuk Data Uji : **87%** ")

    # Download training data
    download_training_data(f'{working_dir}/dataset/parkinsons.csv', "Parkinson's Training Data")

    # Add option for single or batch prediction
    prediction_type = st.radio("Select Prediction Type", ('Single Prediction', 'Batch Prediction'))

    if prediction_type == 'Single Prediction':
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            fo = st.text_input('MDVP:Fo(Hz)')

        with col2:
            fhi = st.text_input('MDVP:Fhi(Hz)')

        with col3:
            flo = st.text_input('MDVP:Flo(Hz)')

        with col4:
            Jitter_percent = st.text_input('MDVP:Jitter(%)')

        with col5:
            Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

        with col1:
            RAP = st.text_input('MDVP:RAP')

        with col2:
            PPQ = st.text_input('MDVP:PPQ')

        with col3:
            DDP = st.text_input('Jitter:DDP')

        with col4:
            Shimmer = st.text_input('MDVP:Shimmer')

        with col5:
            Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

        with col1:
            APQ3 = st.text_input('Shimmer:APQ3')

        with col2:
            APQ5 = st.text_input('Shimmer:APQ5')

        with col3:
            APQ = st.text_input('MDVP:APQ')

        with col4:
            DDA = st.text_input('Shimmer:DDA')

        with col5:
            NHR = st.text_input('NHR')

        with col1:
            HNR = st.text_input('HNR')

        with col2:
            RPDE = st.text_input('RPDE')

        with col3:
            DFA = st.text_input('DFA')

        with col4:
            spread1 = st.text_input('spread1')

        with col5:
            spread2 = st.text_input('spread2')

        with col1:
            D2 = st.text_input('D2')

        with col2:
            PPE = st.text_input('PPE')

        # Prediction button
        if st.button("Parkinson's Disease Test Result"):
            try:
                user_input = [float(x) for x in [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]]
                parkinsons_prediction = parkinsons_model.predict([user_input])

                result = 'Positive' if parkinsons_prediction[0] == 1 else 'Negative'
                st.success(f'The person is {result} for Parkinson\'s disease.')

                # Display result in a single-row table
                result_df = pd.DataFrame([user_input + [result]], columns=['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
                      'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
                      'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE', 'Prediction'])
                st.dataframe(result_df)
            except ValueError:
                st.error("Please provide valid numerical inputs.")

    elif prediction_type == 'Batch Prediction':
        required_columns = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
                            'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
                            'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE',
                            'DFA', 'spread1', 'spread2', 'D2', 'PPE']
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
