import os
import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu  # Import the option_menu function

# Set page configuration
st.set_page_config(page_title="Health Data Analysis and Prediction", layout="wide", page_icon="📊")

# Load the models
diabetes_model = pickle.load(open("Diabetes/Diabetes Prediction.sav", 'rb'))
parkinson_model = pickle.load(open("Parkinson's Disease/parkinsons_model.sav", 'rb'))
heart_disease_model = pickle.load(open("Heart Disease/heart_disease_model.sav", 'rb'))

# Function for the main logic of the app
def main():
    # Sidebar for navigation
    with st.sidebar:
        selected = option_menu(
            'Multiple Disease Prediction and Data Visualization System',  # Title
            ['Diabetes Prediction', 'Heart Disease Prediction', "Parkinson's Prediction", "Data Visualizer"],  # Options
            icons=['activity', 'heart', 'person', 'bar-chart'],  # Icons for each option
            default_index=0  # Default selected option
        )

    # Diabetes Prediction Page
    if selected == 'Diabetes Prediction':
        st.title('Diabetes Prediction using ML')

        # Columns for input fields
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
            BMI = st.text_input('BMI Level')
        with col1:
            DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
        with col2:
            Age = st.text_input('Age of the Person')

        # Code for prediction
        diab_diagnosis = ''

        # Creating a button for prediction
        if st.button('Diabetes Test Result'):
            diab_prediction = diabetes_model.predict(
                [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
            if diab_prediction[0] == 1:
                diab_diagnosis = "The Person is Diabetic"
            else:
                diab_diagnosis = "The Person is not Diabetic"
        st.success(diab_diagnosis)

    # Heart Disease Prediction Page
    elif selected == 'Heart Disease Prediction':
        st.title('Heart Disease Prediction using ML')

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
            chol = st.text_input('Serum Cholesterol in mg/dl')
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
            thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversible defect')

        # Code for Prediction
        heart_diagnosis = ''

        # Creating a button for Prediction
        if st.button('Heart Disease Test Result'):
            user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            user_input = [float(x) for x in user_input]
            heart_prediction = heart_disease_model.predict([user_input])

            if heart_prediction[0] == 1:
                heart_diagnosis = 'The person is having heart disease'
            else:
                heart_diagnosis = 'The person does not have any heart disease'
        st.success(heart_diagnosis)

    # Parkinson's Prediction Page
    elif selected == "Parkinson's Prediction":
        st.title("Parkinson's Disease Prediction using ML")

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

        # Code for Prediction
        parkinsons_diagnosis = ''

        # Creating a button for Prediction
        if st.button("Parkinson's Test Result"):
            user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ,
                          DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
            user_input = [float(x) for x in user_input]
            parkinsons_prediction = parkinson_model.predict([user_input])

            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = "The person has Parkinson's disease"
            else:
                parkinsons_diagnosis = "The person does not have Parkinson's disease"
        st.success(parkinsons_diagnosis)

    # Data Visualizer Page
    elif selected == "Data Visualizer":
        st.title("📊 Data Visualizer")

        # Getting the working directory
        working_dir = os.path.dirname(os.path.abspath(__file__))

        # Specify the folder where your CSV files are located
        folder_path = os.path.join(working_dir, 'datasets')

        # List the files present in the "datasets" folder
        files_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

        # Dropdown to select a file
        selected_file = st.selectbox("Select a file", files_list, index=None)

        if selected_file:
            # Get the complete path of the selected file
            file_path = os.path.join(folder_path, selected_file)
            # Reading the CSV as a pandas dataframe
            df = pd.read_csv(file_path)
            columns = df.columns.tolist()

            # Display important statistics
            st.write("### Important Statistics")
            st.write(df.describe())

            # Data preview
            st.write("### Data Preview")
            st.write(df.head())

            # Select columns for plotting
            st.write("### Select Columns for Plotting")
            x_axis = st.selectbox("Select the X-axis", options=["None"] + columns, index=0)
            y_axis = st.selectbox("Select the Y-axis", options=["None"] + columns, index=0)

            # Filter data
            st.write("### Filter Data")
            filter_column = st.selectbox("Filter Column", options=["None"] + columns, index=0)
            filter_value = st.text_input("Filter Value")

            if filter_column != "None" and filter_value:
                df = df[df[filter_column] == filter_value]

            # Select plot type
            st.write("### Select Plot Type")
            plot_list = ["Line Plot", "Bar Chart", "Scatter Plot", "Distribution Plot", "Count Plot"]
            selected_plot = st.selectbox("Select a Plot", options=plot_list, index=0)

            # Plot customizations
            st.write("### Plot Customizations")
            color = st.color_picker("Pick a Color", "#69b3a2")
            line_style = st.selectbox("Line Style", options=["-", "--", "-.", ":"], index=0)

            # Generate the plot based on user selection
            if st.button("Generate Plot"):
                fig, ax = plt.subplots(figsize=(8, 5))

                if x_axis != "None" and y_axis != "None":
                    if selected_plot == "Line Plot":
                        sns.lineplot(x=df[x_axis], y=df[y_axis], ax=ax, color=color, linestyle=line_style)
                    elif selected_plot == "Bar Chart":
                        sns.barplot(x=df[x_axis], y=df[y_axis], ax=ax, color=color)
                    elif selected_plot == "Scatter Plot":
                        sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax, color=color)
                    elif selected_plot == "Distribution Plot":
                        sns.histplot(df[x_axis], kde=True, ax=ax, color=color)
                    elif selected_plot == "Count Plot":
                        sns.countplot(x=df[x_axis], ax=ax, color=color)
                elif selected_plot == "Distribution Plot" and x_axis != "None":
                    sns.histplot(df[x_axis], kde=True, ax=ax, color=color)
                elif selected_plot == "Count Plot" and x_axis != "None":
                    sns.countplot(x=df[x_axis], ax=ax, color=color)

                ax.tick_params(axis='x', labelsize=10)
                ax.tick_params(axis='y', labelsize=10)
                plt.title(f'{selected_plot} of {y_axis} vs {x_axis}', fontsize=12)
                plt.xlabel(x_axis, fontsize=10)
                plt.ylabel(y_axis, fontsize=10)

                st.pyplot(fig)

if __name__ == '__main__':
    main()




    