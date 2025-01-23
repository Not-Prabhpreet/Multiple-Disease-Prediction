import os
import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Data Analysis and Prediction", layout="wide", page_icon="📊")

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stPlotlyChart {
        width: 100%;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load the models
diabetes_model = pickle.load(open("Diabetes/Diabetes Prediction.sav", 'rb'))
parkinson_model = pickle.load(open("Parkinson's Disease/parkinsons_model.sav", 'rb'))
heart_disease_model = pickle.load(open("Heart Disease/heart_disease_model.sav", 'rb'))

# Function for the main logic of the app
def main():
    # Sidebar for navigation
   # Sidebar for navigation
    with st.sidebar:
        selected = option_menu(
            'Multiple Disease Prediction and Data Visualization System',
            ['Data Visualizer', 'Diabetes Prediction', 'Heart Disease Prediction', "Parkinson's Prediction"],
            icons=['bar-chart', 'activity', 'heart', 'person'],
            default_index=0
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
        folder_path = os.path.join(working_dir, 'datasets')
        files_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

        # File selection with additional info
        selected_file = st.selectbox("Select a file", files_list, index=None)

        if selected_file:
            # Cache the data loading
            @st.cache_data
            def load_data(filepath):
                return pd.read_csv(filepath)
            
            # Read the CSV
            file_path = os.path.join(folder_path, selected_file)
            df = load_data(file_path)
            columns = df.columns.tolist()

            # Dataset Overview Metrics
            st.header("📊 Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", f"{len(df):,}")
            with col2:
                st.metric("Total Features", len(df.columns))
            with col3:
                missing_values = df.isna().sum().sum()
                st.metric("Missing Values", f"{missing_values:,}")
            with col4:
                memory_usage = df.memory_usage().sum() / 1024**2
                st.metric("Memory Usage", f"{memory_usage:.2f} MB")

            # Create tabs for different analysis views
           # Modified tab section with enhanced features
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Data Overview",
                "📊 Statistical Analysis",
                "🔍 Feature Relations",
                "🎨 Custom Visualizations"
            ])

            with tab1:
                st.subheader("Quick Data Preview")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(df.head(), use_container_width=True)
                with col2:
                    # Enhanced data info with better formatting
                    dtype_df = pd.DataFrame({
                        'Column': df.dtypes.index,
                        'Type': df.dtypes.values,
                        'Non-Null': df.count().values,
                        'Missing': df.isna().sum().values,
                        'Unique Values': [df[col].nunique() for col in df.columns]
                    })
                    st.dataframe(dtype_df, use_container_width=True)

            with tab2:
                st.subheader("Distribution Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Numeric Analysis with enhanced visualizations
                    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                    if len(numeric_cols) > 0:
                        selected_num_col = st.selectbox("Select Numeric Feature", numeric_cols)
                        
                        # Distribution plot with both histogram and box plot
                        fig = px.histogram(df, x=selected_num_col, 
                                         marginal="box",  # adds boxplot at margin
                                         color_discrete_sequence=['#1f77b4'],
                                         title=f'Distribution of {selected_num_col}')
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Categorical Analysis with pie charts
                    cat_cols = df.select_dtypes(include=['object']).columns
                    if len(cat_cols) > 0:
                        selected_cat_col = st.selectbox("Select Categorical Feature", cat_cols)
                        
                        # Enhanced pie chart with better styling
                        value_counts = df[selected_cat_col].value_counts()
                        fig = px.pie(values=value_counts.values,
                                   names=value_counts.index,
                                   title=f'Distribution of {selected_cat_col}',
                                   hole=0.4)  # Makes it a donut chart
                        fig.update_traces(textposition='outside', textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)

            with tab3:
                st.subheader("Feature Relationships")
                
                # Enhanced correlation heatmap
                numeric_df = df.select_dtypes(include=['float64', 'int64'])
                if not numeric_df.empty and len(numeric_df.columns) > 1:
                    fig = px.imshow(numeric_df.corr(),
                                  title="Feature Correlation Heatmap",
                                  color_continuous_scale='RdBu_r',
                                  aspect='auto')
                    fig.update_traces(text=numeric_df.corr().round(2), texttemplate='%{text}')
                    st.plotly_chart(fig, use_container_width=True)

                    # Add feature comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        x_feature = st.selectbox("Select X-axis feature", numeric_df.columns)
                    with col2:
                        y_feature = st.selectbox("Select Y-axis feature", numeric_df.columns)
                    
                    fig = px.scatter(df, x=x_feature, y=y_feature,
                                   trendline="ols",
                                   title=f'Relationship between {x_feature} and {y_feature}')
                    st.plotly_chart(fig, use_container_width=True)

            with tab4:
                st.subheader("Custom Visualizations")
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    plot_type = st.selectbox(
                        "Select Plot Type",
                        ["Scatter Plot", "Line Plot", "Bar Plot", "Box Plot", 
                         "Violin Plot", "3D Scatter Plot", "Pie Chart", 
                         "Bubble Plot", "Area Plot", "Density Heatmap"]
                    )
                
                # Axis selection based on plot type
                if plot_type != "Pie Chart":
                    col_x, col_y = st.columns(2)
                    with col_x:
                        x_axis = st.selectbox("Select X-axis", columns)
                    with col_y:
                        y_axis = st.selectbox("Select Y-axis", columns)
                    
                    # Add z-axis for 3D plot
                    if plot_type == "3D Scatter Plot":
                        z_axis = st.selectbox("Select Z-axis", columns)
                else:
                    values_col = st.selectbox("Select Values Column", 
                                            df.select_dtypes(include=['float64', 'int64']).columns)
                    names_col = st.selectbox("Select Names Column", 
                                           df.select_dtypes(include=['object']).columns)

                # Advanced styling options
                st.write("### Styling Options")
                style_col1, style_col2 = st.columns(2)
                
                with style_col1:
                    # Color selection
                    color_method = st.radio("Color Selection Method", 
                                          ["Color Picker", "RGB Values", "Column Color"])
                    
                    if color_method == "Color Picker":
                        color = st.color_picker("Pick a Color", "#1f77b4")
                    elif color_method == "RGB Values":
                        r = st.slider("R", 0, 255, 31)
                        g = st.slider("G", 0, 255, 119)
                        b = st.slider("B", 0, 255, 180)
                        color = f"rgb({r},{g},{b})"
                    else:
                        color_column = st.selectbox("Color by column", ["None"] + list(df.columns))
                        color = color_column if color_column != "None" else None

                with style_col2:
                    # Additional styling options
                    if plot_type in ["Line Plot", "Scatter Plot"]:
                        marker_size = st.slider("Marker Size", 1, 20, 8)
                        opacity = st.slider("Opacity", 0.0, 1.0, 0.7)
                    
                    if plot_type == "Bar Plot":
                        orientation = st.selectbox("Orientation", ["vertical", "horizontal"])
                        barmode = st.selectbox("Bar Mode", ["group", "stack"])
                    
                    if plot_type in ["Pie Chart"]:
                        donut = st.checkbox("Donut Chart", value=False)
                        if donut:
                            hole_size = st.slider("Hole Size", 0.1, 0.9, 0.5)

                # Generate plot button
                if st.button("Generate Plot"):
                    try:
                        if plot_type == "Scatter Plot":
                            fig = px.scatter(df, x=x_axis, y=y_axis,
                                           color=color if color_method == "Column Color" else None,
                                           color_discrete_sequence=[color] if color_method != "Column Color" else None,
                                           opacity=opacity,
                                           title=f"Scatter Plot: {y_axis} vs {x_axis}")
                            fig.update_traces(marker=dict(size=marker_size))

                        elif plot_type == "Line Plot":
                            fig = px.line(df, x=x_axis, y=y_axis,
                                        color=color if color_method == "Column Color" else None,
                                        color_discrete_sequence=[color] if color_method != "Column Color" else None,
                                        title=f"Line Plot: {y_axis} vs {x_axis}")
                            fig.update_traces(line=dict(width=marker_size))

                        elif plot_type == "Bar Plot":
                            fig = px.bar(df, x=x_axis if orientation == "vertical" else y_axis,
                                       y=y_axis if orientation == "vertical" else x_axis,
                                       color=color if color_method == "Column Color" else None,
                                       color_discrete_sequence=[color] if color_method != "Column Color" else None,
                                       barmode=barmode,
                                       orientation="v" if orientation == "vertical" else "h",
                                       title=f"Bar Plot: {y_axis} vs {x_axis}")

                        elif plot_type == "Box Plot":
                            fig = px.box(df, x=x_axis, y=y_axis,
                                       color=color if color_method == "Column Color" else None,
                                       color_discrete_sequence=[color] if color_method != "Column Color" else None,
                                       title=f"Box Plot: {y_axis} by {x_axis}")

                        elif plot_type == "Violin Plot":
                            fig = px.violin(df, x=x_axis, y=y_axis,
                                          color=color if color_method == "Column Color" else None,
                                          color_discrete_sequence=[color] if color_method != "Column Color" else None,
                                          title=f"Violin Plot: {y_axis} by {x_axis}")

                        elif plot_type == "3D Scatter Plot":
                            fig = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis,
                                              color=color if color_method == "Column Color" else None,
                                              color_discrete_sequence=[color] if color_method != "Column Color" else None,
                                              title=f"3D Scatter Plot")

                        elif plot_type == "Pie Chart":
                            fig = px.pie(df, values=values_col, names=names_col,
                                       color_discrete_sequence=[color] if color_method != "Column Color" else None,
                                       hole=hole_size if donut else None,
                                       title=f"Pie Chart of {values_col} by {names_col}")

                        elif plot_type == "Bubble Plot":
                            size_col = st.selectbox("Select Size Column", 
                                                  df.select_dtypes(include=['float64', 'int64']).columns)
                            fig = px.scatter(df, x=x_axis, y=y_axis,
                                           size=size_col,
                                           color=color if color_method == "Column Color" else None,
                                           color_discrete_sequence=[color] if color_method != "Column Color" else None,
                                           title=f"Bubble Plot: {y_axis} vs {x_axis}")

                        elif plot_type == "Density Heatmap":
                            fig = px.density_heatmap(df, x=x_axis, y=y_axis,
                                                   color_continuous_scale='RdBu_r',
                                                   title=f"Density Heatmap: {y_axis} vs {x_axis}")

                        # Update layout for all plots
                        fig.update_layout(
                            plot_bgcolor='white',
                            title_x=0.5,
                            width=800,
                            height=600
                        )
                        
                        # Display the plot
                        st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error generating plot: {str(e)}")
                        st.error("Please check if your selected columns and plot type are compatible.")
if __name__ == '__main__':
    main()

    