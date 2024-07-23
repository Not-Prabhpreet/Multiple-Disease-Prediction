# Multiple Disease Prediction and Data Visualization

This project is a comprehensive Streamlit web application designed to predict multiple diseases (Diabetes, Heart Disease, and Parkinson's Disease) using machine learning algorithms(Support Vector Machine and Logistic Regression) and to visualize health datasets. 

## Live Demo

Try it out [Multiple Disease Prediction and Data Visualization](https://multiple-disease-prediction-2qhhx84xr6gnb6u7n4e5sz.streamlit.app/).


## Features

- **Diabetes Prediction**: Utilizes a Support Vector Classifier (SVC) to predict the likelihood of diabetes based on user input.
- **Heart Disease Prediction**: Employs a Logistic Regression model to predict the likelihood of heart disease.
- **Parkinson's Disease Prediction**: Uses a Random Forest Classifier and Support Vector Classifier(SVC) to predict the likelihood of Parkinson's disease.
- **Data Visualizer**: Allows users to upload datasets and visualize them using various types of plots, enhancing data analysis capabilities.

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Machine Learning Libraries**: scikit-learn, pandas, numpy
- **Visualization Libraries**: matplotlib, seaborn
- **Model Persistence**: pickle


## Folder Structure
<pre>
Multiple Disease Prediction/
│
├── app.py
├── requirements.txt
├── datasets/
│   ├── diabetes.csv
│   ├── heart_disease.csv
│   └── parkinsons.csv
├── Diabetes/
│   └── Diabetes Prediction.sav
    └── diabetes_prediction
├── Parkinson's Disease/
│   └── parkinsons_model.sav
    └── parkinsons_disease_prediction
└── Heart Disease/
    └── heart_disease_model.sav
    └── heart_disease_prediction

</pre>
## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Not-Prabhpreet/Multiple-Disease-Prediction.git
   cd Multiple-Disease-Prediction
2. **Install the required packages**:

   <pre><code>pip install -r requirements.txt</code></pre>

3. **Run the application**:

   <pre><code>streamlit run app.py</code></pre>
   

## Usage

-Navigate to the desired prediction or data visualization page using the sidebar.
-Fill in the required inputs for disease prediction or upload a dataset for visualization.
-Click the button to get the prediction or generate the plot.

## License

This project is licensed under the MIT License

