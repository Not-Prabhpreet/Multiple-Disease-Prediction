import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# Set wide layout and page config
st.set_page_config(
    page_title="Advanced Data Analytics Dashboard",
    layout="wide",
    page_icon="📊"
)

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

# Title with custom formatting
st.title("📊 Advanced Data Analytics Dashboard")
st.markdown("---")

# Getting the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(working_dir, 'datasets')
files_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# File selection with additional info
st.sidebar.header("📁 File Selection")
selected_file = st.sidebar.selectbox(
    "Choose a CSV file",
    files_list,
    index=None,
    help="Select a CSV file to analyze"
)

if selected_file:
    # Load and cache data
    @st.cache_data
    def load_data(file_path):
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None

    file_path = os.path.join(folder_path, selected_file)
    df = load_data(file_path)

    if df is not None:
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
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Data Overview",
            "📊 Statistical Analysis",
            "🔍 Feature Relations",
            "🎨 Custom Visualizations"
        ])

        with tab1:
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            st.subheader("Data Information")
            col1, col2 = st.columns(2)
            
            with col1:
                # Data types info
                dtype_df = pd.DataFrame({
                    'Column': df.dtypes.index,
                    'Type': df.dtypes.values,
                    'Non-Null Count': df.count().values,
                    'Null Count': df.isna().sum().values
                })
                st.dataframe(dtype_df, use_container_width=True)
            
            with col2:
                # Basic statistics
                st.dataframe(df.describe(), use_container_width=True)
            with tab2:
                st.subheader("Statistical Analysis")
                col1, col2 = st.columns(2)
            
            with col1:
                # Numeric columns analysis
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                if len(numeric_cols) > 0:
                    st.write("#### Numeric Features Distribution")
                    selected_num_col = st.selectbox("Select Numeric Column", numeric_cols)
                    
                    # Distribution plot with both histogram and KDE
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=df[selected_num_col],
                        name='Histogram',
                        nbinsx=30,
                        histnorm='probability'
                    ))
                    
                    # Add statistics annotations
                    stats_text = (
                        f"Mean: {df[selected_num_col].mean():.2f}<br>"
                        f"Median: {df[selected_num_col].median():.2f}<br>"
                        f"Std Dev: {df[selected_num_col].std():.2f}<br>"
                        f"Skewness: {df[selected_num_col].skew():.2f}"
                    )
                    fig.add_annotation(
                        x=0.95, y=0.95,
                        xref="paper", yref="paper",
                        text=stats_text,
                        showarrow=False,
                        align="right"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Categorical columns analysis
                cat_cols = df.select_dtypes(include=['object']).columns
                if len(cat_cols) > 0:
                    st.write("#### Categorical Features Distribution")
                    selected_cat_col = st.selectbox("Select Categorical Column", cat_cols)
                    
                    # Value counts and pie chart
                    value_counts = df[selected_cat_col].value_counts()
                    fig = px.pie(
                        values=value_counts.values,
                        names=value_counts.index,
                        title=f'Distribution of {selected_cat_col}'
                    )
                    st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("Feature Relations")
            
            # Correlation analysis for numeric features
            numeric_df = df.select_dtypes(include=['float64', 'int64'])
            if not numeric_df.empty and len(numeric_df.columns) > 1:
                st.write("#### Correlation Heatmap")
                corr_matrix = numeric_df.corr()
                fig = px.imshow(
                    corr_matrix,
                    title="Feature Correlation Heatmap",
                    color_continuous_scale='RdBu_r',
                    aspect='auto'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Scatter plot for selected features
                st.write("#### Feature Relationships")
                col1, col2 = st.columns(2)
                with col1:
                    x_feature = st.selectbox("Select X-axis feature", numeric_df.columns)
                with col2:
                    y_feature = st.selectbox("Select Y-axis feature", numeric_df.columns)
                
                fig = px.scatter(
                    df,
                    x=x_feature,
                    y=y_feature,
                    title=f'{x_feature} vs {y_feature}',
                    trendline="ols"
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.subheader("Custom Visualizations")
            
            # Advanced plotting options
            plot_type = st.selectbox(
                "Select Plot Type",
                ["Scatter Plot", "Line Plot", "Bar Plot", "Box Plot", "Violin Plot", "3D Scatter Plot"]
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                x_axis = st.selectbox("Select X-axis", df.columns)
            with col2:
                y_axis = st.selectbox("Select Y-axis", df.columns)
            with col3:
                if plot_type == "3D Scatter Plot":
                    z_axis = st.selectbox("Select Z-axis", df.columns)
                color_by = st.selectbox("Color by", ["None"] + list(df.columns))

            # Create the selected plot
            if st.button("Generate Plot"):
                if plot_type == "Scatter Plot":
                    fig = px.scatter(df, x=x_axis, y=y_axis, 
                                   color=None if color_by == "None" else color_by)
                elif plot_type == "Line Plot":
                    fig = px.line(df, x=x_axis, y=y_axis, 
                                color=None if color_by == "None" else color_by)
                elif plot_type == "Bar Plot":
                    fig = px.bar(df, x=x_axis, y=y_axis, 
                               color=None if color_by == "None" else color_by)
                elif plot_type == "Box Plot":
                    fig = px.box(df, x=x_axis, y=y_axis, 
                               color=None if color_by == "None" else color_by)
                elif plot_type == "Violin Plot":
                    fig = px.violin(df, x=x_axis, y=y_axis, 
                                  color=None if color_by == "None" else color_by)
                elif plot_type == "3D Scatter Plot":
                    fig = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis, 
                                      color=None if color_by == "None" else color_by)
                
                fig.update_layout(title=f"{plot_type} of {y_axis} vs {x_axis}")
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Please upload a valid CSV file to begin analysis.")