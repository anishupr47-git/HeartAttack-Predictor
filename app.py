import streamlit as st
import pandas as pd
import numpy as np
import os
import time

from src src.cleaning import absolute_preprocessing_pipeline, generate_eda_stats
from src.architectures import master_training_orchestrator
from src.visualizations import plot_confusion_matrix, plot_metrics_comparison, plot_eda_target_distribution
from src.explainability import generate_shap_explanation

#now we use st to turn in into a running web
#we setup for it to run now

#use setpageconfig for the layout
st.set_page_config(page_title="Multi-Model Prediction System", layout="wide", initial_sidebar_state="expanded")

#st sidebar creates a panel on the left side of the screen
st.sidebar.title("System Control Panel")
st.sidebar.markdown("Navigate through multi-model suite")

#cached initialization
#use decorator 
#use st.cache_resource so the entire code doesnt rerun again and again
@st.cache_resource(show_spinner=False)
def initialize_system_v2():
    #1Define where our data lives
    filepath = 'heart_disease_comprehensive_dataset.xlsx'

    #check if the file exists
    if not os.path.exists(filepath):
        st.error(f"Dataset not found at {filepath}. Please ensure the file exists")
        st.stop()

    #use st.spinner to show a loading msg
    with st.spinner("Step 1/3: Running Data Profiling"):
        eda_stats= generate_eda_stats(filepath)
        time.sleep(0.5)

    with st.spinner("Step 2/3: Execute Preprocessing"):
        X_scaled, y, scaler, feature_names, ui_config, cat_cols = absolute_preprocessing_pipeline(filepath)

    with st.spinner("Step 3/3: Training 12 Architecutres"):
        #we pass our cleaned and actual data to master
        #it trains
        models, metrics, X_train, X_val = master_training_orchestrator(X_scaled, y)

    return eda_stats, scaler, feature_names, ui_config, cat_cols, models, metrics, X_train

#call the functions here because of the cache it only runs once
eda_stats, scaler, feature_names, ui_config, cat_cols, models, metrics, X_train = initialize_system_v2()

#main user interface
st.title("Multi-Model Heart Disease Studio")
st.markdown("An advanced 12-architectures machine learning system to predict heart-attack chances")

#TABS
tab1, tab2, tab3 = st.tabs([
    "Data Exploration(EDA)",
    "Models & Metrics Hub",
    "Predicition Studio"
])

#tab 1
with tab1:
    st.header("Dataset Overview")
    st.markdown("Raw Data profiles loaded prior to absolute preprocessing")

    #st.colims split the screen vertically
    #here we create 3 equal-width columns

    #st.metric is a great way to display a big number with a label