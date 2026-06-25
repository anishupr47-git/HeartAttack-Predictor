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
    col1, col2, col3 = st.columns(3)

    #display big num with label
    col1.metric("Total Rows", eda_stats['total_rows'])
    col2.metric("Total Columns", eda_stats['total_cols'])
    col3.metric("Missing Data", len(eda_stats['missing_columns']))

    st.divider()

    col_a, col_b  = st.columns([1,1])
    with col_a:
        st.subheader("Raw Data Sample")
        st.dataframe(eda_stats['raw_df'])

        st.download_button(
            label="Download Full Dataset(Excel)",
            data=open('heart_disease_comprehensive_dataset.xlsx', 'rb').read(),
            file_name='heart_disease_dataset.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

        with col_b:
            st.subheader("Target Distribution")
            fig_pie= plot_eda_target_distribution(eda_stats['target_distribution'])
            st.plotly_chart(fig_pie, use_container_width=True)

    #tab 2
    with tab2:
        st.header("Architecture Evaluation")
        st.markdown("Validation metrics computed across all different 12 models")

        #we display our massive bar chart compiling all models
        fig_metrics = plot_metrics_comparison(metrics)
        st.plotly_chart(fig_metrics, use_container_width=True)

        st.divider()

        st.subheader("Interactive CM")
        st.markdown("Check every true and false details")

        cm_cols = st.columns(3)
        model_names = list(models.keys())

        #we use enamurate for both idx and name
        for idx, m_name in enumerate(model_names):
            #use moduloo to cycle through the 3 columns
            col = cm_cols[idx%3]
            with col:
                st.markdown(f"{m_name}")
                m_metrics = metrics[m_name]
                #use caption
                st.caption(f"ACC: {m_metrics['Accuracy']:.3f} | AUC: {m_metrics['ROC-AUC']:.3f}")

                #fetch cm
                cm = m_metrics['Confusion_Matrix']
                #create the plotly
                fig_cm = plot_confusion_matrix(cm, m_name)
                st.plotly_chart(fig_cm, use_container_width=True)

#tab 3 prediciton studio
with tab3:
    st.header("Prediction Studio")
    st.markdown("Live inference using all 12 architectures")

    st.subheader("1. Patient Profile")

    input_data = {}

    input_cols = st.columns(4)
    raw_features = list(ui_config.keys())

    for idx, features in enumerate(raw_features):
        col= input_cols[idx%4]
        bound = ui_config[feature]

        with col:
            #if it is categorical
            if bound['type'] == 'categorical':
                opts = bound['options']
                mode_val= bound['mode']

                #keep all value to 0
                default_idx = 0

                input_data[feature] = st.number_input(
                    label=feature,
                    min_value=bound['min'],
                    max_value=bound['max']
                    value=start_val,
                    key=f"input_{feature}"
                )

    st.divider()


    st.subheader("Execute Inference and Explanability")
    col_sel, col_btn = st.columns([1,2])
    with col_sel:
        #let user pick
        selected_model_name = st.selectbox("Select Target Architecture", options=model_names)

    with col_btn:
        st.write("")
        st.write("")
        run_pred = st.button("Run Prediciton", use_container_width=True)


    #if it becomes true
    if run_pred:
        #st.status creates a nice animated box showing process
        with st.status("Executing Multi-Status Pipeline") as status:
            st.write("Constructing feature vector")
            raw_input_df = pd.DataFrame([input_data])

            st.write("Encoding categorical variable natively")
            #use one hot encoding
            encoded_input_df = pd.get_dummies(raw_input_df, columns=cat_cols)
            #align all
            encoded_input_df= encoded_input_df.reindex(columns=feature_names, fill_value=0)

            st.write("Applying Standard Transformation")
            input_scaled = scaler.transform(encoded_input_df.values)

            st.write(f"Querying {selected_model_name}")
            model = models[selected_model_name]

            #predict
            if selected_model_name == 'DNN':
                prob = model.predict(input_scaled)[0][0]
                pred = int(prob > 0.5)
            elif hasattr(model, 'predict_proba'):
                prob_arr = model.predict_proba(input_scaled)
                if prob_arr.shape[1] > 1:
                    prob = prob_arr[0][1]
                else:
                    prob = float(model.predict(input_scaled)[0])
                pred = int(model.predict(input_scaled)[0])
            else:
                #fallnacl for models
                pred= int(models.predict(input_scaled)[0])
                prob=float(pred)

            st.write("Generating SHAP explainibility")
            try:
                fig_shap = generate_shap_explanation(model, X_train, input_scaled, feature_names, selected_model_name)
                shap_success = True
            except Exception as e:
                shap_success = False
                shap_err = str(e)
            
            status.update(label="Inference & Explanability Complete", state="complete", expanded=True)

        #outcome
        st.subheader("Prediciton Output")
        if pred == 1:
            st.error(f"High Risk Of Heart Disease Detected Probability:{prob * 100:.2f}%")
        else:
            st.success(f"Low Risk of Heart Disease Probability:{prob *100:.2f}%")

        st.divider()

        #Explainability Presentation
        st.subheader("What's Happening? (Prediction Explainability)")
        if shap_success:
            st.markdown("The chart below shows exactly which features pushed the model's prediction higher (red) or lower (blue) compared to the base baseline expectation.")
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.warning("SHAP Explainability encountered an error for this specific architecture fallback.")
            st.code(shap_err)