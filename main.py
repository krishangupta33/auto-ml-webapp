from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import ydata_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 



with st.sidebar:
    st.image("https://developer.apple.com/assets/elements/icons/create-ml/create-ml-96x96_2x.png", width=200)
    st.title("Sidebar")
    choice=st.radio("Navigation",["Upload","Profiling","ML","Download"])

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset in csv format")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    #profile df
    profile_df = pd.read_csv('dataset.csv')
    profile = ydata_profiling.ProfileReport(profile_df)

    st_profile_report(profile)

if choice == "ML": 
    df = pd.read_csv('dataset.csv')
    non_numeric_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = df.drop(non_numeric_columns, axis=1)
    
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        setup(df, target=chosen_target)
        # setup_df = pull()
        # st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')

if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")

