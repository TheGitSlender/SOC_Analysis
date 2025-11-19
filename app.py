# Imports

import pandas as pd
import matplotlib.pyplot as plt
import os
import streamlit as st
from  streamlit_option_menu import option_menu

# Helper functions
from scripts.model_methods import load_data, vis_heatmap, model

def on_button_click():
    st.session_state.button_clicked = True



# Config Variables

dataset_path = "./data/cpu_load_data.csv"
st.set_page_config(page_title="SOC Analysis tool", layout="wide", initial_sidebar_state="collapsed")
data = load_data(dataset_path)




with st.sidebar:
    selected = option_menu(
        menu_title= "SOC Tools",
        options= ["Home", "Data Analysis", "Model"],
        default_index=0
    )


if selected == "Home":
    st.title("SOC Analyst Helper")
    st.write("This work is in the context of an academic project. The objective is to give a user the ability to train a KNN and test different variations of k and test them with Cross validation.")
    st.write("Let's start with some Data Analysis!")
    
elif selected == "Data Analysis":
    if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = False
    st.write("Preview the data")
    st.button("Preview data", on_click=on_button_click)
    if st.session_state.button_clicked:
        st.dataframe(data.head())
    if st.button("Visualize heatmap:"):
        st.pyplot(vis_heatmap(data))

elif selected == "Model":
    st.title("Model Configuration and Training")
    k_value = st.selectbox("Amount of Neighbors for the KNN", [1,5,9,13])
    model = model(k_value)
    