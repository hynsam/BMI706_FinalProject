import altair as alt
import pandas as pd
import streamlit as st


@st.cache
def load_data():
    df = pd.read_csv("processed_data.csv")
    return df

df = load_data()

st.write("## Correlation between lifestyle-related features and health outcomes")

