import streamlit as st
import requests
st.title('Visualization of information - Final Project')


# # import streamlit as st
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import numpy as np
# # from sklearn.preprocessing import MinMaxScaler


# def first_vis(df: pd.DataFrame, header_title: str, fake_value: str):
#     st.subheader('Comparison of Jewish & Muslim (boys and girls) names over the years'))
#     years_list = df.columns.unique().tolist()[1:]
#     selected_years = st.selectbox(f"Select {header_title}", [fake_value] + years_list)

#     return (years_list, selected_years)

# """
# songs_normalize = pd.read_csv("/content/songs_normalize.csv")
# songs_normalize
