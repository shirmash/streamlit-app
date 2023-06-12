import streamlit as st
import requests

# Fetch the code from your GitHub repository
response = requests.get('VisualizationProject.ipynb')
code = response.text

# Execute the code
exec(code)


# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler



# def build_st_query_for_bar_charts(df: pd.DataFrame, header_title: str, fake_value: str):
#     st.subheader('Compare most popular names by year')
#     years_list = df.columns.unique().tolist()[1:]
#     selected_years = st.selectbox(f"Select {header_title}", [fake_value] + years_list)

#     return (years_list, selected_years)

# """
# songs_normalize = pd.read_csv("/content/songs_normalize.csv")
# songs_normalize
