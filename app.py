import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
np.bool = np.bool_



st.title('Visualization: Final Project')

# Load the data
data = pd.read_csv('songs_normalize.csv')


# def second_vis(data, key_suffix=""):
#     # Preprocess the data
#     data = data.copy()
#     la = LabelEncoder()
#     label = la.fit_transform(data["genre"])
#     data["genre"] = label
#     data.drop(["artist", "song"], axis=1, inplace=True)
#     # Split the data into train and test sets
#     x = data.drop(["popularity"], axis=1)
#     y = data["popularity"]
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
#
#     # Train the model
#     model = DecisionTreeRegressor()
#     model.fit(x_train, y_train)
#
#     # Extract feature names from the original dataset or use x.columns
#     feature_names = x.columns.tolist()
#     # Get unique popularity values
#     popularity_values = np.sort(data['popularity'].unique())
#
#     # Create the SHAP explainer using TreeExplainer
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(x, check_additivity=False)
#
#     # Calculate average SHAP values for each feature filtered by popularity
#     shap_values_avg = []
#     for popularity in popularity_values:
#         filtered_indices = data[data['popularity'] == popularity].index
#         filtered_shap_values = shap_values[filtered_indices]
#         shap_values_avg.append(np.mean(filtered_shap_values, axis=0))
#     shap_df_avg = pd.DataFrame(shap_values_avg, columns=feature_names)
#
#     # Select the popularity value using Streamlit's selectbox
#     popularity_dropdown = st.selectbox("Popularity:", popularity_values, index=0, key='popularity_dropdown' + key_suffix)
#
#     # Update the graph based on the selected popularity
#     popularity_index = np.where(popularity_values == popularity_dropdown)[0][0]
#     shap_values = shap_df_avg.iloc[popularity_index]
#     sorted_indices = shap_values.abs().argsort()[::-1]
#     sorted_shap_vals = shap_values[sorted_indices]
#     sorted_feature_names = np.array(feature_names)[sorted_indices]
#
#     # Create a custom color map for the bars
#     colors = ['lightcoral' if val < 0 else 'limegreen' for val in sorted_shap_vals]
#
#     # Create the graph using Streamlit's pyplot and beta_columns
#     col1, col2 = st.columns([1, 2])
#
#     with col1:
#         # Create the graph
#         fig, ax = plt.subplots(figsize=(50,50))  # Adjust the figsize as desired
#         ax.invert_yaxis()  # Invert the y-axis to have the highest feature at the top
#         ax.barh(sorted_feature_names, sorted_shap_vals, color=colors, alpha=0.8)
#         ax.set_xlabel("SHAP Value")
#         ax.set_ylabel("Feature")
#         ax.set_title(f"SHAP Values for Popularity {popularity_dropdown}")
#         plt.tight_layout()
#         st.pyplot(fig)
#
#     with col2:
#         # Display the popularity dropdown
#         popularity_dropdown

def first_vis(data):
    songs_normalize = data.copy()
    songs_normalize = songs_normalize.drop(['explicit', 'genre'], axis=1)

    scaler = MinMaxScaler()
    songs_normalize[songs_normalize.columns.difference(['artist', 'song', 'year', 'explicit'])] = scaler.fit_transform(
        songs_normalize[songs_normalize.columns.difference(['artist', 'song', 'year', 'explicit'])])

    # Get the columns names and save only the relevant ones
    column_names = list(songs_normalize.columns.values)
    features_to_remove = ['song', 'explicit', 'artist', 'year', 'popularity']
    features_names = [item for item in column_names if item not in features_to_remove]

    # Convert non-numeric columns to numeric
    non_numeric_columns = songs_normalize.select_dtypes(exclude=np.number).columns
    songs_normalize[non_numeric_columns] = songs_normalize[non_numeric_columns].apply(pd.to_numeric, errors='coerce')

    avg_popularity = songs_normalize.groupby(['year'], as_index=False)[features_names].mean()

    # Create the lines for the plot
    lines = []
    for column in avg_popularity.columns:
        if column != 'year':
            line = go.Scatter(x=avg_popularity['year'], y=avg_popularity[column], name=column)
            lines.append(line)

    # Create the layout with checklist dropdown
    layout = go.Layout(
        title='Average Feature Value per Year',
        title_x=0.3,  # Set the title position to the center
        title_y=0.9,  # Set the title position to the upper part
        xaxis_title='Year',
        yaxis_title='Average Value',
        legend=dict(
            title='Choose Features',
            title_font=dict(size=16),
        ),
        annotations=[
            dict(
                x=1.25,
                y=-0.05,  # Adjust the y-coordinate to position the note below the legend
                xref='paper',
                yref='paper',
                text='One click to remove the feature',
                showarrow=False,
                font=dict(size=10),
            )
        ],
        updatemenus=[  # the user can choose to see all features in one click
            dict(
                buttons=list([
                    dict(
                        label='All',
                        method='update',
                        args=[{'visible': [True] * len(lines)}, {'title': 'Average Feature Value per Year'}]
                    )
                ]),
                direction='down',  # the position of the dropdown
                showactive=True,
                x=1.1,
                xanchor='right',
                y=1.15,
                yanchor='top'
            )
        ]
    )

    # Create the figure
    fig = go.Figure(data=lines, layout=layout)

    # Display the figure
    st.plotly_chart(fig)

def second_vis(data, key_suffix=""):
    # Preprocess the data
    data = data.copy()
    la = LabelEncoder()
    label = la.fit_transform(data["genre"])
    data["genre"] = label
    data.drop(["artist", "song"], axis=1, inplace=True)
    # Split the data into train and test sets
    x = data.drop(["popularity"], axis=1)
    y = data["popularity"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

    # Train the model
    model = DecisionTreeRegressor()
    model.fit(x_train, y_train)

    # Extract feature names from the original dataset or use x.columns
    feature_names = x.columns.tolist()
    # Get unique popularity values
    popularity_values = np.sort(data['popularity'].unique())

    # Create the SHAP explainer using TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x, check_additivity=False)

    # Calculate average SHAP values for each feature filtered by popularity
    shap_values_avg = []
    for popularity in popularity_values:
        filtered_indices = data[data['popularity'] == popularity].index
        filtered_shap_values = shap_values[filtered_indices]
        shap_values_avg.append(np.mean(filtered_shap_values, axis=0))
    shap_df_avg = pd.DataFrame(shap_values_avg, columns=feature_names)

    # Select the popularity value using Streamlit's selectbox
    popularity_dropdown = st.selectbox("Popularity:", popularity_values, index=0, key='popularity_dropdown' + key_suffix)

    # Update the graph based on the selected popularity
    popularity_index = np.where(popularity_values == popularity_dropdown)[0][0]
    shap_values = shap_df_avg.iloc[popularity_index]
    sorted_indices = shap_values.abs().argsort()[::-1]
    sorted_shap_vals = shap_values[sorted_indices]
    sorted_feature_names = np.array(feature_names)[sorted_indices]

    # Create a custom color map for the bars
    colors = ['lightcoral' if val < 0 else 'limegreen' for val in sorted_shap_vals]

    # Set a fixed size for the graph
    fig, ax = plt.subplots(figsize=(45, 45))

    # Create the graph using Streamlit's pyplot and beta_columns
    ax.invert_yaxis()  # Invert the y-axis to have the highest feature at the top
    ax.barh(sorted_feature_names, sorted_shap_vals, color=colors, alpha=0.8)
    ax.set_xlabel("SHAP Value", fontsize=85)
    ax.set_ylabel("Feature", fontsize=85)
    ax.set_title(f"SHAP Values for Popularity {popularity_dropdown}", fontsize=90)
    ax.tick_params(axis='both', which='major', labelsize=80)
    st.pyplot(plt.gcf())

    return fig


def display_side_by_side(data):
    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)

    # Display the first graph in the first column
    with col1:
        fig1 = second_vis(data, key_suffix="_1")
        st.pyplot(fig1)

    # Display the second graph in the second column
    with col2:
        fig2 = second_vis(data, key_suffix="_2")
        st.pyplot(fig2)


# Set the default Plotly renderer to be 'iframe' for better rendering in Streamlit
pio.renderers.default = 'iframe'


# Call the function to run the Streamlit app
first_vis(data)
display_side_by_side(data)
