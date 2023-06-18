import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import matplotlib.pyplot as plt
import shap
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
from pandas.plotting import parallel_coordinates
np.bool = np.bool_


st.set_page_config(layout="wide",page_title="Spotify Music Insights")

st.title('Visualization: Final Project')

# Load the data
data = pd.read_csv('songs_normalize.csv')

def first_vis(data):
    songs_normalize = data.copy()
    songs_normalize = songs_normalize.drop(['explicit','genre'], axis=1)
    songs_normalize.sort_values('popularity', inplace=True)
    scaler = MinMaxScaler()
    songs_normalize[songs_normalize.columns.difference(['artist','song', 'year','genre','popularity'])] = scaler.fit_transform(songs_normalize[songs_normalize.columns.difference(['artist','song', 'year','genre','popularity'])])

    popularity_ranges = [range(89, 79, -1), range(79, 69, -1), range(69, 59, -1), range(59, 49, -1),
                         range(49, 39, -1), range(39, 29, -1), range(29, 19, -1), range(19, 9, -1), range(9,0,-1)]  
    sorted_popularities = [f"{range.stop}-{range.start - 1}" for range in reversed(popularity_ranges)]
    def map_to_range(value):
        for i, rng in enumerate(popularity_ranges):
          if value == 0:
            return f"{0}-{8}"
          elif value in rng:         
            return f"{rng.stop}-{rng.start-1}"
            
    songs_normalize['PopularityRange'] = songs_normalize['popularity'].apply(map_to_range)
    # Get the columns names and save only the relevant ones
    songs_normalize = songs_normalize.drop('popularity', axis=1)
    
    #get the columns names and save only the relevents
    column_names = list(songs_normalize.columns.values)
    features_to_remove = ['song', 'artist','genre', 'year','PopularityRange']
    features_names = [item for item in column_names if item not in features_to_remove]

    # Convert non-numeric columns to numeric
    # non_numeric_columns = songs_normalize.select_dtypes(exclude=np.number).columns
    # songs_normalize[non_numeric_columns] = songs_normalize[non_numeric_columns].apply(pd.to_numeric, errors='coerce')
    # avg_popularity = songs_normalize.groupby(['year'], as_index=False)[features_names].mean()

    fig = go.Figure()
    # Create the boxes for the plot
    boxes = []
    for column in features_names:
      fig.add_trace(go.Box(y=songs_normalize[column], x=songs_normalize['PopularityRange'], name=column))
  
    # Create dropdown menu options
    dropdown_options = []
    for i, feature in enumerate(features_names):
        visibility = [i == j for j in range(len(features_names))]
        dropdown_options.append({'label': feature, 'method': 'update', 'args': [{'visible': visibility}, {'title': f'{feature} by Popularity Ranges'}]})

    select_feature = st.selectbox('Choose feature:', features_names)
    # Set the initial visible column
    visible_column = [False] * len(features_names)
    visible_column[0] = True  
    # Set the initial visibility of the bars
    for box, visibility in zip(fig.data, visible_column):
        box.visible = visibility

    # Update the layout
    fig.update_layout(
        title='Feature Values by Popularity Ranges',
        title_font=dict(size=20, bold=True),
        xaxis_title='Popularity Ranges',
        yaxis_title='Feature Values',
        title_x=0.35,  # Set the title position to the center
        title_y=0.9,  # Set the title position to the upper part
        showlegend=False,
        updatemenus=[{'buttons': dropdown_options, 'direction': 'down', 'showactive': True, 'x': 1.1, 'xanchor': 'right', 'y': 1.15, 'yanchor': 'top'}]
    )
    # Create the figure
    
    fig.update_traces(line=dict(width=2.5))
    fig.update_layout(
        width=900,  # Set the width of the chart
        height=500,  # Set the height of the chart
    )
    # Display the figure
    col1, col2 = st.columns([1, 16])
    with col1:
        st.write("")
    with col2:
        st.plotly_chart(fig)


def second_vis(data):
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

    # Create the SHAP explainer using TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x, check_additivity=False)

    # Calculate average SHAP values for each feature filtered by popularity
    popularity_values = np.sort(data['popularity'].unique())
    shap_values_avg = []
    for feature in feature_names:
        filtered_shap_values = shap_values[:, feature_names.index(feature)]
        shap_values_avg.append(np.mean(filtered_shap_values, axis=0))
    shap_df_avg = pd.DataFrame(shap_values_avg, index=feature_names, columns=['Average SHAP Value'])

    # Select the feature using Streamlit's selectbox
    feature_dropdown = st.selectbox("Feature:", feature_names)

    # Update the graph based on the selected feature
    shap_values_feature = shap_values[:, feature_names.index(feature_dropdown)]
    popularity_ranges = [range(89, 79, -1), range(79, 69, -1), range(69, 59, -1), range(59, 49, -1),
                         range(49, 39, -1), range(39, 29, -1), range(29, 19, -1), range(19, 9, -1), range(9, 0, -1)]
    sorted_shap_vals = [np.mean(shap_values_feature[np.isin(data['popularity'], range)]) for range in reversed(popularity_ranges)]
    sorted_popularities = [f"{range.stop}-{range.start - 1}" for range in reversed(popularity_ranges)]



    # Create the bar chart using go.Bar and go.Figure
    fig = go.Figure(data=go.Bar(
        y=sorted_popularities,
        x=sorted_shap_vals,
        orientation='h',
        marker=dict(
            color=['lightcoral' if val < 0 else 'limegreen' for val in sorted_shap_vals],
            opacity=0.8
        )
    ))
    fig.update_layout(
        title={
            'text': f"Average SHAP Values by Popularity Range for {feature_dropdown}",
            'y': 0.9,  # Adjust the y-coordinate to center the title
            'x': 0.5,  # Set the x-coordinate to the center of the graph
            'xanchor': 'center',
            'yanchor': 'top'
        },
        yaxis=dict(tickfont=dict(size=10)),
        xaxis=dict(tickfont=dict(size=10)),
        width=900,  # Set the width of the chart
        height=500,
    )
    # Display the graph using st.plotly_chart
    col1, col2 = st.columns([1, 16])
    with col1:
        st.write("")
    with col2:
        st.plotly_chart(fig)


def second_vis_alt(data):
    # Preprocess the data
    data = data.copy()
    la = LabelEncoder()
    label = la.fit_transform(data["genre"])
    data["genre"] = label
    data.drop(["artist", "song"], axis=1, inplace=True)
    # Split the data into train and test sets
    x = data.drop(["popularity", 'year'], axis=1)

    # Extract feature names from the original dataset or use x.columns
    feature_names = x.columns.tolist()

    # Exclude 'year', 'popularity', and 'genre' features from the dropdown menu
    feature_dropdown = st.selectbox("Feature:",
                                    [col for col in feature_names if col not in ['year', 'popularity', 'genre']])

    # Calculate average feature values for each popularity range
    popularity_ranges = [range(89, 79, -1), range(79, 69, -1), range(69, 59, -1), range(59, 49, -1),
                         range(49, 39, -1), range(39, 29, -1), range(29, 19, -1), range(19, 9, -1), range(9, 0, -1)]

    feature_avg_values = []
    for popularity_range in reversed(popularity_ranges):
        avg_value = np.mean(data[data['popularity'].isin(popularity_range)][feature_dropdown])
        feature_avg_values.append(avg_value)

    sorted_popularities = [f"{range.stop}-{range.start - 1}" for range in reversed(popularity_ranges)]

    # Normalize the feature average values between a small positive value and 1
    min_value = np.min(feature_avg_values)
    max_value = np.max(feature_avg_values)
    if min_value != max_value:
        normalized_values = 0.01 + (feature_avg_values - min_value) / (max_value - min_value) * 0.99
    else:
        normalized_values = feature_avg_values

    # Create the bar chart using go.Bar and go.Figure
    fig = go.Figure(data=go.Bar(
        x=normalized_values,
        y=sorted_popularities,
        orientation='h',
        marker=dict(
            color='rgb(63, 81, 181)',  # Specify the bar color
            line=dict(
                color='rgb(40, 55, 71)',  # Specify the bar border color
                width=1.5  # Specify the bar border width
            )
        ),
        opacity=0.8  # Specify the bar opacity
    ))
    fig.update_layout(
        yaxis_title='Popularity range',
        xaxis_title='Average Normalized Value',
        title={
            'text': f"Average Feature Values by Popularity Range for {feature_dropdown}",
            'y': 0.9,  # Adjust the y-coordinate to center the title
            'x': 0.5,  # Set the x-coordinate to the center of the graph
            'xanchor': 'center',
            'yanchor': 'top'
        },
        yaxis=dict(
            tickfont=dict(size=10),
            gridcolor='rgb(238, 238, 238)'  # Specify the grid color
        ),
        xaxis=dict(
            tickfont=dict(size=10),
            gridcolor='rgb(238, 238, 238)'  # Specify the grid color
        ),
        width=900,  # Set the width of the chart
        height=500,
        plot_bgcolor='rgb(255, 255, 255)',  # Specify the plot background color
        paper_bgcolor='rgb(255, 255, 255)',  # Specify the paper background color
    )

    # Display the graph using st.plotly_chart
    col1, col2 = st.columns([1, 16])
    with col1:
        st.write("")
    with col2:
        st.plotly_chart(fig)

# def display_side_by_side(data):
#     # Create two columns for side-by-side display
#     col1, col2 = st.columns(2)
#
#     # Display the first graph in the first column
#     with col1:
#         fig1 = second_vis(data, key_suffix="_1")
#         st.pyplot(fig1)
#
#     # Display the second graph in the second column
#     with col2:
#         fig2 = second_vis(data, key_suffix="_2")
#         st.pyplot(fig2)

# Set the default Plotly renderer to be 'iframe' for better rendering in Streamlit
pio.renderers.default = 'iframe'
def third_vis(data):
    data = data.copy()
    # Drop rows with missing genre values
    data.dropna(subset=['genre'], inplace=True)

    # Split multiple genres into a list
    data['genre'] = data['genre'].str.split(',')

    # Remove leading/trailing whitespace from each genre
    data['genre'] = data['genre'].apply(lambda genres: [genre.strip() for genre in genres])

    # Keep only the first genre for each song
    data['genre'] = data['genre'].apply(lambda genres: genres[0])
    avg_popularity_genre = data.groupby(['year', 'genre'])['popularity'].mean().reset_index()
    avg_popularity_genre = avg_popularity_genre.pivot(index='year', columns='genre', values='popularity')

    # Create the bars for the plot
    bars = []
    genres = []
    for column in avg_popularity_genre.columns:
        if column != 'set()':
            bar = go.Bar(
                x=avg_popularity_genre.index,
                y=avg_popularity_genre[column],
                name=column,
                marker=dict(color='orange', line=dict(color='black', width=1))
            )
            bars.append(bar)
            genres.append(column)

    layout = go.Layout(
        title='Popularity of Different Genres Over the Years',
        barmode='stack',  # Set the barmode to 'stack' for stacked bars
        title_x=0.35,  # Set the title position to the center
        title_y=0.9,  # Set the title position to the upper part
        xaxis_title='Year',
        yaxis_title='Average Popularity',
        showlegend=False,
    )

    # Create the initial selectbox
    select_genre = st.selectbox('Choose genre:', genres)

    # Set the visibility of the bars based on the selected genre
    visible_column = [column == select_genre for column in genres]
    for bar, visibility in zip(bars, visible_column):
        bar.visible = visibility

    # Create the figure
    fig = go.Figure(data=bars, layout=layout)
    fig.update_layout(
        width=900,  # Set the width of the chart
        height=500,  # Set the height of the chart
    )

    # Display the figure in Streamlit
    col1, col2 = st.columns([1,16])
    with col1:
        st.write("")
    with col2:
        st.plotly_chart(fig)

        
def second_vis_alt1(data):
    # Preprocess the data
    data = data.copy()
    la = LabelEncoder()
    label = la.fit_transform(data["genre"])
    data["genre"] = label
    data.drop(["artist", "song"], axis=1, inplace=True)
    # Split the data into train and test sets
    x = data.drop(["popularity", 'year'], axis=1)

    # Extract feature names from the original dataset or use x.columns
    feature_names = x.columns.tolist()

    # Exclude 'year', 'popularity', and 'genre' features from the dropdown menu
    feature_dropdown = st.selectbox("Feature1:",
                                    [col for col in feature_names if col not in ['year', 'popularity', 'genre']])

    # Calculate average feature values for each popularity range
    popularity_ranges = [range(89, 79, -1), range(79, 69, -1), range(69, 59, -1), range(59, 49, -1),
                         range(49, 39, -1), range(39, 29, -1), range(29, 19, -1), range(19, 9, -1), range(9, 0, -1)]

    feature_avg_values = []
    for popularity_range in reversed(popularity_ranges):
        avg_value = np.mean(data[data['popularity'].isin(popularity_range)][feature_dropdown])
        feature_avg_values.append(avg_value)

    sorted_popularities = [f"{range.stop}-{range.start - 1}" for range in reversed(popularity_ranges)]

    min_value = np.min(feature_avg_values)
    max_value = np.max(feature_avg_values)
    if min_value != max_value:
        normalized_values = 0.01 + (feature_avg_values - min_value) / (max_value - min_value) * 0.99
    else:
        normalized_values = feature_avg_values

    # Create a DataFrame with the normalized average feature values
    df_avg_values = pd.DataFrame({feature_dropdown: normalized_values, 'Popularity Range': sorted_popularities})

    # Create a radar chart
    categories = df_avg_values['Popularity Range']
    values = df_avg_values[feature_dropdown].tolist()

    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        marker=dict(color='rgb(63, 81, 181)')
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title={
            'text': f"Average Feature Values by Popularity Range for {feature_dropdown}",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )

    # Display the radar chart
    col1, col2 = st.columns([1, 16])
    with col1:
        st.write("")
    with col2:
        st.plotly_chart(fig)
        
st.header('What are the trends and patterns in popular music from 2000 to 2019, based on the Top Hits Spotify dataset?')
st.header("Are there any notable differences between popular songs from different years? ")
st.write("Explore the change in diffrent features in spotify most popular songs over the years. Each line represents the average value of a specific feature over the years. You can select individual features to see their trends over time by clicking on their names in the legend. To see all the features together, simply choose the 'All' option from the dropdown menu. You can also temporarily remove a feature from the graph by clicking on its name.")
first_vis(data)
st.header('What are the characteristics that have the strongest influence on the popularity of a song? ')
st.write("Explore the factors that shape a song's popularity. By selecting different features from the dropdown menu, you can observe how various characteristics influence a song's popularity.")
st.write(" A positive SHAP value suggests that as a feature's value increases, it tends to increase the song's popularity. On the other hand, a negative SHAP value indicates that as a feature's value increases, it may have a diminishing effect on the song's popularity.For instance, take the feature 'duration_ms'  that is shown below as an example. As the duration of the song increases, it may have a negative impact on the song's popularity. ")
second_vis(data)
second_vis_alt(data)
second_vis_alt1(data)
st.header('How has the popularity of different genres changed over time?')
st.write("Explore the popularity of different music genres over the years. The graph displays the average popularity of the selected genre across different years. The height of each bar represents the popularity level, where higher values indicate greater popularity.")
third_vis(data)
