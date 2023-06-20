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
from plotly.subplots import make_subplots
np.bool = np.bool_


st.set_page_config(layout="wide",page_title="Spotify Music Insights")

st.title('Visualization: Final Project')

# Load the data
data = pd.read_csv('songs_normalize.csv')
map_data= pd.read_csv('map_data.csv')
def first_vis(data):
    x='popularity'
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

    fig = go.Figure()
    # Create the boxes for the plot
    boxes = []
    for column in features_names:
      fig.add_trace(go.Box(y=songs_normalize[column], x=songs_normalize['PopularityRange'], name=column))
  
    # # Create dropdown menu options
    # dropdown_options = []
    # for i, feature in enumerate(features_names):
    #     visibility = [i == j for j in range(len(features_names))]
    #     dropdown_options.append({'label': feature, 'method': 'update', 'args': [{'visible': visibility}, {'title': f'{feature} by Popularity Ranges'}]})

    select_feature = st.selectbox('Choose feature:', features_names)

    # Set the visibility of the bars based on the selected genre
    visible_column = [column == select_feature for column in features_names]
    for box, visibility in zip(fig.data, visible_column):
        box.visible = visibility
    
    # Update the layout
    fig.update_layout(
         title={
            'text': f"{select_feature} by Popularity Ranges",
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Popularity Ranges',
        yaxis_title='Feature Values',
        title_x=0.35,  # Set the title position to the center
        title_y=0.9,  # Set the title position to the upper part
        showlegend=False   
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
        
def first_vis_alt(data):
    songs_normalize = data.copy()
    songs_normalize = songs_normalize.drop(['explicit','genre'], axis=1)
    songs_normalize.sort_values('popularity', inplace=True)
    scaler = MinMaxScaler()
    songs_normalize[songs_normalize.columns.difference(['artist','song', 'year','genre','popularity'])] = scaler.fit_transform(songs_normalize[songs_normalize.columns.difference(['artist','song', 'year','genre','popularity'])])

    # Get the columns names and save only the relevant ones
    songs_normalize = songs_normalize.drop('popularity', axis=1)
    
    # Get the feature names
    column_names = list(songs_normalize.columns.values)
    features_to_remove = ['song', 'artist','genre', 'year']
    features_names = [item for item in column_names if item not in features_to_remove]

    # Select feature using Streamlit
    select_feature = st.selectbox('Choose feature 1:', features_names)

    fig = px.scatter(songs_normalize, x='popularity', y=select_feature, title=f'{select_feature} by Popularity', width=900, height=500)
    
    # Update the layout
    fig.update_layout(
        xaxis_title='Popularity',
        yaxis_title='Feature Values',
        title_x=0.5,  # Set the title position to the center
        showlegend=False   
    )
    
    # Create the figure
    fig.update_traces(marker=dict(size=5))
    
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

    # Define the year ranges for the facets
    year_ranges = [(1998, 2003), (2004, 2009), (2010, 2015), (2016, 2020)]

    # Define the popularity ranges
    popularity_ranges = [range(0, 18), range(18, 36), range(36, 54), range(54, 72), range(72, 90)]

    # Dropdown menu for feature selection
    feature_names = data.columns.tolist()
    feature_dropdown = st.selectbox("Feature:", feature_names)

    # Create a figure with subplots for each facet
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"{start}-{end}" for start, end in year_ranges])

    # Iterate over the year ranges and create the facets
    for i, (start_year, end_year) in enumerate(year_ranges):
        facet_data = data[(data['year'] >= start_year) & (data['year'] <= end_year)]

        # Calculate average feature values for each popularity range in the facet
        feature_avg_values = []
        for popularity_range in popularity_ranges:
            avg_value = facet_data[facet_data['popularity'].isin(popularity_range)][feature_dropdown].mean()
            feature_avg_values.append(avg_value)

        # Normalize the feature average values between 0 and 1
        min_value = min(feature_avg_values)
        max_value = max(feature_avg_values)
        if min_value != max_value:
            normalized_values = (feature_avg_values - min_value) / (max_value - min_value)
        else:
            normalized_values = feature_avg_values

        # Create the bar chart for the facet
        row = (i // 2) + 1
        col = (i % 2) + 1
        fig.add_trace(go.Bar(
            x=normalized_values,
            y=[f"Popularity {p.start}-{p.stop-1}" for p in popularity_ranges],
            orientation='h',
            marker=dict(
                color='rgb(63, 81, 181)',
                line=dict(
                    color='rgb(40, 55, 71)',
                    width=1.5
                )
            ),
            opacity=0.8,
            showlegend=False
        ), row=row, col=col)

    fig.update_layout(
        yaxis_title='Popularity Range',
        xaxis_title='Average Normalized Value',
        title={
            'text': f"Average Feature Values by Popularity Range for {feature_dropdown}",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        yaxis=dict(
            tickfont=dict(size=10),
            gridcolor='rgb(238, 238, 238)'
        ),
        xaxis=dict(
            tickfont=dict(size=10),
            gridcolor='rgb(238, 238, 238)'
        ),
        width=800,
        height=600
    )

    st.plotly_chart(fig)

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
        

def map_vis(map_data):
    avg_popularity = map_data.groupby(['Country'])['popularity'].mean().reset_index()
    avg_popularity['popularity'] = avg_popularity['popularity'].round(2)

    # Create the choropleth map using plotly express
    fig = px.choropleth(avg_popularity, locations='Country', locationmode='country names',
                        color='popularity', color_continuous_scale='RdYlBu',
                        labels={'value': 'Average Popularity'}, projection="natural earth")

    # Update the layout to position the title in the middle
    fig.update_layout(
        title={
            'text': 'Average Popularity by Country',
            'x': 0.5,  # Set x position to the middle of the graph
            'xanchor': 'center',  # Anchor the x position to the center
            'yanchor': 'top'  # Position the title at the top
        },     
        width=900,  # Set the width of the chart
        height=500,  # Set the height of the chart
    )

    # Display the figure in Streamlit
    col1, col2 = st.columns([1,16])
    with col1:
        st.write("")
    with col2:
        st.plotly_chart(fig)
        


st.header('What are the trends and patterns in popular music from 2000 to 2019, based on the Top Hits Spotify dataset?')
st.header("Are there any notable differences between popular songs from different years? ")
st.write("Explore the change in diffrent features in spotify most popular songs over the years. Each line represents the average value of a specific feature over the years. You can select individual features to see their trends over time by clicking on their names in the legend. To see all the features together, simply choose the 'All' option from the dropdown menu. You can also temporarily remove a feature from the graph by clicking on its name.")
first_vis(data)
first_vis_alt(data)
st.header('What are the characteristics that have the strongest influence on the popularity of a song? ')
st.write("Explore the factors that shape a song's popularity. By selecting different features from the dropdown menu, you can observe how various characteristics influence a song's popularity.")
st.write(" A positive SHAP value suggests that as a feature's value increases, it tends to increase the song's popularity. On the other hand, a negative SHAP value indicates that as a feature's value increases, it may have a diminishing effect on the song's popularity.For instance, take the feature 'duration_ms'  that is shown below as an example. As the duration of the song increases, it may have a negative impact on the song's popularity. ")
second_vis(data)
second_vis_alt(data)
map_vis(map_data)
st.header('How has the popularity of different genres changed over time?')
st.write("Explore the popularity of different music genres over the years. The graph displays the average popularity of the selected genre across different years. The height of each bar represents the popularity level, where higher values indicate greater popularity.")
third_vis(data)
