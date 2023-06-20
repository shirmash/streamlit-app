import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
from pandas.plotting import parallel_coordinates
from plotly.subplots import make_subplots
np.bool = np.bool_

st.set_page_config(layout="wide",page_title="Spotify Music Insights")
data = pd.read_csv('songs_normalize.csv')
map_data = pd.read_csv('map_data.csv')
# Display the Spotify logo image
st.image("Spotify_Logo.png", width=170)
st.title('Visualization: Final Project')

def first_vis(data):
    data = data.copy()
    range_data = data.copy()
    range_data = range_data.drop(['explicit', 'genre', 'key'], axis=1)
    # Scale the data
    scaler = MinMaxScaler()
    range_data[range_data.columns.difference(['artist', 'song', 'year', 'explicit', 'mode','key'])] = scaler.fit_transform(range_data[range_data.columns.difference(['artist', 'song', 'year', 'explicit', 'mode'])])
    column_names = list(range_data.columns.values)
    features_to_remove = ['song', 'explicit', 'artist', 'year', 'popularity', 'mode','key']
    features_names = [item for item in column_names if item not in features_to_remove]
    non_numeric_columns = range_data.select_dtypes(include=['object']).columns
    range_data[non_numeric_columns] = range_data[non_numeric_columns].apply(pd.to_numeric, errors='coerce')

    selected_feature = st.selectbox("Select Feature:", features_names)  # dropdown
    x_min, x_max = st.slider('select feature range:', 0.0, 1.0, (0.0, 1.0))
    year_ranges = [(1999, 2004), (2005, 2010), (2011, 2015), (2016, 2020)]
    colors = ['#d62728', '#F9564F', '#2ca02c', '#98df8a']  # colors for the year ranges
    # Create a dictionary mapping year ranges to colors
    color_map = {range_start: color for (range_start, _), color in zip(year_ranges, colors)}
    filtered_data = range_data[(range_data[selected_feature] >= x_min) & (range_data[selected_feature] <= x_max)]
    filtered_data['year_range'] = pd.cut(filtered_data['year'], bins=[range_start for (range_start, _) in year_ranges] + [ filtered_data['year'].max()],labels=[f'{start}-{end}' for (start, end) in year_ranges], right=False)
    traces = []
    legend_labels = []  # Store the legend labels
    # Iterate over the year ranges
    for range_index, (start, end) in enumerate(year_ranges):
        range_label = f'{start}-{end}'
        range_data = filtered_data[filtered_data['year_range'] == range_label]
        # Round the values of popularity and selected feature to 2 decimal places
        rounded_popularity = range_data['popularity'].round(2)
        rounded_feature = range_data[selected_feature].round(2)
        # Create a scatter trace for each year range
        trace = go.Scatter(x=rounded_feature,y=rounded_popularity,mode='markers',marker=dict(color=colors[range_index]),text=data['song'].astype(str) + ' - ' + data['artist'].astype(str), name=range_label )
        traces.append(trace)
        legend_labels.append(range_label)
    layout = go.Layout(
        title={
            'text': f"{selected_feature} Impact On Songs Popularity",'x': 0.43, 'y': 0.9,'xanchor': 'center','yanchor': 'top'},
        xaxis_title=selected_feature,
        yaxis_title='Popularity',
        showlegend=True,
        legend=dict(title='Year Range'),  # Set the legend labels
        annotations=[
            dict(x=1.08, y=0.65, xref="paper", yref="paper", xanchor="center", yanchor="bottom", text="One click to remove",
                 showarrow=False, font=dict(size=13))])

    fig = go.Figure(data=traces, layout=layout)
    fig.update_layout(width=900, height=500)  # Set the height and width of the chart
     # Display the figure in Streamlit
    col1, col2 = st.columns([1,16])
    with col1:
        st.write("")
    with col2:
        st.plotly_chart(fig)

def second_vis(map_data):
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
        width=900, height=500)

    # Display the figure in Streamlit
    col1, col2 = st.columns([1,16])
    with col1:
        st.write("")
    with col2:
        st.plotly_chart(fig)

def third_vis(data):
    # Drop rows with missing genre values
    data.dropna(subset=['genre'], inplace=True)
    # Split multiple genres into a list
    data['genre'] = data['genre'].str.split(',')
    # Remove leading/trailing whitespace from each genre
    data['genre'] = data['genre'].apply(lambda genres: [genre.strip() for genre in genres])
    # Remove 'classical' and 'jazz' genres
    data['genre'] = data['genre'].apply(lambda genres: [genre for genre in genres if genre not in ['classical', 'jazz']])
    df_songs_genres = pd.DataFrame.from_records(data, columns=['year', 'popularity', 'genre'])
    df_songs_genres = df_songs_genres.explode('genre').reset_index(drop=True)
    avg_popularity_genre = df_songs_genres.groupby(['year', 'genre'])['popularity'].mean().reset_index()
    avg_popularity_genre = avg_popularity_genre.pivot(index='year', columns='genre', values='popularity')
     # Create the bars for the plot
    bars = []
    genres = []
    for column in avg_popularity_genre.columns:
                bar = go.Bar(
                    x=avg_popularity_genre.index,
                    y=avg_popularity_genre[column],
                    name=column,
                    marker=dict(color='orange', line=dict(color='black', width=1))
                )
                bars.append(bar)
                genres.append(column)
    # Create the selectbox
    select_genre = st.selectbox('Choose genre:', genres)
    # Set the visibility of the bars based on the selected genre
    visible_column = [column == select_genre for column in genres]
    for bar, visibility in zip(bars, visible_column):
        bar.visible = visibility
    layout = go.Layout(
        barmode='stack',  # Set the barmode to 'stack' for stacked bars
        xaxis_title='Year',
        yaxis_title='Average Popularity',
        showlegend=False,)    
    fig = go.Figure(data=bars, layout=layout)
    fig.update_layout(width=900,  height=500, 
        title={
            'text': f"Popularity of the genre {select_genre} Over the Years",
            'x': 0.3, 'y': 0.85 })
    col1, col2 = st.columns([1,16])
    with col1:
        st.write("")
    with col2:
        st.plotly_chart(fig)
    

st.header('What are the trends and patterns in popular music from 2000 to 2019, based on the Top Hits Spotify dataset?')
st.subheader("How does the value of a feature influence the popularity of songs across different time periods?")
st.write("Discover how different features influence the popularity of songs. Each data point represents a song. Use the slider to narrow down the songs within a specific range of the selected feature, gaining insights into their trends.")
st.write("To further refine your analysis, simply click on the name of a year range in the legend to temporarily remove it from the graph. This enables you to focus on specific periods of interest. In addition you can hover over a data point to see the name of the song and the artist.")
first_vis(data)
st.subheader('From which countries do the artists with the most popular songs originate?')
st.write("This visualization displays the average popularity of songs in different countries.Each country on the map is color-coded to reflect the average popularity of songs in that region. Warm colors like red indicate lower popularity, while cool colors like blue represent higher popularity. By hovering over a country, you can uncover its specific average popularity value, gaining insights into the musical preferences of different regions.Feel free to move around by dragging the map and use the zoom controls to get a closer look at specific areas of interest.")
second_vis(map_data)
st.subheader('How has the popularity of different genres changed over time?')
st.write("Explore the popularity of different music genres over the years. The graph displays the average popularity of the selected genre across different years. The height of each bar represents the average popularity level of the genre in this year, where higher values indicate greater popularity.")
third_vis(data)
