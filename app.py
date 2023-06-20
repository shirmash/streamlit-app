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
st.title('Visualization: Final Project')

def first_vis(data):
    data = data.copy()
    data = data.drop(['explicit', 'genre'], axis=1)

    scaler = MinMaxScaler()
    data[data.columns.difference(['artist', 'song', 'year', 'explicit','mode'])] = scaler.fit_transform(
        data[data.columns.difference(['artist', 'song', 'year', 'explicit','mode'])])

    column_names = list(data.columns.values)
    features_to_remove = ['song', 'explicit', 'artist', 'year', 'popularity','mode']
    features_names = [item for item in column_names if item not in features_to_remove]

    non_numeric_columns = data.select_dtypes(include=['object']).columns
    data[non_numeric_columns] = data[non_numeric_columns].apply(pd.to_numeric, errors='coerce')

    st.title("Feature Analysis")

    selected_feature = st.selectbox("Select Feature:", features_names)

    feature_min = round(data[selected_feature].min(), 2)
    feature_max = round(data[selected_feature].max(), 2)
    x_min, x_max = st.slider('Select X-axis Range:', float(feature_min), float(feature_max), (float(feature_min), float(feature_max)))
    year_ranges = [
    (1999, 2004),
    (2005, 2010), 
    (2011, 2015), 
    (2016, 2020)]
    #colors =['#ff9896', '#ED553B','#3CAEA3', '#F9564F']# Assign colors for each year range
    colors=['#d62728', '#F9564F','#2ca02c', '#98df8a']
    # Create a dictionary mapping year ranges to colors
    color_map = {range_start: color for (range_start, _), color in zip(year_ranges, colors)}
    filtered_data = data[(data[selected_feature] >= x_min) & (data[selected_feature] <= x_max)]
    # Add a new column for the categorical color based on year range
    filtered_data['year_range'] = pd.cut(filtered_data['year'], bins=[range_start for (range_start, _) in year_ranges] + [filtered_data['year'].max()], labels=[f'{start}-{end}' for (start, end) in year_ranges], right=False)
    
    # Create an empty list to store the traces
    traces = []
    
    # Iterate over the year ranges
    for range_index, (start, end) in enumerate(year_ranges):
        range_label = f'{start}-{end}'
        range_data = filtered_data[filtered_data['year_range'] == range_label]
        
        # Create a scatter trace for each year range
        trace = go.Scatter(
            x=range_data[selected_feature],
            y=range_data['popularity'],
            mode='markers',
            marker=dict(color=colors[range_index]),
            name=range_label
        )
        
        # Add the trace to the list
        traces.append(trace)
    
        layout = go.Layout(
        title=f"Feature: {selected_feature} vs Popularity",
        xaxis_title=selected_feature,
        yaxis_title='Popularity',
        showlegend=True,
        legend=dict(title='Year Range'),
        annotations=[
            dict(
                x=1.15,
                y=0.5,
                xref="paper",
                yref="paper",
                xanchor="center",
                yanchor="bottom",
                text="One click to remove the feature",
                showarrow=False,
                font=dict(size=12) )] )
    
    # Create the figure
    fig = go.Figure(data=traces, layout=layout)

    
    # Show the plot
    col1, col2 = st.columns([1,16])
    with col1:
        st.write("")
    with col2:
        st.plotly_chart(fig)


# def first_vis(data):
#     songs_popular = data.copy()
    
#     # Filter songs by popularity range
#     popularity_range = st.slider('Select Popularity Range', min_value=0, max_value=89, value=(0, 89))
#     songs_popular = songs_popular[(songs_popular['popularity'] >= popularity_range[0]) & (songs_popular['popularity'] <= popularity_range[1])]
    
#     # Get the column names and save only the relevant ones
#     column_names = list(songs_popular.columns.values)
#     features_to_remove = ['song', 'explicit', 'artist', 'year', 'popularity']
#     features_names = [item for item in column_names if item not in features_to_remove]
    
#     # Convert non-numeric columns to numeric
#     non_numeric_columns = songs_popular.select_dtypes(exclude=np.number).columns
#     songs_popular[non_numeric_columns] = songs_popular[non_numeric_columns].apply(pd.to_numeric, errors='coerce')
    
#     # Normalize the features
#     scaler = MinMaxScaler()
#     songs_popular[features_names] = scaler.fit_transform(songs_popular[features_names])
    
#     avg_popularity = songs_popular.groupby(['year'], as_index=False)[features_names].mean()
#     avg_popularity[features_names]=avg_popularity[features_names].round(2)
#     # Create the lines for the plot
#     lines = []
#     for column in avg_popularity.columns:
#         if column != 'year':
#             line = go.Scatter(x=avg_popularity['year'], y=avg_popularity[column], name=column)
#             lines.append(line)

#     # Create the layout with checklist dropdown
#     layout = go.Layout(
#         title='Average Feature Value per Year',
#         title_x=0.3,  # Set the title position to the center
#         title_y=0.9,  # Set the title position to the upper part
#         xaxis_title='Year',
#         yaxis_title='Average Normalized Value',
#         legend=dict(
#             title='Choose Features',
#             title_font=dict(size=18),
#         ),
#         annotations=[
#             dict(
#                 x=1.16,
#                 y=0.31,  # Adjust the y-coordinate to position the note below the legend
#                 xref='paper',
#                 yref='paper',
#                 text='One click to remove the feature',
#                 showarrow=False,
#                 font=dict(size=10),
#             )
#         ],
#         updatemenus=[  # the user can choose to see all features in one click
#             dict(
#                 buttons=list([
#                     dict(
#                         label='All',
#                         method='update',
#                         args=[{'visible': [True] * len(lines)}, {'title': 'Average Feature Value per Year'}]
#                     )
#                 ]),
#                 direction='down', # the position of the dropdown
#                 showactive=True,
#                 x=1.1,
#                 xanchor='right',
#                 y=1.15,
#                 yanchor='top'
#             )
#         ]
#     )

#     # Create the figure
#     fig = go.Figure(data=lines, layout=layout)
#     fig.update_layout(
#         width=1000,  # Set the width of the chart
#         height=600,  # Set the height of the chart
#     )
    
#     # Display the figure
#     st.plotly_chart(fig)
def third_vis(data):
    # Drop rows with missing genre values
    data.dropna(subset=['genre'], inplace=True)
    
    # Split multiple genres into a list
    data['genre'] = data['genre'].str.split(',')
    
    # Remove leading/trailing whitespace from each genre
    data['genre'] = data['genre'].apply(lambda genres: [genre.strip() for genre in genres])
    df_songs_genres = pd.DataFrame.from_records(data, columns=['year', 'popularity', 'genre'])
    df_songs_genres = df_songs_genres.explode('genre').reset_index(drop=True)
    avg_popularity_genre = df_songs_genres.groupby(['year', 'genre'])['popularity'].mean().reset_index()
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

    # Create the initial selectbox
    select_genre = st.selectbox('Choose genre:', genres)

    # Set the visibility of the bars based on the selected genre
    visible_column = [column == select_genre for column in genres]
    for bar, visibility in zip(bars, visible_column):
        bar.visible = visibility
        
    layout = go.Layout(
        barmode='stack',  # Set the barmode to 'stack' for stacked bars
        xaxis_title='Year',
        yaxis_title='Average Popularity',
        showlegend=False,
    )    
    fig = go.Figure(data=bars, layout=layout)
    fig.update_layout(
        width=900,  # Set the width of the chart
        height=500,  # Set the height of the chart
        title={
            'text': f"Popularity of the genre {select_genre} Over the Years",
            'x': 0.3,  # Set the title position to the middle horizontally
            'y': 0.85  # Set the title position slightly below the top vertically
        }
    )
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
st.header("Are there any notable differences in the features between popular songs from different years? ")
st.write("Explore the change in different features in Spotify's most popular songs over the years. Each line represents the average normalized value of a specific feature over the years. To focus on songs within a specific popularity range, you can use the slider to select the desired range. The graph will display the average feature values for songs falling within the selected popularity range.")
st.write(" To view all the features together, simply choose the 'All' option from the dropdown menu. This allows you to compare how different features evolve over time. Additionally, you have the flexibility to temporarily remove a specific feature from the graph by clicking on its name.")
first_vis(data)
st.header('What are the characteristics that have the strongest influence on the popularity of a song? ')
st.write("This visualization displays the average popularity of songs in different countries.Each country on the map is color-coded to reflect the average popularity of songs in that region. Warm colors like red indicate lower popularity, while cool colors like blue represent higher popularity. By hovering over a country, you can uncover its specific average popularity value, gaining insights into the musical preferences of different regions.Feel free to move around by dragging the map and use the zoom controls to get a closer look at specific areas of interest.")
map_vis(map_data)
st.header('How has the popularity of different genres changed over time?')
st.write("Explore the popularity of different music genres over the years. The graph displays the average popularity of the selected genre across different years. The height of each bar represents the popularity level, where higher values indicate greater popularity.")
third_vis(data)
