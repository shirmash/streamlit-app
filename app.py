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
    songs_normalize = songs_normalize.drop(['explicit', 'genre'], axis=1)
    songs_normalize.sort_values('popularity', inplace=True)
    scaler = MinMaxScaler()
    songs_normalize[songs_normalize.columns.difference(['artist', 'song', 'year', 'genre', 'popularity'])] = scaler.fit_transform(songs_normalize[songs_normalize.columns.difference(['artist', 'song', 'year', 'genre', 'popularity'])])

    # Get the feature names
    column_names = list(songs_normalize.columns.values)
    features_to_remove = ['song', 'artist', 'genre', 'popularity']
    features_names = [item for item in column_names if item not in features_to_remove]

    # Select feature using Streamlit
    select_feature = st.selectbox('Choose feature :', features_names)

    year_ranges = [(1998, 2003), (2004, 2009), (2010, 2015), (2016, 2020)]

    fig = make_subplots(rows=2, cols=2, subplot_titles=[f'{range[0]}-{range[1]}' for range in year_ranges])

    for i, range in enumerate(year_ranges):
        range_df = songs_normalize[(songs_normalize['year'] >= range[0]) & (songs_normalize['year'] <= range[1])]
        fig.add_trace(go.Scatter(x=range_df[select_feature], y=range_df['popularity'], mode='markers', name=f'{range[0]}-{range[1]}'), row=(i // 2) + 1, col=(i % 2) + 1)

    fig.update_layout(title=f'Popularity by {select_feature} and Year', width=900, height=900)

    # Update the layout
    fig.update_xaxes(title_text=select_feature, row=1, col=1)
    fig.update_xaxes(title_text=select_feature, row=1, col=2)
    fig.update_yaxes(title_text='Popularity', row=1, col=1)
    fig.update_yaxes(title_text='Popularity', row=2, col=1)

    # Display the figure
    st.plotly_chart(fig)
def first_vis_alt(data):
    songs_normalize = data.copy()
    songs_normalize = songs_normalize.drop(['explicit', 'genre'], axis=1)

    scaler = MinMaxScaler()
    songs_normalize[songs_normalize.columns.difference(['artist', 'song', 'year', 'explicit'])] = scaler.fit_transform(songs_normalize[songs_normalize.columns.difference(['artist', 'song', 'year', 'explicit'])])
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

    # Create reference lines for popular songs
    popular_songs = songs_normalize[songs_normalize['popularity'] > 80]
    reference_lines = []
    for _, song in popular_songs.iterrows():
        reference_line = go.Scatter(x=[song['year'], song['year']], y=[0, 1], name='Popular Song', mode='lines', line=dict(dash='dash'))
        reference_lines.append(reference_line)

    # Create the layout with checklist dropdown
    layout = go.Layout(
        title='Average Feature Value per Year',
        title_x=0.3,  # Set the title position to the center
        title_y=0.9,  # Set the title position to the upper part
        xaxis_title='Year',
        yaxis_title='Average Normalized Value',
        legend=dict(
            title='Choose Features',
            title_font=dict(size=18),
        ),
        annotations=[
            dict(
                x=1.18,
                y=0.37,  # Adjust the y-coordinate to position the note below the legend
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
    fig = go.Figure(data=lines + reference_lines, layout=layout)
    fig.update_layout(
        width=1000,  # Set the width of the chart
        height=600,  # Set the height of the chart
    )
    # Display the figure
    col1, col2 = st.columns([1, 7])
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
