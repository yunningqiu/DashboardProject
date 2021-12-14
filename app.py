from __future__ import print_function
# data manipulation
import pandas as pd
import numpy as np
# plotly 
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# dashboards
import dash 
import dash_table as dt
from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
import dash_bootstrap_components as dbc


from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import requests
import matplotlib as plt
import subprocess
import os 

app = dash.Dash(external_stylesheets=[dbc.themes.LUX])

df = pd.read_csv('DSO545project(movie).csv')
df['Year'] = df['Release Date']
variable_labels = {
    'Hidden Gem Score': 'Hidden Gem Score',
    'IMDb Score': 'IMDb Score',
    'Rotten Tomatoes Score': 'Rotten Tomatoes Score', 
    'Metacritic Score': 'Metacritic Score',
    'Boxoffice': 'Dollars at box office',
    'Year': 'Year',
}

# Overall trend over years
data = df.loc[:,['Title','Hidden Gem Score','IMDb Score','Rotten Tomatoes Score','Metacritic Score','Boxoffice','Release Date']]
data = data[(data != 0).all(1)]
pd.cut(data['Release Date'], 6)
data['Year'] = pd.cut(data['Release Date'], 6, labels=["~1945","1945-1960","1960-1975","1975-1990","1990-2005","2005-2020"])
df_rating = data.groupby('Year')[['Hidden Gem Score', 'IMDb Score', 'Rotten Tomatoes Score', 'Metacritic Score']].mean()
df_box = data.groupby('Year')['Boxoffice'].mean()
box_fig = px.line(data, x=df_box.index, y=data.groupby('Year')['Boxoffice'].mean(),
              title = f'Average Box Office of Movies Over Time',range_x=[-0.5,5.5],
              labels={'x':'Year','y':'Average Box Office'}
)
df_parallel = df.loc[:,['Title','Series or Movie','Runtime','Release Date','Awards Received','Awards Nominated For']]
df_parallel = df_parallel[(df_parallel['Series or Movie'] == 'Movie') & (df_parallel['Release Date'] != 0)]
bins = pd.IntervalIndex.from_tuples([(1929.91, 1945.0),(1945.0, 1960.0),(1960.0, 1975.0), (1975.0, 1990.0), (1990.0, 2005.0), (2005.0, 2020.0)])
pd.cut(df_parallel['Release Date'], bins, labels=["~1945","1945-1960","1960-1975","1975-1990","1990-2005","2005-2020"])
df_parallel['Year'] = pd.cut(df_parallel['Release Date'], [1929.91,1945,1960,1975,1990,2005,2020], labels=["~1945","1945-1960","1960-1975","1975-1990","1990-2005","2005-2020"])
df_parallel['Awards Nominated For'] = [1 if x != 0 else 0 for x in df_parallel["Awards Nominated For"]]
df_parallel['Awards Received'] = [1 if x != 0 else 0 for x in df_parallel["Awards Received"]]

# Table
table = df.drop(columns = ['Unnamed: 0', 'Country Availability', 'Writer', 'Actors', 'Production House', 'Summary', \
                   'IMDb Votes', 'Trailer Site', 'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', \
                   'Drama', 'Family', 'Fantasy', 'Romance', 'Thriller', 'Other', 'Tags', 'Runtime', 'Netflix Release Date'
                 ])
table = table.iloc[:, [0, 1, 2, 3, 5, 6, 4, 8, 9, 7, 10, 11, 12, 13]]

table['id'] = table['Title']
table.set_index('id', inplace=True, drop=False)

# Create dimensions
year_dim = go.parcats.Dimension(
    values=df_parallel['Year'], label="Year of Release"
)

runtime_dim = go.parcats.Dimension(
    values=df_parallel['Runtime'], label="Runtime", categoryorder='category ascending'
)

award_nom_dim = go.parcats.Dimension(
    values=df_parallel['Awards Nominated For'], label="Awards Nominated", categoryarray=[0, 1],
    ticktext=['Nominated', 'Not Nominated']
)

award_win_dim = go.parcats.Dimension(
    values=df_parallel['Awards Received'], label="Awards Received", categoryarray=[0, 1],
    ticktext=['Received', 'Not Received']
)

# Create parcats trace
color = df_parallel['Awards Received'];
colorscale = [[0, 'mediumseagreen'], [1, 'lightsteelblue']];

parallel_cat = go.Figure(data = [go.Parcats(dimensions=[year_dim, runtime_dim, award_nom_dim, award_win_dim],
        line={'color': color, 'colorscale': colorscale},
        hoveron='color', hoverinfo='count+probability',
        labelfont={'size': 18, 'family': 'Times'},
        tickfont={'size': 16, 'family': 'Times'},
        arrangement='freeform')])

# Boxes
# limit the scope to 2011-2020
data10 = df[(df['Release Date'] >= 2011) & (df['Release Date'] <=2020)]

# calculate the average boxoffice
boxoffice_11_20 = data10.loc[(df['Boxoffice']!= 0), ['Title', 'Release Date', 'Boxoffice']]
avg_11_20 = round((boxoffice_11_20['Boxoffice'].mean()/1000000),2)

# figure out the most popular view rating
pop_view_rating = data10[['View Rating', 'Boxoffice']].groupby('View Rating').\
    agg({'Boxoffice':sum}).\
        sort_values(by = 'Boxoffice', ascending=False).head(1)
rating = [r for r in pop_view_rating.index]
rating = rating[0]

# figure out the most popular genre
genre_list = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Drama', 'Family', 'Fantasy', 'Romance', 'Thriller']
genre_boxoffice = {}
for i in genre_list:
    genre_data = data10.loc[(data10[i] == 1) & (data10['Boxoffice']!=0), 'Boxoffice']
    genre_boxoffice[i] = genre_data.mean()/1000000
for i in genre_boxoffice:
    if genre_boxoffice[i] == max(genre_boxoffice.values()):
        pop_genre = i
       
# calculate the awards recevied for both movies and series
data10['Awards'] = [1 if i > 0 else 0 for i in data10['Awards Received']]
awards = round(data10['Awards'].mean()*100,2)

# Genres
rating_list = ['Hidden Gem Score', 'IMDb Score', 'Rotten Tomatoes Score', 'Metacritic Score']

# Boxes
cards = [
    dbc.Card(
        [
            html.H3(f"{avg_11_20}M $", className="card-title"),
            html.P("Average Boxoffice", className="card-text"),
        ],
        body=True,
        color="light",
    ),
    dbc.Card(
        [
            html.H3(f"{rating}", className="card-title"),
            html.P("Most Popular View Rating", className="card-text"),
        ],
        body=True,
        color="dark",
        inverse=True,
    ),
    dbc.Card(
        [
            html.H3(f"{pop_genre}", className="card-title"),
            html.P("Most Popular Genre", className="card-text"),
        ],
        body=True,
        color="light"
    ),
    dbc.Card(
        [
            html.H3(f"{awards}%", className="card-title"),
            html.P("Awards Received %", className="card-text"),
        ],
        body=True,
        color="dark",
        inverse=True,
    ),
]

def build_modal_info_overlay(id, side, content):
    """
    Build div representing the info overlay for a plot panel
    """
    div = html.Div(
        [  # modal div
            html.Div(
                [  # content div
                    html.Div(
                        [
                            html.H4(
                                [
                                    "Info",
                                    html.Img(
                                        id=f"close-{id}-modal",
                                        src="assets/times-circle-solid.svg",
                                        n_clicks=0,
                                        className="info-icon",
                                        style={"margin": 0},
                                    ),
                                ],
                                className="container_title",
                                style={"color": "white"},
                            ),
                            dcc.Markdown(content),
                        ]
                    )
                ],
                className=f"modal-content {side}",
            ),
            html.Div(className="modal"),
        ],
        id=f"{id}-modal",
        style={"display": "none"},
    )
    return div

app.layout = html.Div([
    dbc.Container([
        html.Br(),
        html.H1('Which kind of movies would receive '),
        html.H1('the highest ratings / highest revenue?')
    ]),
    dbc.Container(
        [   
            html.Hr(),
            dbc.Row([dbc.Col(card) for card in cards]),
        ],
        fluid=False,
    ), 
    html.P("*in recent 10 years", className='text-center'),
    html.Br(),
    html.Br(),

    # Average rating / box office over years
    dbc.Container([
        html.H2('Overall Trend of Ratings / Box Office Over Years'),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Graph(id='avg_rating_year'),
                    dcc.Dropdown(
                        id = 'rating_dropdown',
                        options=[
                            {'label': 'Average Hidden Gem Score', 'value': 'Hidden Gem Score'},
                            {'label': 'Average IMDb Score', 'value': 'IMDb Score'},
                            {'label': 'Average Rotten Tomatoes Score', 'value': 'Rotten Tomatoes Score'},
                            {'label': 'Average Metacritic Score', 'value': 'Metacritic Score'}
                        ],
                        value='Hidden Gem Score',
                    )
                ]),
            ]),
            dbc.Col([
                html.Div([
                    dcc.Graph(id='avg_boxoffice', figure=box_fig)
                ])
            ])
        ]), 
    ]),
    html.Br(),
    html.Br(),
    html.Br(),
    
    # Recent 10 years boxoffice / rating by genre
    dbc.Container([
        html.H2('Average Boxoffice/Rating by Top 10 Genres in recent 10 years'),   
        html.Hr(),     
        dcc.Dropdown(
            id='genre_dropdown',
            options=[{'label': genre, 'value': genre} for genre in genre_list],
            value = 'Drama'
        ),
        
        dcc.RadioItems(
            id='boxoffice-rating',
            options=[{'label': i, 'value': i} for i in ['Boxoffice', 'Rating']],
            value='Boxoffice',
            labelStyle={'display': 'inline-block'}
        ),
        
        dcc.Graph(
            id='genre_output'
        ),
    ]),
    html.Br(),
    html.Br(),
    html.Br(),
    
    # movie / director with most box office
    dbc.Container([
        dbc.Row([       
            dbc.Col([
                html.H2('Movies that collected the most box office'),
                html.Hr(),
                dcc.Slider(
                            id='amount1',
                            min=5,
                            max=20,
                            step=1,
                            value=10,
                            marks={
                            5: 'Top5',
                            10: 'Top10',
                            15: 'Top15',
                            20: 'Top20'
                        }
                        ),
                dcc.Graph(id="movie-bar-chart",figure={})]),
            dbc.Col([
                html.H2('Directors that collected the most box office'),
                html.Hr(),
                dcc.Slider(
                            id='amount2',
                            min=5,
                            max=20,
                            step=1,
                            value=10,
                            marks={
                            5: 'Top5',
                            10: 'Top10',
                            15: 'Top15',
                            20: 'Top20'
                        }
                        ),
                dcc.Graph(id="director-bar-chart",figure={})]),
        ]),
    ]),
    html.Br(),
    html.Br(),
    html.Br(),
    
    # parallel category graph for award winning
    dbc.Container([
        html.H2('Parallel Category Graph of Award Winning Movies'),
        html.Hr(),
        html.Div([
            dcc.Graph(id='parallel_cat', figure=parallel_cat)
        ])
    ]),
    html.Br(),
    html.Br(),
    html.Br(),
    
    # Table
    dbc.Container([
        html.H2('Table of Movies Attributes with Filter'),
        html.Hr(),
        html.Div([
            dt.DataTable(
                id='datatable-row-ids',
                columns=[
                    {'name': i, 'id': i, 'deletable': True} for i in table.columns
                    if i != 'id'
                ],
                fixed_columns={'headers': True, 'data': 1},
                style_table={'minWidth': '100%'},
                data=table.to_dict('records'),
                editable=True,
                filter_action="native",
                sort_action="native",
                sort_mode='multi',
                row_selectable='multi',
                row_deletable=True,
                selected_rows=[],
                page_action='native',
                page_current= 0,
                page_size= 10,
            )
        ])
    ]),
    html.Br(),
    html.Br(),
    html.Br(), 
          
    # Explore specific movies
    dbc.Container([
        html.H2('Explore Specific Movies'),
        html.Hr(),
        html.Div([
            html.Label('Year released'),
            dcc.RangeSlider(
                id='year-released-range-slider',
                min=1910,
                max=df.Year.max(),
                marks={'1910':'1910', '1925':'1925', '1940':'1940', '1955':'1955', '1970':'1970', '1985':'1985', '2000':'2000', '2015':'2015'},
                value=[1910, df.Year.max()]
            ),
            html.Br(),
            html.Label('Minimum number of awards wins'),
            dcc.Slider(
                id='awards-won-slider',
                min=df['Awards Received'].min(),
                max=df['Awards Received'].max(),
                marks={str(o): str(o) for o in range(int(df['Awards Received'].min()), int(df['Awards Received'].max()), 20)},
                value=df['Awards Received'].min(),
                step=None
            ),
            html.Br(),
            html.Label('Dollars at Box Office (millions)'),
            dcc.RangeSlider(
                id='dollars-boxoffice-range-slider',
                min=0,
                max=800,
                marks={str(y): str(y) for y in range(0, 800, 50)},
                value=[0, 800],
            ),
            html.Br(),
        ], style={'marginLeft': 25, 'marginRight': 25}
        ),
        html.Div([
            html.Br(),
            html.Label('X-axis variable'),
            dcc.Dropdown(
                id='x-axis-dropdown',
                options=[
                    {'label': label, 'value': value} for value, label in variable_labels.items()
                ],
                multi=False,
                value='IMDb Score'
            ),
            html.Br(),
            html.Label('Y-axis variable'),
            dcc.Dropdown(
                id='y-axis-dropdown',
                options=[
                    {'label': label, 'value': value} for value, label in variable_labels.items()
                ],
                multi=False,
                value='Boxoffice'
            )
        ], style={'marginLeft': 25, 'marginRight': 25}),
        html.Div([
            html.Label('Graph'),
            dcc.Graph(
                id='scatter-plot-graph',
                animate=True,
                figure={
                    'data': [
                        go.Scatter(
                            x=df[df['Awards Received'] > 0]['IMDb Score'],
                            y=df[df['Awards Received'] > 0]['Boxoffice'],
                            text=df[df['Awards Received'] > 0].Title,
                            mode='markers',
                            opacity=0.5,
                            marker={
                                'color': 'orange',
                                'size': 10,
                                'line': {'width': 1, 'color': 'black'}
                            },
                            name='has_award: yes'
                        ),
                        go.Scatter(
                            x=df[df['Awards Received'] == 0]['IMDb Score'],
                            y=df[df['Awards Received'] == 0]['Boxoffice'],
                            text=df[df['Awards Received'] == 0].Title,
                            mode='markers',
                            opacity=0.5,
                            marker={
                                'color': 'gray',
                                'size': 10,
                                'line': {'width': 1, 'color': 'black'}
                            },
                            name='has_award: no'
                        )
                    ],
                    'layout': go.Layout(
                        margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                        xaxis={'title': 'IMDb Score'},
                        yaxis={'title': 'Dollars at Box Office'},
                        hovermode='closest'
                    )
                }
            ),
            html.Br(),
            html.P('Number of rows selected: {}'.format(len(df.index)), id='dataset-rows-p')
        ], style={'marginLeft': 25, 'marginRight': 25})        

        ]),  

])


@app.callback(
    Output('avg_rating_year', 'figure'),
    Input('rating_dropdown', 'value'))
def update_figure(rating):    
    fig = px.line(df_rating, x=df_rating.index, y=df_rating[rating],
                  title = f'Average {rating} of Movies Over Time',
                  range_x=[-0.5,5.5]
    )
    return fig

@app.callback(Output('genre_output', 'figure'),
              Input('genre_dropdown', 'value'), 
              Input('boxoffice-rating', 'value'))
def update_figure_genre(genre, boxoffice_rating):
    df_g = data10[(data10[genre] == 1)]
    if boxoffice_rating == 'Boxoffice':
        gra = df_g[['Release Date', 'Boxoffice']].groupby(['Release Date']).agg({'Boxoffice':np.mean})
        fig = px.line(gra, y= 'Boxoffice')
        fig.update_layout(xaxis_type = 'category')
    else:
        fig = make_subplots(rows=2, cols=2)  
        for r, ro, co in zip(rating_list, [1,1,2,2], [1,2,1,2]):
            df_r = df_g[['Release Date', r]].groupby(['Release Date']).agg({r:np.mean})
            fig.append_trace(go.Scatter(x = df_r.index, y = df_r[r], mode = 'lines', name = r), row = ro, col = co)
        fig.update_layout(height=600, width=1200, title_text="Rating")
    return fig

@app.callback(
    dash.dependencies.Output('movie-bar-chart', 'figure'),
    [dash.dependencies.Input('amount1', 'value')
    ])
def highest_profit(value):
    highest_profit_df = pd.DataFrame(df.sort_values(by='Boxoffice', ascending=False)[['Title', 'Boxoffice']].head(value)).reset_index(drop=True)
    fig = px.bar(highest_profit_df, x='Title', y='Boxoffice')
    return fig


@app.callback(
    dash.dependencies.Output('director-bar-chart', 'figure'),
    [dash.dependencies.Input('amount2', 'value')
    ])
    
def update_bar_chart(value):
    grouped = df.groupby('Director')
    most_earned_director_df = pd.DataFrame(grouped['Boxoffice'].sum()).sort_values(by='Boxoffice', ascending=False).head(value)
    most_earned_director_df.reset_index(inplace=True)
    fig = px.bar(most_earned_director_df, x='Director', y='Boxoffice')
    return fig



@app.callback(
    dash.dependencies.Output('scatter-plot-graph', 'figure'),
    [
        dash.dependencies.Input('year-released-range-slider', 'value'),
        dash.dependencies.Input('awards-won-slider', 'value'),
        dash.dependencies.Input('dollars-boxoffice-range-slider', 'value'),
        dash.dependencies.Input('x-axis-dropdown', 'value'),
        dash.dependencies.Input('y-axis-dropdown', 'value')
    ]
)
def update_scatter_plot(selected_years_released, selected_nb_awards_won,
                        selected_dollars_boxoffice,x_axis_var, y_axis_var):

    year_released_start, year_released_end = selected_years_released or (1910, df.Year.max())
    awards_won = selected_nb_awards_won or df['Awards Received'].min()
    dollars_boxoffice_min, dollars_boxoffice_max = (amount * 1e6 for amount in selected_dollars_boxoffice) or \
        (df.Boxoffice.min(), df.Boxoffice.max())
    x_axis = x_axis_var or 'IMDb Score'
    y_axis = y_axis_var or 'Dollars at Box Office'

    filtered_df = (
        df.pipe(lambda df: df[(df['Year'] >= year_released_start) & (df['Year'] <= year_released_end)])
        .pipe(lambda df: df[df['Awards Received'] >= awards_won])
        .pipe(lambda df: df[(df['Boxoffice'] >= dollars_boxoffice_min) & (df['Boxoffice'] <= dollars_boxoffice_max)])
    )

    return {
        'data': [
            go.Scatter(
                x=filtered_df[filtered_df['Awards Received'] > 0][x_axis],
                y=filtered_df[filtered_df['Awards Received'] > 0][y_axis],
                text=filtered_df[filtered_df['Awards Received'] > 0]['Title'],
                mode='markers',
                opacity=0.5,
                marker={
                    'color': 'orange',
                    'size': 10,
                    'line': {'width': 1, 'color': 'black'}
                },
                name='has_award: yes'
            ),
            go.Scatter(
                x=filtered_df[filtered_df['Awards Received'] == 0][x_axis],
                y=filtered_df[filtered_df['Awards Received'] == 0][y_axis],
                text=filtered_df[filtered_df['Awards Received'] == 0]['Title'],
                mode='markers',
                opacity=0.5,
                marker={
                    'color': 'gray',
                    'size': 10,
                    'line': {'width': 1, 'color': 'black'}
                },
                name='has_award: no'
            )
        ],
        'layout': go.Layout(
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            xaxis={
                'title': variable_labels[x_axis],
                'range': [
                    filtered_df[x_axis].min(),
                    filtered_df[x_axis].max()
                ]
            },
            yaxis={
                'title': variable_labels[y_axis],
                'range': [
                    filtered_df[y_axis].min(),
                    filtered_df[y_axis].max()
                ]},
            hovermode='closest'
        )
    }


@app.callback(
    dash.dependencies.Output('dataset-rows-p', 'children'),
    [
        dash.dependencies.Input('year-released-range-slider', 'value'),
        dash.dependencies.Input('awards-won-slider', 'value'),
        dash.dependencies.Input('dollars-boxoffice-range-slider', 'value'),
        dash.dependencies.Input('x-axis-dropdown', 'value'),
        dash.dependencies.Input('y-axis-dropdown', 'value')
    ]
)
def update_nb_rows_selected(selected_years_released, selected_nb_awards_won,
                            selected_dollars_boxoffice,
                            x_axis_var, y_axis_var):
    year_released_start, year_released_end = selected_years_released or (1910, df.Year.max())
    awards_won = selected_nb_awards_won or df['Awards Received'].min()
    dollars_boxoffice_min, dollars_boxoffice_max = (amount * 1e6 for amount in selected_dollars_boxoffice) or \
        (df.Boxoffice.min(), df.Boxoffice.max())
    x_axis = x_axis_var or 'IMDb Score'
    y_axis = y_axis_var or 'Dollars at Box Office'

    filtered_df = (
        df.pipe(lambda df: df[(df['Year'] >= year_released_start) & (df['Year'] <= year_released_end)])
        .pipe(lambda df: df[df['Awards Received'] >= awards_won])
        .pipe(lambda df: df[(df['Boxoffice'] >= dollars_boxoffice_min) & (df['Boxoffice'] <= dollars_boxoffice_max)])
    )

    return 'Number of rows selected: {}'.format(len(filtered_df.index))

if __name__ == "__main__":
    app.run_server(debug=True)