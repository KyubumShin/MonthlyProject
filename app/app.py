import dash
from dash_html_components.Div import Div
from dash_html_components.P import P
import pandas as pd
import numpy as np

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

import dash_table

import plotly.express as px
import plotly.figure_factory as ff


data = pd.read_csv("./healthcare-dataset-stroke-data.csv")

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Sidebar", className="display-4"),
        html.Hr(),
        html.P(
            "A simple sidebar layout with navigation links", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Page 1", href="/page-1", active="exact"),
                dbc.NavLink("Page 2", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

controls = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("X 축 변수"),
                dcc.Dropdown(
                    id="x-variable",
                    options = [
                        {"label": i, "value": i} for i in ["avg_glucose_level","bmi"]
                    ],
                    value = "avg_glucose_level"
                )
            ]
        )
    ],
    body=True,
)

numberic_graph = dbc.Container(
    [
        html.H1("수치 변수 그래프"),
        html.P("수치형 변수들"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(controls, md = 4),
                dbc.Col(dcc.Graph(id="agl"),md=8)
            ],
            align="center",
        ),
    ],
    fluid=True,
)



external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

content = html.Div([
    numberic_graph
    ], id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), 
    sidebar, 
    content])

@app.callback(Output('agl', 'figure'),
                [Input('x-variable','value')])
def update_graph(xaxis_name):
    df1 = data[data.stroke == 0][xaxis_name].dropna().tolist()
    df2 = data[data.stroke == 1][xaxis_name].dropna().tolist()
    fig = ff.create_distplot([df1,df2], group_labels=["stroke","not stroke"],show_hist=False)
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)