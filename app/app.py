import dash
from dash_html_components.Div import Div
from dash_html_components.H3 import H3
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

def num_graph(xaxis_name):
    df1 = data[data.stroke == 0][xaxis_name].dropna().tolist()
    df2 = data[data.stroke == 1][xaxis_name].dropna().tolist()
    fig = ff.create_distplot([df1,df2], group_labels=["stroke","not stroke"],show_hist=False)
    return fig

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

controls_num = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("카테고리 범주"),
                dcc.Dropdown(
                    id="hue-variable",
                    options = [
                        {"label": x, "value": i} for i ,x in enumerate(["all","stroke","not stroke"])
                    ],
                    value = 0
                )
            ]
        ),
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
        ),

    ],
    body=True,
)

controls_cat = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("X 축 변수"),
                dcc.Dropdown(
                    id = "x-variable-num",
                    options = [
                        {"label": i, "value":i} for i in ["gender", "ever_married", "work_type", "Residence_type"]
                    ],
                    value = "gender"
                )
            ]
        )
    ]
)

numberic_graph = dbc.Container(
    [
        html.H1("수치 변수 그래프"),
        html.P("평균 글루코스 수치, bmi 수치에 따른 뇌졸증 유무"),
        html.Hr(),
        html.H3("평균 글루코스 수치 / 뇌졸증 그래프"),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="agl", figure=num_graph("avg_glucose_level")),md=6)
            ],
            align="center",
        ),
        html.H3("bmi 수치 / 뇌졸증 그래프"),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="bmi", figure=num_graph("bmi")),md=6)
            ]
        ),
        dbc.Row(
            [
                dbc.Col(controls_num, md = 3),
                dbc.Col(dcc.Graph(id="scat_agl"), md = 9)
            ],
            align="center",
        ),
    ],
    fluid=True,
)

categoric_graph = dbc.Container(
    [
        html.H1("범주형 변수 그래프"),
        html.P("범주형 변수/뇌졸증"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(controls_cat, md = 3)
            ]
        )
    ]
)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

@app.callback(
    Output("scat_agl","figure"),
    [
        Input("hue-variable", "value"),
        Input("x-variable", "value")
    ]
)
def scatter_graph(value, xaxis_value):
    if value == 0:
        ddata = data
    elif value == 1:
        ddata = data[data["stroke"] == 1]
    else:
        ddata = data[data["stroke"] == 0]
    fig = px.scatter(ddata, x="age", y=xaxis_value, color="stroke", color_continuous_scale=[(0.00, "pink"),(0.50, "pink"),(0.50, "green"), (1.00, "green")])
    return fig

content = html.Div([
    html.H1("Montly EDA Project"),
    numberic_graph
    ], id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), 
    sidebar, 
    content]
)

if __name__ == '__main__':
    app.run_server(debug=True)
