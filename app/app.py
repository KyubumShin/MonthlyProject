import dash
from dash_bootstrap_components._components.CardBody import CardBody
from dash_bootstrap_components._components.CardHeader import CardHeader
from dash_bootstrap_components._components.Row import Row
from dash_core_components.Dropdown import Dropdown
from dash_html_components.A import A
from dash_html_components.Col import Col
from dash_html_components.Div import Div
from dash_html_components.H1 import H1
from dash_html_components.H3 import H3
from dash_html_components.H4 import H4
from dash_html_components.Hr import Hr
from dash_html_components.P import P
from dash_html_components.Th import Th
import pandas as pd
import numpy as np

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import dash_table
from pandas.io.formats import style

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
    "scroll-behavior" : "smooth"
}

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "scroll-behavior" : "smooth"
}

CARD_STYLE = {

}

def num_graph(xaxis_name):
    df1 = data[data.stroke == 0][xaxis_name].dropna().tolist()
    df2 = data[data.stroke == 1][xaxis_name].dropna().tolist()
    fig = ff.create_distplot([df1,df2], group_labels=["not stroke","stroke"],show_hist=False,show_rug=False)
    return fig

sidebar = html.Div(
    [
        html.H2("Shin Kyubum", className="display-4"),
        html.Hr(),
        html.P(
            "Monthly EDA Project", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Dataset", href="#dataload", active="exact"),
                dbc.NavLink("Dataset confirm", href="#dataconfirm", active="exact"),
                dbc.NavLink("EDA", href="#EDA", active="exact", external_link=False),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

controls_num = dbc.Row([
            dbc.Col([
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
            ]),
            dbc.Col([
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
            ]),
        ])


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
stylesheet = dbc.themes.BOOTSTRAP
app = dash.Dash(__name__, external_stylesheets=[stylesheet])

dataload = dbc.Container(
    [
        html.H1("1. 사용 데이터셋"),
        html.Hr(),
        html.H3("뇌졸증 예측 데이터셋(Stroke Prediction Dataset)"),
        html.H4("11가지의 임상 특징을 이용한 뇌졸증 예측"),
        dbc.Row([
            dbc.Button("데이터셋 바로가기", href="https://www.kaggle.com/fedesoriano/stroke-prediction-dataset")
        ], justify="center"),
        html.Hr(),
    ],
    fluid=True,
)

cattable_header = [
    html.Thead(html.Tr([html.Th("Category"), html.Th("Type"), html.Th("Detail")]))
]

cattable_body = [html.Tbody([
    html.Tr([html.Td("gender"), html.Td("object"), html.Td("성별")]),
    html.Tr([html.Td("age"), html.Td("float64"), html.Td("나이")]),
    html.Tr([html.Td("hypertension"), html.Td("int64"), html.Td("고혈압의 유무")]),
    html.Tr([html.Td("heart disease"), html.Td("int64"), html.Td("심장질환의 유무")]),
    html.Tr([html.Td("ever married"), html.Td("object"), html.Td("결혼 상태")]),
    html.Tr([html.Td("work type"), html.Td("object"), html.Td("직장 (children : 아이, Govt_jov : 공무원, Never_worked : 무직, Private : 사적, Self-employed : 자영업)")]),
    html.Tr([html.Td("Residence_type"), html.Td("object"), html.Td("거주지 (Urban : 도시, Rural : 농촌)")]),
    html.Tr([html.Td("avg glucose level"), html.Td("float64"), html.Td("평균 혈당치")]),
    html.Tr([html.Td("bmi"), html.Td("float64"), html.Td("bmi 수치")]),
    html.Tr([html.Td("smoking status"), html.Td("object"), html.Td("흡연 상태 (formerly smoked, never smoked, smokes, Unknown)")]),
    html.Tr([html.Td("stroke"), html.Td("int64"), html.Td("뇌졸증 유무")])
])]

cattable = dbc.Table(cattable_header + cattable_body, bordered=True)

settable = dbc.Table.from_dataframe(data.head(5), striped=True, bordered=True, hover=True)

dataconfirm = dbc.Container(
    [
        html.H1("2. 데이터 확인하기"),
        html.Hr(),
        dbc.Card([
            dbc.CardHeader(
                html.H2(
                    dbc.Button(
                        "데이터 범주 확인하기",
                        color = "link",
                        id = "toggle-datacat"
                    )
                )
            ),
            dbc.Collapse(
                dbc.CardBody(cattable),
                id = "collapse-datacat"
            )
        ]),
        dbc.Card([
            dbc.CardHeader(
                html.H2([
                    dbc.Row([
                        dbc.Col(dbc.Button("데이터셋 확인하기", color = "link", id = "toggle-dataset")),
                        dbc.Col(dbc.Input(id = "input", placeholder="5", type="number"), md = 2)
                    ])
                ])
            ),
            dbc.Collapse(
                dbc.CardBody(settable),
                id = "collapse-dataset"
            )
        ]),
        html.Hr(),
    ],
    fluid=True,
)

@app.callback(
    Output("collapse-datacat", "is_open"),
    Input("toggle-datacat", "n_clicks"),
    State("collapse-datacat", "is_open")
)
def toggle_accordion(n, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return False
    return not is_open

@app.callback(
    Output("collapse-dataset", "is_open"),
    Input("toggle-dataset", "n_clicks"),
    State("collapse-dataset", "is_open")
)
def toggle_accordion(n, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return False
    return not is_open


# 3장

data_EDA = dbc.Container(
    [
        html.H1("3. 데이터 분석"),
        html.Br(),
        dbc.Container([
            html.H3("1. 나이가 들면 자연스럽게 여러가지 생체 지표가 떨어지기 시작한다. 나이와 뇌졸증의 상관관계를 조사해보자"),
            html.Hr(),
            html.H5("age / stroke kdeplot"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=num_graph("age")), md = 6),
                dbc.Col(dbc.Card([
                    dbc.CardHeader(html.H3("Result")),
                    dbc.CardBody(html.H4("나이가 증가함에 따라 데이터 안의 뇌졸증의 빈도가 급격하게 증가하는것을 확인할 수 있었다.")),
                ]))
            ],align="center")
        ],fluid=True,),
        html.Br(),
        dbc.Container([
            html.H3("2. 혈관에 관련된 변수들과의 상관관계를 조사하자"),
            html.Hr(),
            html.H5("Average Glucose Level / Stroke"),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="agl", figure=num_graph("avg_glucose_level")),md=6),
                    dbc.Col(dbc.Card([
                        dbc.CardHeader(html.H3("Result")),
                        dbc.CardBody(html.H4("평균 글루코스 수치 / 뇌졸증에 경우에는 뇌졸증인 사람들의 평균 글루코스 수치가 조금 더 높은 경향을 보이기는 하나, 확실한 특징이라고 생각하기에는 충분하지 않다"))
                        ]))
                ],
                align="center",
            ),
            html.Br(),
            html.H5("BMI / Stroke"),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="bmi", figure=num_graph("bmi")),md=6),
                    dbc.Col(dbc.Card(
                        [
                            dbc.CardHeader(html.H3("Result")),
                            dbc.CardBody(html.H4("bmi 수치는 stroke인 사람들과 아닌사람들 모두 비슷한 경향의 그래프를 보여서 특징을 찾을 수 없었다."))
                        ]))
                ],
                align="center",
            ),
            dbc.Row(
                [
                    dbc.Col([
                        dbc.Row(controls_num),
                        dbc.Col(dcc.Graph(id="scat_agl"))
                    ], md = 6)
                ],
                align="center",
            ),
        ],fluid=True),
    ],fluid=True,
)

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
    fig = px.scatter(ddata, x="age", y=xaxis_value, color="stroke", color_continuous_scale=[(0.00, "orange"),(0.50, "orange"),(0.50, "blue"), (1.00, "blue")], opacity=0.5)
    return fig


content = html.Div([
        dbc.Row([
            html.H1("Montly EDA Project", style={"font-size" : "70px"}),
        ],justify= "center", 
        ),
        html.A(id = "dataload"),
        dataload,
        html.A(id = "dataconfirm"),
        dataconfirm,
        html.A(id = "EDA"),
        data_EDA,
        numberic_graph,
    ], id="page-content", style=CONTENT_STYLE
)

app.layout = html.Div(
    [
        dcc.Location(id="url"), 
        sidebar, 
        content,
    ]
)

if __name__ == '__main__':
    app.run_server(debug=True)
