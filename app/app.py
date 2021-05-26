import dash
import pandas as pd
import numpy as np

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

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
                dbc.NavLink("EDA", href="#EDA", active="exact"),
                dbc.NavLink("Conclusion", href="#conclusion", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

controls_num = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.FormGroup([
                dbc.Label("카테고리 범주"),
                dcc.Dropdown(
                    id="hue-variable",
                    options=[
                        {"label": x, "value": i} for i, x in enumerate(["all", "stroke", "not stroke"])
                    ],
                    value=0
                )
            ]),
        ]),
        dbc.Col([
            dbc.FormGroup([
                dbc.Label("Y 축 변수"),
                dcc.Dropdown(
                    id="x-variable",
                    options=[
                        {"label": i, "value": i} for i in ["avg_glucose_level", "bmi"]
                    ],
                    value="avg_glucose_level"
                )
            ]),
        ]),
    ])
])


controls_cat = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.FormGroup([
                dbc.Label("카테고리 범주"),
                dcc.Dropdown(
                    id="hue-variable2",
                    options=[
                        {"label": i, "value":i} for i in ["hypertension", "heart_disease", "smoking_status"]
                    ],
                    value="hypertension"
                )
            ]),
        ]),
    ])
])


controls = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.FormGroup([
                dbc.Label("카테고리 범주"),
                dcc.Dropdown(
                    id="hue-variable3",
                    options=[
                        {"label": i, "value":i} for i in ["gender", "ever_married", "work_type", "Residence_type"]
                    ],
                    value="gender"
                )
            ]),
        ]),
    ])
])

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


# 1장
dataload = dbc.Container(
    [
        html.H1("1. 사용 데이터셋"),
        html.Hr(),
        html.H3("뇌졸증 예측 데이터셋(Stroke Prediction Dataset)"),
        html.H4("11가지의 임상 특징을 이용한 뇌졸증 예측"),
        dbc.Card(
            dbc.Row([
            dbc.Col(dbc.Button("데이터셋 바로가기", href="https://www.kaggle.com/fedesoriano/stroke-prediction-dataset",size="lg"),md = 2),
            dbc.Col(dbc.Button("Github",href = "https://github.com/KyubumShin/MonthlyProject",size="lg"), md = 2)
            ], justify="center"),style={"padding" : "10px"}
        ),
        html.Hr(),
    ],
    fluid=True,
)

# 2장

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
                    dbc.CardBody(html.H4("나이가 증가함에 따라 데이터 안의 뇌졸증의 빈도가 급격하게 증가하는것을 확인할 수 있었다")),
                ]))
            ],align="center")
        ],fluid=True,),
        html.Br(),
        dbc.Container([
            html.H3("2. 혈관에 관련된 변수들과의 상관관계를 조사하자"),
            html.Hr(),
            html.Ul([
                html.Li("Average Glucose Level"),
                html.Li("BMI"),
                html.Li("Hypertension"),
                html.Li("Heart Disease"),
                html.Li("Smoking Status"),
            ]),
            html.Hr(),
            html.H5("Average Glucose Level / Stroke"),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="agl", figure=num_graph("avg_glucose_level")),md=7),
                    dbc.Col(dbc.Card([
                        dbc.CardHeader(html.H3("Result")),
                        dbc.CardBody(html.H4("Average Glucose Level / Stroke에 경우에는 뇌졸증인 사람들의 평균 혈당치가 조금 더 높은 경향을 보이기는 하나, 가장 큰 특징이라고 생각하기에는 충분하지 않다"))
                        ]))
                ],
                align="center",
            ),
            html.Br(),
            html.H5("BMI / Stroke"),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="bmi", figure=num_graph("bmi")),md=7),
                    dbc.Col(dbc.Card(
                        [
                            dbc.CardHeader(html.H3("Result")),
                            dbc.CardBody(html.H4("bmi 수치는 stroke인 사람들과 아닌사람들 모두 비슷한 경향의 그래프를 보여서 특징을 찾을 수 없었다"))
                        ]))
                ],
                align="center",
            ),
            html.Br(),
            html.H5("BMI,Average Glucose Level / Age and Stroke"),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(controls_num),
                            dbc.CardBody(dcc.Graph(id="scat_agl"))
                        ]),
                    ], md = 7),
                    dbc.Col(dbc.Card([
                        dbc.CardHeader(html.H3("Result")),
                        dbc.CardBody(html.H4("bmi와 뇌졸증의 관계에서 특징적인 상관관계를 찾는것이 힘들었고, 평균혈당치도 어느정도 영향을 끼치지만, 뇌졸증에 가장 큰 영향을 끼치는것은 나이라고 생각되어진다"))
                    ]))
                ],
                align="center",
            ),
            html.Br(),
            html.H5("Hypertension, Heart Disease, Smoking Status / Stroke"),
            html.Br(),
            dbc.Row([
                dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(controls_cat),
                            dbc.CardBody(dcc.Graph(id="hyper"))
                        ]),
                ], md = 7),
                dbc.Col(dbc.Card([
                    dbc.CardHeader(html.H3("Result")),
                    dbc.CardBody(html.H4("Hypertension, Heart Disease를 가지고 있는 사람들의 뇌졸증 비율이 가지지 사람에 비해 월등히 높았고, Smoking Status은 생각 외로 각 카테고리에 비해 차이가 크지 않았다"))
                ]))
            ],align="center",),
            html.Br(),
            dbc.Row()
        ],fluid=True),
        dbc.Container([
            html.H3("3. 그 외의 변수들에 대해 조사해보자"),
            html.Hr(),
            html.Ul([
                html.Li("Gender"),
                html.Li("Ever Married"),
                html.Li("Work type"),
                html.Li("Residence type"),
            ]),
            html.Hr(),
            html.H5("Gender, Ever Married, Work type, Residence type / Stroke"),
            html.Br(),
            dbc.Row([
                dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(controls),
                            dbc.CardBody(dcc.Graph(id="else"))
                        ]),
                ], md = 7),
                dbc.Col(dbc.Card([
                    dbc.CardHeader(html.H3("Result")),
                    dbc.CardBody(html.H4("Ever Married와 Work type에서 유의미한 Feature를 볼 수 있었다. Ever Married에서는 기혼자가 더 높은 뇌졸증 비율을 가지고 있는것을 알 수 있고, Work type에서는 Self-employed가 다른 항목보다 높은것을 알 수 있었다."))
                ]))
            ],align="center"),
            html.Br(),
        ],fluid=True),
        html.Br(),
    ],fluid=True,
)

# scatter 조절함수
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

@app.callback(
    Output("hyper","figure"),
    Input("hue-variable2", "value")
)
def pie_graph1(value):
    ddata = data.groupby(value).mean()
    fig = px.pie(ddata, values = "stroke", names = ddata.index)
    return fig

@app.callback(
    Output("else","figure"),
    Input("hue-variable3", "value")
)
def pie_graph2(value):
    ddata = data.groupby(value).mean()
    fig = px.pie(ddata, values = "stroke", names = ddata.index)
    return fig


#4장
result = dbc.Container([
    html.H1("4. 결론"),
    html.Hr(),
    html.H3("1. Age와 Stroke의 상관관계가 매우 크다는것을 발견하였다"),
    html.Br(),
    html.H3("2. 혈관과 관련된 항목중 BMI와 Smoking Status를 제외한 나머지 항목들은 모두 예상했던것처럼 유의미한 Feature를 가진것을 확인했다"),
    html.Ul([
        html.Li("Smoking Status는 항목당 큰 차이는 없었으나 일반적인 상식상, 흡연은 뇌졸증에 큰 영향을 미치는것으로 판명되어있으므로, 데이터가 더 모이면 유의미한 Feature를 보여줄것이라 생각된다"),
        html.Li("BMI에 대해서는 비만도따른 합병증으로 인하여 뇌졸증의 비율도 늘어날 것이라고 생각을 하였는데 생각 외로 그러한 Feature는 보이지 않았다. 이부분에 대해서 조금 더 다른 상관관계들과 연계하여 찾아볼 필요가 있다고 생각된다.")
    ]),
    html.H3("그 외의 항목에서는 Ever Married, Work Type에서 유의미한 Feature를 발견할 수 있었다"),
    html.Ul([
        html.Li("Ever Married에서 기혼자들이 더 높은 뇌졸증 비율을 보였는데 연령대가 높아질수록 기혼비율이 늘어났기 때문이라고 생각되어진다"),
        html.Li("Work Type에서는 예상하지 못한 Feature를 발견했는데 이부분에 대해서는 조금 더 생각을 해 보아야 할 필요가 있다")
    ]),
    dbc.Row(
        
    )
], fluid=True)

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
        html.A(id = "conclusion"),
        result
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
    app.run_server()
