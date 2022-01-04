#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.auto import get_stock_list, get_model_list, get_model_type
from src.auto import submit_train_lsi, submit_train_doc2vec
from src.auto import query_sim_lsi, query_sim_doc2vec

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as px

# Diskcache
#  import diskcache
#  from dash.long_callback import DiskcacheLongCallbackManager
#  cache = diskcache.Cache("./cache")
#  long_callback_manager = DiskcacheLongCallbackManager(cache)

# Celery Redis
#  from dash.long_callback import CeleryLongCallbackManager
#  from celery import Celery
#  celery_app = Celery(
    #  __name__, broker="redis://localhost:6380/0", backend="redis://localhost:6380/1"
#  )
#  long_callback_manager = CeleryLongCallbackManager(celery_app)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#  external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP]
#  app = dash.Dash(__name__, external_stylesheets=external_stylesheets, long_callback_manager=long_callback_manager)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


# -----------------> layout
app.layout = html.Div(children=[
    html.H1(children='股市新闻及评论文本相似查询及训练平台'),
    html.Div(children=[
        html.Hr(),
        html.H2(children='选择内容'),
        dcc.Dropdown(id='content_type', options=[
            {'label': '文章', 'value': 'wemedia'},
            {'label': '新闻', 'value': 'news'},
        ], placeholder='选择文本类型'),
        dcc.Dropdown(id='stock_list', placeholder='选择股票名称'),
        html.Hr(),
        html.H2(children='选择模型'),
        dcc.Dropdown(id='model_type', placeholder='选择模型类型'),
        dcc.Dropdown(id='model_list', placeholder='选择模型参数'),
        html.Hr(),
        html.H2(children='查询相似'),
        dcc.Input(id='input_content', type='text', placeholder='输入查询文本',
                  style={'width': '100%'}),
        html.Br(),
        html.Button(id='start_query', n_clicks=0, children='开始查询'),
        html.Div(id='prompt'),
        html.Hr(),
        html.H2(children='模型训练'),
        html.Br(),
        html.Div(children='模型提交后排队依次完成，未来版本将支持多线程同时训练'),
        html.Br(),
        html.Div(children=[
            html.H3(children='LSI 模型'),
            html.Br(),
            dcc.Input(id='num_topics', type='number', placeholder='主题个数'),
            html.Br(),
            html.Button(id='start_train_lsi', n_clicks=0, children='开始训练 LSI'),
            html.Div(id='prompt_lsi'),
            html.Br(),
            html.Div(id='train_lsi'),
            html.Br(),
        ], style={'width': '50%', 'float': 'left', 'display':'inline-block'}),
        html.Div(children=[
            html.H3(children='doc2vec 模型'),
            html.Br(),
            dcc.Input(id='vector_size', type='number', placeholder='向量维度'),
            dcc.Input(id='min_count', type='number', placeholder='最小词频率'),
            dcc.Input(id='epochs', type='number', placeholder='迭代次数'),
            html.Br(),
            html.Button(id='start_train_doc2vec', n_clicks=0, children='开始训练 doc2vec'),
            html.Div(id='prompt_doc2vec'),
            html.Br(),
            html.Div(id='train_doc2vec'),
            html.Br(),
        ], style={'width': '50%', 'float': 'right', 'display':'inline-block'}),
    ], style={'width': '40%', 'float': 'left', 'display':'inline-block'}),
    html.Div(children=[
        html.Hr(),
        html.Div(children='©️ 版权归马景义课题组所有'),
        html.Div(children='项目代码量：1950 行，文本库大小：34 GB'),
        html.Div(children='项目 github 地址: https://github.com/wuyuchong/nlp_stock'),
        html.Div(children='运行报错请反馈至 email@wuyuchong.com'),
        html.Hr(),
        dcc.Tabs(id="tabs", value='MOST', children=[
            dcc.Tab(label='最相似', value='MOST', children=[
                html.Div(id='MOST')]),
            dcc.Tab(label='次相似', value='SECOND-MOST', children=[
                html.Div(id='SECOND-MOST')]),
            dcc.Tab(label='中位数', value='MEDIAN', children=[
                html.Div(id='MEDIAN')]),
            dcc.Tab(label='最不似', value='LEAST', children=[
                html.Div(id='LEAST')]),
    ]),
    ], style={'width': '50%', 'float': 'right', 'display':'inline-block'}),
])


# -----------------> stock list
@app.callback(Output('stock_list', 'options'),
              Input('content_type', 'value'))
def update_output(content_type):
    if not content_type:
        stock_list = ['请先选择内容类型']
    else:
        stock_list = get_stock_list(content_type)
    return [{'label': i, 'value': i} for i in stock_list]


# -----------------> model type
@app.callback(Output('model_type', 'options'),
              Input('content_type', 'value'),
              Input('stock_list', 'value'))
def update_output(content_type, stock_name):
    try:
        model_type = get_model_type(content_type, stock_name)
    except:
        model_type = ['无可用模型，请先训练']
    return [{'label': i, 'value': i} for i in model_type]


# -----------------> model list
@app.callback(Output('model_list', 'options'),
              Input('content_type', 'value'),
              Input('stock_list', 'value'),
              Input('model_type', 'value'))
def update_output(content_type, stock_name, model_type):
    try:
        model_list = get_model_list(content_type, stock_name, model_type)
    except:
        model_list = ['无可用模型，请先训练']
    return [{'label': i, 'value': i} for i in model_list]


# -----------------> prompt
@app.callback(Output('prompt', 'children'),
              Input('start_query', 'n_clicks'))
def update_output(n_clicks):
    if n_clicks == 0:
        return ''
    return u'''已开始第{}次查询，在结果刷新之前请勿重复点击'''.format(n_clicks)


# -----------------> query
@app.callback(Output('MOST', 'children'),
              Output('SECOND-MOST', 'children'),
              Output('MEDIAN', 'children'),
              Output('LEAST', 'children'),
              Input('start_query', 'n_clicks'),
              State('content_type', 'value'),
              State('model_type', 'value'),
              State('model_list', 'value'),
              State('stock_list', 'value'),
              State('input_content', 'value'))
def update_output(n_clicks, content_type, model_type, model_token, stock_name, input_content):
    if n_clicks == 0 or input_content == '' or not input_content:
        return (('', ) * 4)
    try:
        if model_type == 'lsi':
            outcome = query_sim_lsi(content_type, stock_name, input_content, model_token=model_token)
        elif model_type == 'doc2vec':
            outcome = query_sim_doc2vec(content_type, stock_name, input_content, model_token=model_token)
        return (
            outcome['MOST'], outcome['SECOND-MOST'],
            outcome['MEDIAN'], outcome['LEAST'])
    except:
        return (('出错了，请检查输入是否正确，有问题请反馈到 email@wuyuchong.com', ) * 4)


# -----------------> prompt lsi
@app.callback(Output('prompt_lsi', 'children'),
              Input('start_train_lsi', 'n_clicks'),
              State('content_type', 'value'),
              State('stock_list', 'value'),
              State('num_topics', 'value'))
def update_output(n_clicks, content_type, stock_name, num_topics):
    if n_clicks == 0:
        return ''
    if n_clicks >= 4:
        return '请求被驳回，请勿同时训练过多模型'
    if not num_topics:
        return '请先输入模型参数'
    if not content_type:
        return '请先选择文本类型'
    if not stock_name:
        return '请先选择股票名称'
    submit_train_lsi(content_type, stock_name, num_topics)
    return u'''已提交到后台进行训练，内容类型：{}，股票名称：{}，主题个数：{}, 
        请勿重复提交相同的模型。
        根据个股预料库大小耗时几分钟至几小时不等。
        可关闭此页面，训练结束后可查询到相应模型。'''.format(content_type, stock_name, num_topics)


# -----------------> prompt doc2vec
@app.callback(Output('prompt_doc2vec', 'children'),
              Input('start_train_doc2vec', 'n_clicks'),
              State('content_type', 'value'),
              State('stock_list', 'value'),
              State('vector_size', 'value'),
              State('min_count', 'value'),
              State('epochs', 'value'))
def update_output(n_clicks, content_type, stock_name, vector_size, min_count, epochs):
    if n_clicks == 0:
        return ''
    if n_clicks >= 4:
        return '请求被驳回，请勿同时训练过多模型'
    if not vector_size or not min_count or not epochs:
        return '请先输入模型参数'
    if not content_type:
        return '请先选择文本类型'
    if not stock_name:
        return '请先选择股票名称'
    submit_train_doc2vec(content_type, stock_name, vector_size=vector_size, min_count=min_count, epochs=epochs)
    return u'''已提交到后台进行训练，内容类型：{}，股票名称：{}，向量维度：{}，最小词频率：{}，迭代次数：{}。
        请勿重复提交相同的模型。
        根据个股预料库大小耗时几分钟至几小时不等。
        可关闭此页面，训练结束后可查询到相应模型。'''.format(content_type, stock_name, vector_size, min_count, epochs)


if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0')
