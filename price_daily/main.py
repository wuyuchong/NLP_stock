#!/usr/bin/env python
# -*- coding: utf-8 -*-

# baostock.com
# 股票停牌时，对于日线，开、高、低、收价都相同，且都为前一交易日的收盘价，成交量、成交额为0，换手率为空。

import os
import sys
import baostock as bs
import pandas as pd 

def get_code_name():
    lg = bs.login()
    rs = bs.query_stock_industry()
    print('query_stock_industry error_code:'+rs.error_code)
    print('query_stock_industry respond  error_msg:'+rs.error_msg)
    industry_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        industry_list.append(rs.get_row_data())
    result = pd.DataFrame(industry_list, columns=rs.fields)
    # 结果集输出到csv文件
    result.to_csv("/data/price/code_name.csv", index=False)
    bs.logout()

def get_daily():
    code_name = pd.read_csv("/data/price/code_name.csv", dtype = str)
    lg = bs.login()
    for stock_code in code_name['code']:
        file_name = '/data/price/daily/' + str(stock_code) + '.csv'
        if os.path.isfile(file_name):
            continue
        data_list = []
        rs = bs.query_history_k_data_plus(stock_code, 'date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST,peTTM,pbMRQ,psTTM,pcfNcfTTM', start_date = '2000-01-01', end_date = '2021-03-01', frequency="d", adjustflag="1")
        if rs.error_code != '0':
            print(rs.error_code)
            sys.exit()
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            data_list.append(rs.get_row_data())
        result = pd.DataFrame(data_list, columns=rs.fields)
        result.to_csv(file_name, index=False)
        print('get', stock_code)
    bs.logout()

def get_market():
    # 登陆系统
    lg = bs.login()
    # 显示登陆返回信息
    rs = bs.query_history_k_data_plus("sh.000016",
        "date, close",
        start_date='2000-01-01', end_date='2021-05-31', frequency="d")
    print('query_history_k_data_plus respond error_code:'+rs.error_code)
    print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)
    # 打印结果集
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    # 结果集输出到csv文件
    result.to_csv("/data/price/market.csv", index=False)
    print(result)
    # 登出系统
    bs.logout()

get_daily()

