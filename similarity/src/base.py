#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json


def get_document_content(data_dir, file_name):
    f = open(data_dir + file_name + '.txt')
    contents = f.read()
    dictionary = json.loads(contents)
    content = dictionary['content']
    f.close()
    return(content)


def get_stoplists():
    f = open('stopwords/baidu_stopwords.txt')
    lst = f.read().splitlines()
    f.close()
    f = open('stopwords/cn_stopwords.txt')
    lst += f.read().splitlines()
    f.close()
    f = open('stopwords/hit_stopwords.txt')
    lst += f.read().splitlines()
    f.close()
    f = open('stopwords/scu_stopwords.txt')
    lst += f.read().splitlines()
    f.close()
    return lst


def write_json(content_type, stock_name, model_type,
               input_file_name, sim_label_file_names):
    data_dir = '/data/' + content_type + '/' + stock_name + '/'
    outcome = dict()
    for label, file_name in sim_label_file_names:
        outcome[label] = get_document_content(data_dir, file_name)
    # mkdir
    try:
        os.makedirs('/similarity/output/' + content_type + '/' + stock_name
                    + '/' + model_type + '/')
    except FileExistsError:
        pass
    # write json
    write_dir = '/similarity/output/' + content_type + '/' + stock_name \
        + '/' + model_type + '/' + input_file_name + '.json'
    with open(write_dir, 'w+', encoding='utf-8') as f:
        json.dump(outcome, f, ensure_ascii=False, indent=4)
