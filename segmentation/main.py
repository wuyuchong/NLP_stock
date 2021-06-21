#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hanlp
import json
import ast
import sys
import os

HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)

dir_names = [f for f in os.listdir('/data/news/') if not os.path.isfile(os.path.join('/data/news/', f))]
for dir_name in dir_names:
    dir_path = os.path.join('/data/news/', dir_name)
    file_names = os.listdir(dir_path)
    for file_name in file_names:
        # mkdir
        try:
            os.makedirs("/segmentation/news/title/" + dir_name)
            os.makedirs("/segmentation/news/content/" + dir_name)
        except FileExistsError:
            pass
        # read file
        file = open(os.path.join(dir_path, file_name))
        contents = file.read()
        dictionary = ast.literal_eval(contents)
        title = dictionary['title']
        content = dictionary['content']
        file.close()
        # segmentation for title
        file_path_json = os.path.join('/segmentation/news/title/' + dir_name, file_name.replace('.txt', '.json'))
        if not os.path.isfile(file_path_json):
            data = HanLP(title, tasks=['tok/fine', 'tok/coarse'])
            with open(file_path_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print('.', end='', flush=True)
        else:
            print('^', end='', flush=True)
        # segmentation for content
        file_path_json = os.path.join('/segmentation/news/content/' + dir_name, file_name.replace('.txt', '.json'))
        if not os.path.isfile(file_path_json):
            data = HanLP(content, tasks=['tok/fine', 'tok/coarse'])
            with open(file_path_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print('.', end='', flush=True)
        else:
            print('^', end='', flush=True)
    print('| finish', dir_name)

print('finish all the directories')







