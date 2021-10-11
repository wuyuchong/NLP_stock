#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json


def print_document(data_dir, file_name):
    f = open(data_dir + file_name + '.txt')
    contents = f.read()
    dictionary = json.loads(contents)
    content = dictionary['content']
    f.close()
    print('----------------')
    print(content)

