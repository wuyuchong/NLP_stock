#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json


def get_document_content(data_dir, file_name):
    f = open(data_dir + file_name + '.txt')
    contents = f.read()
    dictionary = json.loads(contents)
    content = dictionary['content']
    f.close()
    return(content)
