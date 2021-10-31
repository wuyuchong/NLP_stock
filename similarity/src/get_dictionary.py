#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gensim
import json
import os


def get_dictionary(seg_dir, stoplists):
    dictionary = gensim.corpora.Dictionary([])
    file_names = os.listdir(seg_dir)
    for file_name in file_names:
        fileload = open(seg_dir + file_name)
        data_all = json.load(fileload)
        fileload.close()
        if 'tok/fine' in data_all.keys():
            dat = [c for c in data_all.get('tok/fine') if c not in stoplists]
        dictionary.add_documents([dat])
    return dictionary
