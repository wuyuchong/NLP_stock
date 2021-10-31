#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os


def get_corpus(seg_dir, stoplists, dictionary):
    corpus = list()
    file_names = os.listdir(seg_dir)
    for file_name in file_names:
        fileload = open(seg_dir + file_name)
        data_all = json.load(fileload)
        fileload.close()
        if 'tok/fine' in data_all.keys():
            dat = [c for c in data_all['tok/fine'] if c not in stoplists]
        corpus.append(dictionary.doc2bow(dat))
    return corpus
