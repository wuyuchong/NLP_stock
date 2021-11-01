#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import gensim


def get_sim_file_name(top_n, seg_dir, stoplists, dictionary, corpus, lsi, file_name):
    file_names_json = os.listdir(seg_dir)
    fileload = open(seg_dir + file_name + '.json')
    data_all = json.load(fileload)
    fileload.close()

    if 'tok/fine' in data_all.keys():
        doc = [c for c in data_all.get('tok/fine') if c not in stoplists]
    else:
        doc = ['该文本没有内容']

    vec_bow = dictionary.doc2bow(doc)
    vec_lsi = lsi[vec_bow]  # convert the query to LSI space
    index = gensim.similarities.MatrixSimilarity(lsi[corpus])  # transform corpus to LSI space and index it
    sims = index[vec_lsi]  # perform a similarity query against the corpus
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    return file_names_json[sims[top_n][0]].replace('.json', '')
