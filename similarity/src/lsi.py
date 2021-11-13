#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import gensim


def get_dictionary(seg_dir, stoplists):
    dictionary = gensim.corpora.Dictionary([])
    file_names = os.listdir(seg_dir)
    for file_name in file_names:
        fileload = open(seg_dir + file_name)
        data_all = json.load(fileload)
        fileload.close()
        if 'tok/fine' in data_all.keys():
            dat = [c for c in data_all.get('tok/fine') if c not in stoplists]
        else:
            dat = ['该文本没有内容']
        dictionary.add_documents([dat])
    return dictionary


def yield_corpus(seg_dir, stoplists, dictionary):
    file_names = os.listdir(seg_dir)
    for file_name in file_names:
        fileload = open(seg_dir + file_name)
        data_all = json.load(fileload)
        fileload.close()
        if 'tok/fine' in data_all.keys():
            dat = [c for c in data_all['tok/fine'] if c not in stoplists]
        else:
            dat = ['该文本没有内容']
        yield dictionary.doc2bow(dat)


def sim_label_file_names(seg_dir, stoplists, dictionary, corpus, lsi,
                         input_file_name, config=False):
    file_names_json = os.listdir(seg_dir)
    fileload = open(seg_dir + input_file_name + '.json')
    data_all = json.load(fileload)
    fileload.close()

    if 'tok/fine' in data_all.keys():
        doc = [c for c in data_all.get('tok/fine') if c not in stoplists]
    else:
        doc = ['该文本没有内容']

    vec_bow = dictionary.doc2bow(doc)
    vec_lsi = lsi[vec_bow]  # convert the query to LSI space
    # transform corpus to LSI space and index it
    index = gensim.similarities.MatrixSimilarity(lsi[corpus])
    sims = index[vec_lsi]  # perform a similarity query against the corpus
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    if not config:
        config = [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2),
                  ('LEAST', len(sims) - 1)]
    outcome = list()
    outcome.append(('INPUT', input_file_name))
    for label, index in config:
        outcome.append((label,
                        file_names_json[sims[index][0]].replace('.json', '')))
    return outcome


def query_sim_label_file_names(seg_dir, dictionary, corpus, lsi,
                             input_words_list, config=False):
    file_names_json = os.listdir(seg_dir)
    vec_bow = dictionary.doc2bow(input_words_list)
    vec_lsi = lsi[vec_bow]  # convert the query to LSI space
    # transform corpus to LSI space and index it
    index = gensim.similarities.MatrixSimilarity(lsi[corpus])
    sims = index[vec_lsi]  # perform a similarity query against the corpus
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    if not config:
        config = [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2),
                  ('LEAST', len(sims) - 1)]
    outcome = list()
    for label, index in config:
        outcome.append((label,
                        file_names_json[sims[index][0]].replace('.json', '')))
    return outcome
