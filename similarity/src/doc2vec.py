#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import gensim
import collections


def yield_corpus(seg_dir, stoplists, tokens_only=False):
    file_names = os.listdir(seg_dir)
    for i, file_name in enumerate(file_names):
        fileload = open(seg_dir + file_name)
        data_all = json.load(fileload)
        fileload.close()
        if 'tok/fine' in data_all.keys():
            tokens = [c for c in data_all['tok/fine'] if c not in stoplists]
        else:
            tokens = ['该文本没有内容']
        if tokens_only:
            yield tokens
        else:
            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


def check_model_health(train_corpus, model):
    ranks = []
    second_ranks = []
    for doc_id in range(len(train_corpus)):
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)
        second_ranks.append(sims[1])
    counter = collections.Counter(ranks)
    return counter


def sim_label_file_names(model, input_file_name,
                         seg_dir, stoplists, config=False):
    file_names_json = os.listdir(seg_dir)
    fileload = open(seg_dir + input_file_name + '.json')
    data_all = json.load(fileload)
    fileload.close()

    if 'tok/fine' in data_all.keys():
        doc = [c for c in data_all.get('tok/fine') if c not in stoplists]
    else:
        doc = ['该文本没有内容']

    inferred_vector = model.infer_vector(doc)
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
    if not config:
        config = [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2),
                  ('LEAST', len(sims) - 1)]
    outcome = list()
    outcome.append(('INPUT', input_file_name))
    for label, index in config:
        outcome.append((label,
                        file_names_json[sims[index][0]].replace('.json', '')))
    return outcome


def new_sim_label_file_names(model, input_words_list, seg_dir, config=False):
    file_names_json = os.listdir(seg_dir)
    inferred_vector = model.infer_vector(input_words_list)
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
    if not config:
        config = [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2),
                  ('LEAST', len(sims) - 1)]
    outcome = list()
    for label, index in config:
        outcome.append((label,
                        file_names_json[sims[index][0]].replace('.json', '')))
    return outcome
