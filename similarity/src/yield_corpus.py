#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import gensim


def yield_corpus(seg_dir, stoplists, tokens_only=False):
    corpus = list()
    file_names = os.listdir(seg_dir)
    for i, file_name in enumerate(file_names):
        fileload = open(seg_dir + file_name)
        data_all = json.load(fileload)
        fileload.close()
        if 'tok/fine' in data_all.keys():
            tokens = [c for c in data_all['tok/fine'] if c not in stoplists]
            if tokens_only:
                yield tokens
            else:
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
