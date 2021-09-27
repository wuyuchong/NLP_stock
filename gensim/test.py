#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import json
import gensim

# init
dictionary = gensim.corpora.Dictionary([])

file_names = os.listdir('/segmentation/wemedia/content/BTI/')
for file_name in file_names:
    fileload = open('/segmentation/wemedia/content/BTI/' + file_name)
    data_all = json.load(fileload)
    fileload.close()
    dictionary.add_documents([data_all['tok/fine']])

print(dictionary)

class MyCorpus:
    def __iter__(self):
        file_names = os.listdir('/segmentation/wemedia/content/BTI/')
        for file_name in file_names:
            fileload = open('/segmentation/wemedia/content/BTI/' + file_name)
            data_all = json.load(fileload)
            fileload.close()
            yield dictionary.doc2bow(data_all['tok/fine'])

corpus = MyCorpus()

for vector in corpus:  # load one vector into memory at a time
    print(vector)
