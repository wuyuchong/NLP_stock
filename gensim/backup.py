#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import json
import gensim
import pandas as pd

from src.get_stoplists import get_stoplists

stoplists = get_stoplists()

# dictionary
dictionary = gensim.corpora.Dictionary([])
file_names = os.listdir('/segmentation/wemedia/content/BTI/')
for file_name in file_names:
    fileload = open('/segmentation/wemedia/content/BTI/' + file_name)
    data_all = json.load(fileload)
    fileload.close()
    dat = [c for c in data_all['tok/fine'] if c not in stoplists]
    dictionary.add_documents([dat])

# corpus: memory not friendly
corpus = list()
for file_name in file_names:
    fileload = open('/segmentation/wemedia/content/BTI/' + file_name)
    data_all = json.load(fileload)
    fileload.close()
    dat = [c for c in data_all['tok/fine'] if c not in stoplists]
    corpus.append(dictionary.doc2bow(dat))

# corpus: memeory friendly
#  class MyCorpus:
    #  def __iter__(self):
        #  file_names = os.listdir('/segmentation/wemedia/content/BTI/')
        #  for file_name in file_names:
            #  fileload = open('/segmentation/wemedia/content/BTI/' + file_name)
            #  data_all = json.load(fileload)
            #  fileload.close()
            #  dat = [c for c in data_all['tok/fine'] if c not in stoplists]
            #  yield dictionary.doc2bow(dat)
#  corpus = MyCorpus()

lsi = gensim.models.LsiModel(corpus, id2word=dictionary, num_topics=2)

fileload = open('/segmentation/wemedia/content/BTI/005ad181a39dd3fe62f0d60f6c713891.json')
data_all = json.load(fileload)
fileload.close()
doc = [c for c in data_all['tok/fine'] if c not in stoplists]
vec_bow = dictionary.doc2bow(doc)

vec_lsi = lsi[vec_bow]  # convert the query to LSI space
index = gensim.similarities.MatrixSimilarity(lsi[corpus])  # transform corpus to LSI space and index it
sims = index[vec_lsi]  # perform a similarity query against the corpus

sims = sorted(enumerate(sims), key=lambda item: -item[1])

file_name_txt = file_names[sims[1][0]].replace('.json', '.txt')

f = open('/data/wemedia/BTI/' + file_name_txt)
contents = f.read()
dictionary = json.loads(contents)
content = dictionary['content']
f.close()
print(content)

print('----------------')

f = open('/data/wemedia/BTI/005ad181a39dd3fe62f0d60f6c713891.txt')
contents = f.read()
dictionary = json.loads(contents)
content = dictionary['content']
f.close()
print(content)

#  for doc_position, doc_score in sims[0:3]:
    #  print(doc_score, file_names[doc_position])
