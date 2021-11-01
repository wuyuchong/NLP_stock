#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gensim
from src.get_stoplists import get_stoplists
from src.get_dictionary import get_dictionary
from src.get_corpus import get_corpus
from src.get_sim_file_name import get_sim_file_name
from src.print_document import print_document

# set directory
seg_dir = '/segmentation/wemedia/content/贵州茅台/'
data_dir = '/data/wemedia/贵州茅台/'
#  seg_dir = '/segmentation/wemedia/content/BTI/'
#  data_dir = '/data/wemedia/BTI/'

# load stoplists
stoplists = get_stoplists()

# init dictionary
dictionary = get_dictionary(seg_dir, stoplists)

# build corpus
corpus = get_corpus(seg_dir, stoplists, dictionary)

# construct Lsimodel
lsi = gensim.models.LsiModel(corpus, id2word=dictionary, num_topics=2)

# set test file
input_file_name = 'ZWTw7mQBCy8qJGFB-iAo'
#  input_file_name = '005ad181a39dd3fe62f0d60f6c713891'

# get similar file name
sim_file_name = get_sim_file_name(1, seg_dir, stoplists, dictionary,
                                  corpus, lsi, input_file_name)

# print content of test file
print_document(data_dir, input_file_name)

# print content of similar file
print_document(data_dir, sim_file_name)
