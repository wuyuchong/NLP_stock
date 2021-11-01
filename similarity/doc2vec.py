#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Don't forget to install BLAS for fast computing
# apt install libatlas-base-dev
# install Intel MKL and switch to it:
# sudo update-alternatives --config libblas.so.3-x86_64-linux-gnu
# https://csantill.github.io/RPerformanceWBLAS/

import gensim
from src.get_stoplists import get_stoplists
from src.yield_corpus import yield_corpus
from src.check_model_health import check_model_health
from src.get_sim_label_file_names import get_sim_label_file_names
from src.get_document_content import get_document_content

#  import logging
#  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',

# set directory
seg_dir = '/segmentation/wemedia/content/贵州茅台/'
data_dir = '/data/wemedia/贵州茅台/'
#  seg_dir = '/segmentation/wemedia/content/BTI/'
#  data_dir = '/data/wemedia/BTI/'

# load stoplists
stoplists = get_stoplists()

# yield corpus
train_corpus = list(yield_corpus(seg_dir, stoplists))
#  test_corpus = list(yield_corpus(seg_dir, stoplists, tokens_only=True))

# build vocabulary
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=20)
model.build_vocab(train_corpus)
print(f"Word '上涨' appeared {model.wv.get_vecattr('上涨', 'count')} times in the training corpus.")
print(f"Word '下跌' appeared {model.wv.get_vecattr('下跌', 'count')} times in the training corpus.")

# training
model.train(train_corpus, total_examples=model.corpus_count,
            epochs=model.epochs)

# check model health
print(check_model_health(train_corpus, model))

# set input file
input_file_name = 'ZWTw7mQBCy8qJGFB-iAo'
#  input_file_name = '005ad181a39dd3fe62f0d60f6c713891'
# random TODO

# get similar file names with their labels
sim_label_file_names = get_sim_label_file_names(train_corpus, model,
                                                input_file_name, seg_dir)

# stdout
for label, file_name in sim_label_file_names:
    print(label)
    print(get_document_content(data_dir, file_name))
    print('-----------')
