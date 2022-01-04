#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Don't forget to install BLAS for fast computing
# apt install libatlas-base-dev
# install Intel MKL and switch to it:
# sudo update-alternatives --config libblas.so.3-x86_64-linux-gnu
# https://csantill.github.io/RPerformanceWBLAS/

# TODO: memory friendly >> class corpus >> yield

from src.auto import train_sim_lsi, query_sim_lsi
from src.auto import train_sim_doc2vec, query_sim_doc2vec

# print log or not
log = False

# set type and stock and file
content_type = 'wemedia'
input_content = '股市不是很重要的几个公式'
#  stock_name = 'BTI'
#  input_file_name = '005ad181a39dd3fe62f0d60f6c713891'
stock_name = '贵州茅台'
input_file_name = 'ZWTw7mQBCy8qJGFB-iAo'
#  stock_name = '大消费'
#  input_file_name = 'c07a4986be45dcd6a4e6b9fa00a947dc'

# lsi model
print('start lsi ---------->')
train_sim_lsi(content_type, stock_name, num_topics=2, log=log)
print('train: finish lsi ---------->',
      'OUTPUT THERE: /similarity/model/.../lsi/....')
query_sim_lsi(content_type, stock_name, input_content, num_topics=2, log=log)
print('query: finish lsi ---------->',
      'OUTPUT THERE: /similarity/output/.../lsi/....')

# doc2vec model
print('start doc2vec ---------->')
train_sim_doc2vec(content_type, stock_name, vector_size=50, min_count=2, epochs=20, log=log)
print('train: finish doc2vec ---------->',
      'OUTPUT THERE: /similarity/model/.../doc2vec/....')
query_sim_doc2vec(content_type, stock_name, input_content, vector_size=50, min_count=2, epochs=20, log=log)
print('query: finish doc2vec ---------->',
      'OUTPUT THERE: /similarity/output/.../doc2vec/....')
