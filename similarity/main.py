#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Don't forget to install BLAS for fast computing
# apt install libatlas-base-dev
# install Intel MKL and switch to it:
# sudo update-alternatives --config libblas.so.3-x86_64-linux-gnu
# https://csantill.github.io/RPerformanceWBLAS/

# TODO: memory friendly >> class corpus >> yield

from src.auto import sim_doc2vec, sim_lsi

# print log or not
log = False

# set type and stock and file
content_type = 'wemedia'
#  stock_name = 'BTI'
#  input_file_name = '005ad181a39dd3fe62f0d60f6c713891'
stock_name = '贵州茅台'
input_file_name = 'ZWTw7mQBCy8qJGFB-iAo'

# lsi model
print('start lsi ---------->')

sim_lsi(content_type, stock_name, input_file_name, num_topics=2, log=log)

print('finish lsi ---------->',
      'OUTPUT THERE: /similarity/output/.../lsi/....')

# doc2vec model
print('start doc2vec ---------->')

sim_doc2vec(content_type, stock_name, input_file_name,
            vector_size=50, min_count=2, epochs=20, log=log)

print('finish doc2vec ---------->',
      'OUTPUT THERE: /similarity/output/.../doc2vec/....')
