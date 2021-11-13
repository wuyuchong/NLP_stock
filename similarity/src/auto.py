#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import gensim
from time import gmtime, strftime
import src.base
import src.doc2vec
import src.lsi


def get_stock_list(content_type):
    dir_master = '/data/' + content_type + '/'
    return [f for f in os.listdir(dir_master) if not os.path.isfile(os.path.join(dir_master, f))]


def get_model_type(content_type, stock_name):
    dir_master = '/similarity/model/' + content_type + '/content/' + stock_name + '/'
    return [f for f in os.listdir(dir_master) if not os.path.isfile(os.path.join(dir_master, f))]


def get_model_list(content_type, stock_name, model_type):
    dir_master = '/similarity/model/' + content_type + '/content/' + stock_name + '/' + model_type + '/'
    return [f for f in os.listdir(dir_master) if not os.path.isfile(os.path.join(dir_master, f))]


def submit_train_lsi(content_type, stock_name, num_topics):
    submit_dir = '/similarity/submit/lsi/'
    # mkdir
    try:
        os.makedirs(submit_dir)
    except FileExistsError:
        pass
    arguments = ['content_type', 'stock_name', 'num_topics']
    values = [content_type, stock_name, num_topics]
    write_dir = submit_dir + strftime("%Y-%m-%d|%H:%M:%S", gmtime())
    with open(write_dir, 'w+', encoding='utf-8') as f:
        json.dump(dict(zip(arguments, values)), f, ensure_ascii=False, indent=4)


def submit_train_doc2vec(content_type, stock_name, vector_size, min_count, epochs):
    submit_dir = '/similarity/submit/doc2vec/'
    # mkdir
    try:
        os.makedirs(submit_dir)
    except FileExistsError:
        pass
    arguments = ['content_type', 'stock_name', 'vector_size', 'min_count', 'epochs']
    values = [content_type, stock_name, vector_size, min_count, epochs]
    write_dir = submit_dir + strftime("%Y-%m-%d|%H:%M:%S", gmtime())
    with open(write_dir, 'w+', encoding='utf-8') as f:
        json.dump(dict(zip(arguments, values)), f, ensure_ascii=False, indent=4)


def train_sim_doc2vec(content_type, stock_name, vector_size=50,
                      min_count=2, epochs=20, log=False):

    if log:
        import logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                            level=logging.INFO)

    seg_dir = '/segmentation/' + content_type + '/content/' + stock_name + '/'
    save_dir = '/similarity/model/' + content_type + '/content/' + stock_name +  \
        '/doc2vec/vec_' + str(vector_size) + '_min_' + str(min_count) + '_epo_' + str(epochs) + '/'

    # load stoplists
    stoplists = src.base.get_stoplists()

    # mkdir
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        pass

    # yield corpus
    train_corpus = list(src.doc2vec.yield_corpus(seg_dir, stoplists))

    # build vocabulary
    model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size,
                                          min_count=min_count,
                                          epochs=epochs)
    model.build_vocab(train_corpus)
    #  print(f"Word '上涨' appeared {model.wv.get_vecattr('上涨', 'count')} times in the training corpus.")
    #  print(f"Word '下跌' appeared {model.wv.get_vecattr('下跌', 'count')} times in the training corpus.")

    # training
    model.train(train_corpus, total_examples=model.corpus_count,
                epochs=model.epochs)
    model.save(save_dir + '/default.d2v')

    # check model health
    #  print(src.doc2vec.check_model_health(train_corpus, model))


def query_sim_doc2vec(content_type, stock_name, input_content, vector_size=50,
                      min_count=2, epochs=20, log=False, model_token=''):
    if log:
        import logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                            level=logging.INFO)

    # load stoplists
    stoplists = src.base.get_stoplists()

    seg_dir = '/segmentation/' + content_type + '/content/' + stock_name + '/'

    save_dir = '/similarity/model/' + content_type + '/content/' + stock_name +  \
        '/doc2vec/vec_' + str(vector_size) + '_min_' + str(min_count) + '_epo_' + str(epochs) + '/'
    if model_token != '':
        save_dir = '/similarity/model/' + content_type + '/content/' + stock_name +  \
            '/doc2vec/' + model_token + ''

    # load model
    model = gensim.models.doc2vec.Doc2Vec.load(save_dir + '/default.d2v')

    # segmentation process
    input_words_list = src.base.segment_process(input_content, stoplists)

    # get similar file names with their labels
    sim_label_file_names = src.doc2vec.query_sim_label_file_names(model, input_words_list, seg_dir)

    # write json file
    src.base.query_write_json(content_type, stock_name, 'doc2vec',
                            sim_label_file_names)

    return src.base.outcome(content_type, stock_name, sim_label_file_names)


def train_sim_lsi(content_type, stock_name, num_topics=2, log=False):
    if log:
        import logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                            level=logging.INFO)

    seg_dir = '/segmentation/' + content_type + '/content/' + stock_name + '/'
    save_dir = '/similarity/model/' + content_type + '/content/' + stock_name +  \
        '/lsi/topic_' + str(num_topics) + '/'

    # load stoplists
    stoplists = src.base.get_stoplists()

    # mkdir
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        pass

    # init dictionary
    dictionary = src.lsi.get_dictionary(seg_dir, stoplists)
    dictionary.save(save_dir + '/default.dict')

    # build corpus
    corpus = list(src.lsi.yield_corpus(seg_dir, stoplists, dictionary))
    gensim.corpora.MmCorpus.serialize(save_dir + '/default.mm', corpus)

    # construct Lsimodel
    lsi = gensim.models.LsiModel(corpus, id2word=dictionary,
                                 num_topics=num_topics)
    lsi.save(save_dir + '/default.lsi')



def query_sim_lsi(content_type, stock_name, input_content,
                num_topics=2, log=False, model_token=''):
    if log:
        import logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                            level=logging.INFO)

    seg_dir = '/segmentation/' + content_type + '/content/' + stock_name + '/'

    save_dir = '/similarity/model/' + content_type + '/content/' + stock_name +  \
        '/lsi/topic_' + str(num_topics) + '/'
    if model_token != '':
        save_dir = '/similarity/model/' + content_type + '/content/' + stock_name +  \
            '/lsi/' + model_token + '/'

    # load stoplists
    stoplists = src.base.get_stoplists()

    # segmentation process
    input_words_list = src.base.segment_process(input_content, stoplists)

    # load model
    dictionary = gensim.corpora.Dictionary.load(save_dir + '/default.dict')
    corpus = gensim.corpora.MmCorpus(save_dir + '/default.mm')
    lsi = gensim.models.LsiModel.load(save_dir + '/default.lsi')

    # get similar file name
    sim_label_file_names = src.lsi.query_sim_label_file_names(seg_dir, dictionary, corpus, lsi, input_words_list)

    # write json file
    src.base.query_write_json(content_type, stock_name, 'lsi',
                              sim_label_file_names)

    return src.base.outcome(content_type, stock_name, sim_label_file_names)
