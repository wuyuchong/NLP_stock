#!/usr/bin/env python
# -*- coding: utf-8 -*-


def sim_doc2vec(content_type, stock_name, input_file_name,
                vector_size=50, min_count=2, epochs=20, log=False):

    if log:
        import logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                            level=logging.INFO)

    seg_dir = '/segmentation/' + content_type + '/content/' + stock_name + '/'

    # load stoplists
    stoplists = src.base.get_stoplists()

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

    # check model health
    #  print(src.doc2vec.check_model_health(train_corpus, model))

    # get similar file names with their labels
    sim_label_file_names = src.doc2vec.sim_label_file_names(train_corpus,
                                                            model,
                                                            input_file_name,
                                                            seg_dir, stoplists)

    # write json file
    src.base.write_json(content_type, stock_name, 'doc2vec',
                        input_file_name, sim_label_file_names)


def use_sim_doc2vec(content_type, stock_name, input_file_name, vector_size=50,
                    min_count=2, epochs=20, log=False):
    if log:
        import logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                            level=logging.INFO)

    # load stoplists
    stoplists = src.base.get_stoplists()

    seg_dir = '/segmentation/' + content_type + '/content/' + stock_name + '/'
    save_dir = '/similarity/model/' + content_type + '/content/' + stock_name +  \
        '/vec_' + vector_size + '_min_' + min_count + '_epo_' + epochs + '/'

    model = gensim.models.doc2vec.Doc2Vec.load(save_dir + '/default.d2v')
    # get similar file names with their labels
    sim_label_file_names = src.doc2vec.sim_label_file_names(model,
                                                            input_file_name,
                                                            seg_dir, stoplists)

    # write json file
    src.base.write_json(content_type, stock_name, 'doc2vec',
                        input_file_name, sim_label_file_names)


def sim_lsi(content_type, stock_name, input_file_name,
            num_topics=2, log=False):
    if log:
        import logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                            level=logging.INFO)

    seg_dir = '/segmentation/' + content_type + '/content/' + stock_name + '/'

    # load stoplists
    stoplists = src.base.get_stoplists()

    # init dictionary
    dictionary = src.lsi.get_dictionary(seg_dir, stoplists)

    # build corpus
    corpus = list(src.lsi.yield_corpus(seg_dir, stoplists, dictionary))

    # construct Lsimodel
    lsi = gensim.models.LsiModel(corpus, id2word=dictionary,
                                 num_topics=num_topics)

    # get similar file name
    sim_label_file_names = src.lsi.sim_label_file_names(seg_dir, stoplists,
                                                        dictionary, corpus,
                                                        lsi, input_file_name)

    # write json file
    src.base.write_json(content_type, stock_name, 'lsi',
                        input_file_name, sim_label_file_names)


def use_sim_lsi(content_type, stock_name, input_file_name,
                num_topics=2, log=False):
    if log:
        import logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                            level=logging.INFO)

    seg_dir = '/segmentation/' + content_type + '/content/' + stock_name + '/'
    save_dir = '/similarity/model/' + content_type + '/content/' + stock_name +  \
        '/topic_' + vector_size + '/'

    # load stoplists
    stoplists = src.base.get_stoplists()

    # load model
    dictionary = gensim.corpora.Dictionary.load(save_dir + '/default.dict')
    corpus = gensim.corpora.MmCorpus(save_dir + '/default.mm')
    lsi = gensim.models.LsiModel.load(save_dir + '/default.lsi')

    # get similar file name
    sim_label_file_names = src.lsi.sim_label_file_names(seg_dir, stoplists,
                                                        dictionary, corpus,
                                                        lsi, input_file_name)

    # write json file
    src.base.write_json(content_type, stock_name, 'lsi',
                        input_file_name, sim_label_file_names)

