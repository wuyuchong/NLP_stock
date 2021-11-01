#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os


def get_sim_label_file_names(train_corpus, model, input_file_name,
                             seg_dir, config=False):
    file_names_json = os.listdir(seg_dir)
    doc_id = file_names_json.index(input_file_name + '.json')
    outcome = list()
    outcome.append(('INPUT', input_file_name))
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
    if not config:
        config = [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2),
                  ('LEAST', len(sims) - 1)]
    for label, index in config:
        outcome.append((label,
                        file_names_json[sims[index][0]].replace('.json', '')))
    return outcome
