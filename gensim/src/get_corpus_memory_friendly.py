#!/usr/bin/env python
# -*- coding: utf-8 -*-


#  corpus: memeory friendly
class MyCorpus:
    def __iter__(self):
        file_names = os.listdir('/segmentation/wemedia/content/BTI/')
        for file_name in file_names:
            fileload = open('/segmentation/wemedia/content/BTI/' + file_name)
            data_all = json.load(fileload)
            fileload.close()
            dat = [c for c in data_all['tok/fine'] if c not in stoplists]
            yield dictionary.doc2bow(dat)
corpus = MyCorpus()
