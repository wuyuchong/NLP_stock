#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hanlp
import json
import ast
import sys
import os

#  HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)

size = os.path.getsize('/data/news/豪悦护理/f5041bf0d22cbd0d3d7f0951aa15a18b.txt')
print(size > 100000)
sys.exit()

file = open('/data/news/豪悦护理/f5041bf0d22cbd0d3d7f0951aa15a18b.txt')
contents = file.read()
#  dictionary = ast.literal_eval(contents)
dictionary = json.loads(contents)
title = dictionary['title']
content = dictionary['content']
file.close()

data = HanLP(content, tasks=['tok/fine', 'tok/coarse'])
print(data)

