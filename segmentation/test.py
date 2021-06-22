#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
with open('/data/news/格力电器/CNY1MmUBCy8qJGFBg9hn.txt') as f:
    data = f.read()
js = json.loads(data)
print(js['web'])

