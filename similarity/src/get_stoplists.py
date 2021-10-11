#!/usr/bin/env python
# -*- coding: utf-8 -*-

def get_stoplists():
    f = open('stopwords/baidu_stopwords.txt')
    lst = f.read().splitlines()
    f.close()
    f = open('stopwords/cn_stopwords.txt')
    lst += f.read().splitlines()
    f.close()
    f = open('stopwords/hit_stopwords.txt')
    lst += f.read().splitlines()
    f.close()
    f = open('stopwords/scu_stopwords.txt')
    lst += f.read().splitlines()
    f.close()
    return lst

