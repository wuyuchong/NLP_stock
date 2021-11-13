#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import time
from src.auto import train_sim_lsi

while True:
    submit_dir = '/similarity/submit/lsi/'
    # mkdir
    try:
        os.makedirs(submit_dir)
    except FileExistsError:
        pass
    tasks = os.listdir(submit_dir)
    if tasks != []:
        print(tasks)
        task = tasks[0]
        f = open(submit_dir + task)
        arguments = json.load(f)
        f.close()
        os.remove(submit_dir + task)
        print('start --------> ', task)
        train_sim_lsi(**arguments)
        print('finish --------> ', task)
    time.sleep(3)
