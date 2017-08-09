#!/usr/bin/python

import os
import numpy as np
import random

per = 0.9
abs_path = os.path.abspath(__file__)
abs_path = "/".join(abs_path.split("/")[:-1])
abs_path += "/digitals/origin/"
# print abs_path

train_list = abs_path + "train.txt"
test_list = abs_path + "test.txt"
if os.path.isfile(train_list):
    os.remove(train_list)
if os.path.isfile(test_list):
    os.remove(test_list)
train_file = open(train_list, 'w')
test_file = open(test_list, 'w')

for num in range(10):
    label = num
    file_dir = abs_path + str(num)
    files = os.listdir(file_dir)
    for index, val in enumerate(files):
        if os.path.isfile(val):
            del files[index]
    print "0:\t\t" + str(len(files)) + "files."
    print "train set:\n"
    sample = random.sample(range(len(files)), int(len(files)*per))
    for index in sample:
        sub_path = str(num) + "/" + files[index]
        print sub_path
        train_file.write(sub_path + " " + str(label) + "\n")
    new_files = []
    for i in range(len(files)):
        if i not in sample:
            new_files.append(files[i])
    files = new_files
    print "test set:\n"
    for iter_file in files:
        sub_path = str(num) + "/" + iter_file
        print sub_path
        test_file.write(sub_path + " " + str(num) + "\n")
train_file.close()
test_file.close()
