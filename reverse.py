#!/usr/bin/python

import os
import argparse
import cv2
import numpy as np


def path_merge(path1, path2):
    path = path1.split("/")
    if path[-1] == "":
        return "/".join(path) + "/" + path2
    else:
        return path1 + "/" + path2

parser = argparse.ArgumentParser(description="reverse image color")
parser.add_argument("file_dir", help="directory storing images")
parser.add_argument("output_dir", help="directory to store color reversed images")
args = parser.parse_args()

file_dir = args.file_dir
output_dir = args.output_dir
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

files = os.listdir(file_dir)
for file in files:
    if not os.path.isdir(file):
        img_file = path_merge(file_dir, file)
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        if not img is None:
            img = 255 - img
            output_file = path_merge(output_dir, file)
            cv2.imwrite(output_file, img)
        else:
            print "load img " + img_file + " failed."
