#!/usr/bin/python

import os
import numpy as np
import argparse
import cv2


def path_merge(path1, path2):
    path = path1.split("/")
    if path[-1] == "":
        return "/".join(path) + "/" + path2
    else:
        return path1 + "/" + path2

parser = argparse.ArgumentParser(description="convert 3-channel images to 1-channel")
parser.add_argument("input_dir", help="directory storing images")
parser.add_argument("output_dir", help="directory to store 1-channel images")
args = parser.parse_args()

file_dir = args.input_dir
output_dir = args.output_dir
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

files = os.listdir(file_dir)
for file in files:
    if not os.path.isdir(file):
        img_file = path_merge(file_dir, file)
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

        output_file = path_merge(output_dir, file)
        cv2.imwrite(output_file, img)

