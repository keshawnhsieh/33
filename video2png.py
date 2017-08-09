#!/usr/bin/python

import cv2
import argparse
import os
import shutil


def path_merge(path1, path2):
    path = path1.split("/")
    if path[-1] == "":
        return "/".join(path) + "/" + path2
    else:
        return path1 + "/" + path2

parser = argparse.ArgumentParser(description="video2png converter")
parser.add_argument("video_file", help="video file")
parser.add_argument("output_dir", help="output png files directory")
args = parser.parse_args()

video_file = args.video_file
output_dir = args.output_dir
shutil.rmtree(output_dir)
os.mkdir(output_dir)

cap = cv2.VideoCapture(video_file)
while(cap.isOpened()):
    ret, frame = cap.read()
    if not frame is None:
        index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        subpath = str(index).zfill(5) + ".png"
        output_file = path_merge(output_dir, subpath)
        cv2.imwrite(output_file, frame)
    else:
        break
cap.release()
