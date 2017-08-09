#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
测试demo:./motion.py t2.webm 15 100 0.6
第一个参数是视频文件名称
第二个参数是mse改变阈值
第三个参数是对位点突变阈值,相邻两帧对应点之间的差值超过该值则认为该位置发生了突变
第四个参数是突变点比率阈值,突变点数量超过该比例时认为发生的是全局突变,那么这种突变很有可能是由光照改变引起的
参数调优方法:
1. 只取消mse函数中的print语句注释,通过打印出来的mse值配合观察图片的真实情况选定合适的mse阈值
2. 随便设定第三个参数,只取消mutation_num函数中的print语句注释,
通过打印出来的diff ratio值配合图片出现的光照骤变选择合适的ratio阈值,
如果光照发生和未发生该ratio值没什么大的变化,则适当调整一下第三个参数
'''

import numpy as np
import cv2
import argparse
import os
import shutil


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # equ = cv2.equalizeHist(blur)
    back = gray
    return back


def mse(image1, image2):
    err = np.sum((image1.astype('float32') - image2.astype('float32')) ** 2)
    err /= float(image1.shape[0] * image2.shape[1])
    # print 'mse err = ' + str(err)
    return err


def mutation_num(image1, image2, threshold):
    diff = np.abs(image1 - image2)
    num = np.sum((diff > threshold).astype(int))
    # print 'diff ratio = ' + str(np.float32(num) / (image1.shape[0] * image1.shape[1]))
    return num


def detect(last_img, curr_img, mse_threshold, diff_threshold, ratio_threshold):
    if last_img.shape != curr_img.shape:
        return Flase
    height = curr_img.shape[0]
    width = curr_img.shape[1]

    err = mse(last_img, curr_img)
    if err > np.float64(mse_threshold) \
            and mutation_num(last_img, curr_img, diff_threshold) < ratio_threshold * height * width:
        return True

def main():
    parser = argparse.ArgumentParser(description='motion detection')
    parser.add_argument('video_file', help='video file name')
    # 为相邻帧之间的mse设定的阈值
    parser.add_argument('mse_threshold', type=int, help='mse threshold')
    # 超过该阈值的点认为发生了突变
    parser.add_argument('diff_threshold', type=int, help='different threshold')
    # 发生突变点的数量未超过该阈值才认为发生的是局部突变,而非光照引起的全局突变,该阈值是一个0-1之间的比例
    parser.add_argument('ratio_threshold', type=float, help='number threshold')
    args = parser.parse_args()

    mse_threshold = args.mse_threshold
    diff_threshold = args.diff_threshold
    ratio_threshold = args.ratio_threshold

    cap = cv2.VideoCapture(args.video_file)
    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('t2.avi', fourcc, 20.0, (cap_w, cap_h))

    out_dir = 'res/'
    shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    last_frame = None
    if cap.isOpened():
        _, last_frame = cap.read()
        last_frame = preprocess(last_frame)

    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        current_frame = frame
        current_frame = preprocess(current_frame)
        err = mse(current_frame, last_frame)
        if err > np.float64(mse_threshold) \
                and mutation_num(current_frame, last_frame, diff_threshold) < ratio_threshold * cap_h * cap_w:
            cv2.putText(frame, 'Motion detected!', (30, 30), cv2.FONT_HERSHEY_COMPLEX, 0.65, (0, 0, 255), 2)
        last_frame = current_frame
        cv2.imwrite(out_dir + str(int(cap.get(cv2.CAP_PROP_POS_FRAMES))).zfill(5) + '.png', frame)
        out.write(frame)

        cv2.imshow('preprocessed', current_frame)
        cv2.imshow("frame", frame)
        c = cv2.waitKey(1)
        if c == ord('q'):
            break

if __name__ == '__main__':
    main()

