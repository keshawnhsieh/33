#!/usr/bin/python

import rnn
import cv2
import numpy as np
import argparse
import os
import shutil

# global virables
area = []
x1, y1 = -1, -1
x2, y2 = -1, -1
get_x1y1 = False
get_x2y2 = False
clone = None

# used during testing
roi_ = None


def sort(array, col):
    if array == []:
        return array
    else:
        matrix = np.array(array)
        matrix = matrix[np.argsort(matrix[:, col])]
        return matrix.tolist()


class Area:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.contour_area_thresh = 200
        self.h_thresh = 15
        self.bounding_box = []
        self.roi = None
        self.predict = []

    def prompt_height(self):
        height = 10000
        for box in self.bounding_box:
            if height > box[3] - box[1]:
                height = box[3] - box[1]
        return height

    def prompt_area(self):
        areas = 10000
        for box in self.bounding_box:
            if areas > (box[2] - box[0]) * (box[3] - box[1]):
                areas = (box[2] - box[0]) * (box[3] - box[1])
        return areas

    def bug4(self, bounding_box):
        '''fix the double 4 connection problem'''
        for index, box in enumerate(bounding_box):
            if box[2] - box[0] > box[3] - box[1]:
                del bounding_box[index]
                bounding_box.append([box[0], box[1], (box[0] + box[2])/2, (box[3])])
                bounding_box.append([(box[0] + box[2])/2, box[1], box[2], box[3]])
        return bounding_box

    def filling(self, patch):
        h = patch.shape[0]
        w = patch.shape[1]
        if h > w:
            patch = np.hstack([255 * np.ones((h, (h-w)/2), dtype=np.uint8), patch, 255 * np.ones((h, (h-w)/2), dtype=np.uint8)])

        return patch

    def segment(self, image):
        # global roi_
        # clear the old image's info
        self.bounding_box = []
        self.predict = []
        self.roi = image[self.y1: self.y2, self.x1: self.x2]
        # cv2.imshow("roi", self.roi)
        # cv2.waitKey()
        im2, contours, hierarchy = cv2.findContours(self.roi,
                                                    cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > self.contour_area_thresh:
                # print cv2.contourArea(cnt)
                [x, y, w, h] = cv2.boundingRect(cnt)
                # cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.imshow("roi", roi)
                self.bounding_box.append([x, y, x+w, y+h])

        self.bounding_box = sort(self.bounding_box, 0)[1:]
        self.bounding_box = self.bug4(self.bounding_box)
        self.bounding_box = sort(self.bounding_box, 0)

    def rnn_forward(self, model_proto, model_caffemodel):
        for iter_box in self.bounding_box:
            roi_patch = self.roi[iter_box[1]: iter_box[3], iter_box[0]: iter_box[2]]
            roi_patch = self.filling(roi_patch)
            max_prob_index, prob = rnn.forward_pass(model_proto, model_caffemodel, 255 - roi_patch)
            self.predict.append(max_prob_index)

    def draw_result(self):
        global clone
        cv2.rectangle(clone, (self.x1, self.y1), (self.x2, self.y2), (0, 0, 255), 2)
        for idx, val in enumerate(self.bounding_box):
            digit = self.predict[idx]
            cv2.rectangle(clone, (self.x1 + val[0], self.y1 + val[1]), (self.x1 + val[2], self.y1 + val[3]),
                          (0, 255, 0), 1)
            cv2.putText(clone, str(digit), (self.x1 + val[2], self.y2 + 20), cv2.FONT_HERSHEY_COMPLEX,
                        0.6, (255, 0, 0), 1)
        # cv2.imshow("roi" + str(np.random.random()), self.roi)
        # cv2.imshow("img", clone)

        # cv2.waitKey()


def mouse_callback(event, x, y, flags, param):
    global x1, y1, x2, y2
    global get_x1y1, get_x2y2
    global clone
    global area
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1 = x, y
        get_x1y1 = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if get_x1y1:
            x2, y2 = x, y
            get_x2y2 = True
    elif event == cv2.EVENT_LBUTTONUP:
        get_x1y1 = False
        get_x2y2 = False
        cv2.rectangle(clone, (x1, y1), (x, y), (0, 0, 255), 2)
        area.append(Area(x1, y1, x2, y2))
        

def main():
    global clone, area, roi_
    parser = argparse.ArgumentParser(description="digital prediction")
    parser.add_argument("model_proto", help="model_proto")
    parser.add_argument("model_caffemodel", help="model_caffemodel")
    parser.add_argument("-P", "--image_file", help="image file")
    parser.add_argument("-V", "--video_file", help="video file")
    args = parser.parse_args()

    model_proto = args.model_proto
    model_caffemodel = args.model_caffemodel

    if args.image_file is None:
        if args.video_file is None:
            print "must have image or video input argument!"
        else:

            cap = cv2.VideoCapture(args.video_file)
            cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

            # Problem: the video saved on ubuntu can not be played?
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('t1.avi', fourcc, 20.0, (cap_w, cap_h))
            # Alternative solution: save all pics to disk and use command below to merge to a video
            # ffmpeg -i %5d.png -vcodec mpeg4 r1.avi
            out_dir = "res/"
            shutil.rmtree(out_dir)
            os.makedirs(out_dir)

            init = False
            first_frame = True
            cv2.namedWindow("img")
            cv2.setMouseCallback("img", mouse_callback)
            while cap.isOpened():
                ret, frame = cap.read()
                if frame is None:
                    break

                clone = frame.copy()
                while not init:
                    img0 = clone.copy()
                    if get_x1y1 and get_x2y2:
                        cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.imshow("img", img0)
                    key = cv2.waitKey(1)
                    if key == 13:
                        init = True
                    elif key == ord('q'):
                        exit(0)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # need to be more specific: blur, erosion
                blur = cv2.GaussianBlur(gray, (1, 1), 0)
                kernel = np.ones((1, 1), np.uint8)
                erosion = cv2.erode(255 - blur, kernel, iterations=1)
                thresh = cv2.adaptiveThreshold(255 - erosion, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, 11, 2)
                cv2.imshow('thresh', thresh)
                cv2.waitKey()

                e1 = cv2.getTickCount()
                for index, area_iter in enumerate(area):
                    area_iter.segment(thresh)
                    if first_frame and index == 0:
                        print 'Minimum height \t=\t' + str(area_iter.prompt_height())
                        print 'Minimum area \t=\t' + str(area_iter.prompt_area())
                        first_frame = False
                    area_iter.rnn_forward(model_proto, model_caffemodel)
                    area_iter.draw_result()
                e2 = cv2.getTickCount()
                time = (e2 - e1) / cv2.getTickFrequency() * 1000
                cv2.putText(clone, str(time) + "ms", (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.65, (0, 0, 255), 2)

                out.write(clone)
                cv2.imshow("img", clone)

                # out_file = out_dir + str(int(cap.get(cv2.CAP_PROP_POS_FRAMES))).zfill(5) + ".png"
                # cv2.imwrite(out_file, clone)

                key = cv2.waitKey(1)
                if key == ord("q"):
                    out.release()
                    cap.release()
                    cv2.destroyAllWindows()
                    exit(0)
            out.release()
            cap.release()
            cv2.destroyAllWindows()
    else:
        image_file = args.image_file
        img = cv2.imread(image_file)

        cv2.namedWindow("img")
        cv2.setMouseCallback("img", mouse_callback)

        clone = img.copy()
        while True:
            img0 = clone.copy()
            if get_x1y1 and get_x2y2:
                cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imshow("img", img0)
            key = cv2.waitKey(1)
            if key == 13:
                break
            elif key == ord('q'):
                exit(0)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (1, 1), 0)

        # now, let's play erode operation to break connections between digitals
        kernel = np.ones((1, 1), np.uint8)
        erosion = cv2.erode(255 - blur, kernel, iterations=1)

        # here we get three options to choose:
        # 1. cv2.threshold
        # 2. cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, ...)
        # 3. cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, ...)
        #       ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        thresh = cv2.adaptiveThreshold(255 - erosion, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        for area_iter in area:
            area_iter.segment(thresh)
            print 'average height = ' + str(area_iter.prompt_height())
            print 'average area = ' + str(area_iter.prompt_area())
            area_iter.rnn_forward(model_proto, model_caffemodel)
            area_iter.draw_result()
            # for bounding_box in area_iter.bounding_box:
            #     cv2.rectangle(img, (area_iter.x1 + bounding_box[0], area_iter.y1 + bounding_box[1]),
            #                   (area_iter.x1 + bounding_box[2], area_iter.y1 + bounding_box[3]), (0, 0, 255), 2)
            #     cv2.imshow("img", img)
            #     cv2.waitKey()
        cv2.imshow("img", clone)
        cv2.waitKey()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
