#!/usr/bin/env python

import cv2
import rnn
import numpy as np


def sort(array, col):
    if array is []:
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
        self.width = x2 - x1
        self.height = y2 - y1


class DigitalArea(Area):
    def __init__(self, x1, y1, x2, y2, contour_area_thresh, h_thresh, debug=False):
        Area.__init__(self, x1, y1, x2, y2)
        self.contour_area_thresh = contour_area_thresh
        self.h_thresh = h_thresh
        self.bounding_box = []
        self.roi = None
        self.predict = []
        self.debug = debug

    def promote_height(self):
        height = 10000
        for box in self.bounding_box:
            if height > box[3] - box[1]:
                height = box[3] - box[1]
        return height

    def promote_area(self):
        areas = 10000
        for box in self.bounding_box:
            if areas > (box[2] - box[0]) * (box[3] - box[1]):
                areas = (box[2] - box[0]) * (box[3] - box[1])
        return areas

    def filling(self, patch):
        h = patch.shape[0]
        w = patch.shape[1]
        if h > w:
            patch = np.hstack([255 * np.ones((h, (h-w)/2), dtype=np.uint8),
                               patch,
                               255 * np.ones((h, (h-w)/2), dtype=np.uint8)])
        return patch

    def segment(self, image):
        self.bounding_box = []
        self.predict = []
        self.roi = image[self.y1: self.y2, self.x1: self.x2]

        # debug...
        # cv2.imshow('roi', self.roi)
        # cv2.waitKey()

        im2, contours, hierarchy = cv2.findContours(self.roi,
                                                    cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > self.contour_area_thresh:
                [x, y, w, h] = cv2.boundingRect(cnt)
                self.bounding_box.append([x, y, x+w, y+h])
        self.bounding_box = sort(self.bounding_box, 0)[1:]
        if self.debug:
            print '----- DIGITAL RECOGNITION STATISTICS -----'
            print '%25s %d' % ('AREA WIDTH:', self.width)
            print '%25s %d' % ('AREA HEIGHT:', self.height)
            print '%25s %d' % ('AREA SIZE:', self.width * self.height)
            print '%25s %d' % ('MINI AREA:', self.promote_area())
            print '%25s %d' % ('MINI HEIGHT:', self.promote_height())
            print '\n'

    def rnn_forward(self, model_proto, model_caffemodel):
        for iter_box in self.bounding_box:
            roi_patch = self.roi[iter_box[1]: iter_box[3], iter_box[0]: iter_box[2]]
            roi_patch = self.filling(roi_patch)
            max_prob_index, prob = rnn.forward_pass(model_proto, model_caffemodel, 255 - roi_patch)
            self.predict.append(max_prob_index)


class CurveArea(Area):
    def __init__(self, x1, y1, x2, y2, mse_thresh, diff_thresh, ratio_thresh, debug=False):
        Area.__init__(self, x1, y1, x2, y2)
        self.last_roi = None
        self.mse_thresh = mse_thresh
        self.diff_thresh = diff_thresh
        self.ratio_thresh = ratio_thresh
        self.debug = debug

    def mse(self, image1, image2):
        err = np.sum((image1.astype('float32') - image2.astype('float32')) ** 2)
        err /= float(image1.shape[0] * image2.shape[1])
        # print 'mse err = ' + str(err)
        return err

    def mutation_num(self, image1, image2, threshold):
        diff = np.abs(image1 - image2)
        num = np.sum((diff > threshold).astype(int))
        # print 'diff ratio = ' + str(np.float32(num) / (self.width * self.height)
        return num

    def detect(self, img):
        curr_roi = img[self.y1: self.y2, self.x1: self.x2]
        if self.last_roi is None:
            self.last_roi = curr_roi
            return False

        err = self.mse(curr_roi, self.last_roi)
        mutation = self.mutation_num(curr_roi, self.last_roi, self.diff_thresh)

        if err > np.float64(self.mse_thresh) and mutation < self.ratio_thresh * self.height * self.width:
            self.last_roi = curr_roi
            if self.debug:
                print '----- MOTION DETECTION STATISTICS -----'
                print '%20s %f' % ('MSE:', err)
                print '%20s %f' % ('MUTATION RATIO:', mutation.astype(float) / self.height / self.width)
                print '\n'
            return True

