#!/usr/bin/python

import numpy as np
import cv2
import cv2.dnn as dnn
import argparse

def get_max_class(prob):
    return np.argmax(prob)

def forward_pass(model_proto, model_caffemodel, image):
    net = dnn.readNetFromCaffe(model_proto, model_caffemodel)
    if net.empty():
        print "load net: " + model_caffemodel + " failed."
    input_blob = dnn.blobFromImage(image, 1, (28, 28))
    net.setInput(input_blob, "data")
    prob = net.forward("prob")
    max_prob_index = get_max_class(prob)
    probability = prob[0, max_prob_index]
    return max_prob_index, probability

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="rnn forward")
    parser.add_argument("image", help="image to process")
    args = parser.parse_args()
    image_file = args.image

    model_proto = "lenet.prototxt"
    model_caffemodel = "lenet_iter_5000.caffemodel"
    # image_file = "/home/xieqixin/cv/mnist_png/training/0/1209.png"
    image = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
    index, probability = forward_pass(model_proto, model_caffemodel, image)
    print "best class:\t" + str(index)
    print "probability:\t" + str(probability * 100) + "%"
    
