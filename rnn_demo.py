#!/usr/bin/python

import numpy as np
import cv2
import cv2.dnn as dnn

def get_max_class(prob):
    return np.argmax(prob)

model_proto = "bvlc_googlenet.prototxt"
model_caffemodel = "bvlc_googlenet.caffemodel"
image_file = "space_shuttle.jpg"
label_file = "synset_words.txt"

net = dnn.readNetFromCaffe(model_proto, model_caffemodel)
if net.empty():
    print "load net: " + model_caffemodel + " failed."

img = cv2.imread(image_file)
if img is None:
    print "read image file " + image_file + " failed."

input_blob = dnn.blobFromImage(img, 1, (224, 224), (104, 117, 123))
print input_blob.shape
net.setInput(input_blob, "data")
prob = net.forward("prob")

max_prob_index = get_max_class(prob)
max_probability = prob[0, max_prob_index] * 100
labels = np.loadtxt(label_file, str, delimiter='\t')
max_label = labels[max_prob_index]
print "best class:\t" + " ".join(max_label.split()[1:])
print "probability:\t" + str(max_probability) + "%"
