# Description
**This project aims to automatically recognize the digital numbers and detect the variation in areas given on the screen. All codes developed and tested pass on Ubuntu 14.04+ with OpenCV version 3.3-rc including extra modules and [caffe](https://github.com/BVLC/caffe).**

The recognition function is achieved by using convolutional neural network and the CNN net used in this program is lenet which is a famous convolutional neural network used in handwritten number recognition task. 

I fine tuning the net weights pre-trained on mnist dataset using digital numbers download from [here](https://github.com/liuruoze/EasyPR) under the deep learning framework [caffe](https://github.com/BVLC/caffe). Then I use opencv library to call this trained caffe model to predict digital numbers segmented from the screen.

The detection function is achieved by simply subtracting two adjacent pictures, and set a specific threshold to control sensitivity. 

# Usage
**The first three steps shown as below is not necessary if the digital_iter_10000.caffemodel has been generated and saved in the directory.**

In order to speed up the github sync, I config the dericotry *mnist_png, digitals* not to be upload to github. And the mnist and digitals dataset have been save as *mnist_png.tar.gz* and *digitals.tar.gz*. Remember to decompression them if necessary.
## Generate lmdb file for training and testing
We need to generate the subpath/label description file used in **convert_imageset** tool.
```
# use generate_image_list.py to generate subpath/label description file, change the path in codes if necessary
./generate_image_list.py 
```

Then we use **convert_iamgeset** tool provided by caffe to create lmdb training and testing data. **DO NOT forget the argument -gray!**
```
# cd to caffe root directory
./build/tools/convert_imageset -shuffle -gray -resize_height 28 -resize_width 28 /path/to/root/of/samples/ /path/to/list/file/ /dir/to/store/lmdb/
```

## Fine tuning model
Fine tuning weight using caffe.
```
# cd to caffe root directory
./build/tools/caffe train -sovler /abspath/to/lenet_solver.prototxt -weights /abspath/to/lenet_iter_5000.caffemodel
```

## Predict digital numbers and detect motion
Now, we have a trained caffemodel named digital_iter_10000.caffemodel. Use it to predict digital numbers with help of OpenCV. And detect motions in the given area with a naive method.

**Be careful with layer description file \*.prototxt, the prototxt file used below is a little bit different from that used in training stage. Remember to remove the train data input layer and test data input layer in the beginning and add some lines writen below. What's more, change the last layer to prob layer.**
```
input: "data"
input_dim: 64
input_dim: 1
input_dim: 28
input_dim: 28
```

Then use cmd.py to predict digital numbers. The parameters used are illustrated as comment line in python files.
```
./cmd.py -S demo2_cut.mp4 -P lenet.prototxt -C digital_iter_10000.caffemodel -A 200 -H 20 -M 1000 -D 50 -R 0.6 -O res.mp4
```

## audio detection
The audio.py script is used to detect audio abnormality. The code are designed roughly.

## Something other useful tools
Use `video2png.py` to convert video to png sequences.

Use `reverse.py` to reverse the color of foreground and background of gray images.

`rnn_demo.py` is the python version of official [demo](http://docs.opencv.org/trunk/d5/de7/tutorial_dnn_googlenet.html) to demonstrate how to call caffemodel in OpenCV.




