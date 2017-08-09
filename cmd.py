#!/usr/bin/env python
# -*- coding: utf-8 -*-

# usage:
# ./cmd.py -S demo2_cut.mp4 -P lenet.prototxt -C digital_iter_10000.caffemodel -A 200 -H 20 -M 1000 -D 50 -R 0.6 -O res.mp4
# choose area to recognize digital numbers using mouse, then press 'd'
# choose area to detect motions using mouse, then press 'm'
# after finishing choosing areas, press ENTER to run this demo

# parameter tuning tutorial:
#

from area import *
import cv2
import argparse

# global variables used in mouse callback function and main function
x1, y1 = -1, -1
x2, y2 = -1, -1
chosen_x2, chosen_y2 = -1, -1
button_down = False
button_up = False
moved = False


def mouse_callback(event, x, y, flags, param):
    global x1, y1, x2, y2, chosen_x2, chosen_y2
    global button_down, button_up, moved

    if event is cv2.EVENT_LBUTTONDOWN:
        x1, y1 = x, y
        button_down = True
        button_up = False
        moved = False
    elif event is cv2.EVENT_MOUSEMOVE:
        x2, y2 = x, y
        moved = True
    elif event is cv2.EVENT_LBUTTONUP:
        x2, y2 = x, y
        chosen_x2 = x2
        chosen_y2 = y2
        button_up = True


def main():
    # variables interacted between mouse callback function and main function
    global x1, y1, x2, y2, chosen_x2, chosen_y2
    global button_down, button_up, moved
    digital_areas = []
    curve_areas = []

    parser = argparse.ArgumentParser(description='33 project')
    # input video file name or video stream
    parser.add_argument('-S', '--stream', type=str, help='')
    # caffe net definition prototxt file
    parser.add_argument('-P', '--proto', type=str, help='')
    # caffemodel file
    parser.add_argument('-C', '--caffemodel', type=str, help='')
    # the minimum area to be considered containing a number
    parser.add_argument('-A', '--area_thresh', type=int, help='')
    # the minimum height to be considered containing a number, not used in this version
    parser.add_argument('-H', '--height_thresh', type=int, help='')
    # the minimum mse variation threshold
    parser.add_argument('-M', '--mse_thresh', type=int, help='')
    # the minimum variation threshold use to determined whether a pixel changes
    parser.add_argument('-D', '--diff_thresh', type=int, help='')
    # it will be considered as a global variation, such as illumination if the number of changed pixels is over ratio_threshold * width * height
    parser.add_argument('-R', '--ratio_thresh', type=float, help='')
    # specify the output video name if you want to save result as a video
    parser.add_argument('-O', '--output_video', type=str, help='')

    args = parser.parse_args()
    video = args.stream
    model_proto = args.proto
    model_caffemodel = args.caffemodel
    contour_area_thresh = args.area_thresh
    height_thresh = args.height_thresh
    mse_thresh = args.mse_thresh
    diff_thresh = args.diff_thresh
    ratio_thresh = args.ratio_thresh
    output = args.output_video

    cap = cv2.VideoCapture(video)
    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if output is not None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(output, fourcc, 20.0, (cap_w, cap_h))

    cv2.namedWindow('img')
    cv2.setMouseCallback('img', mouse_callback)

    first_frame = True

    # set debug flag True if you want to print debug info
    motion_debug = False
    digital_debug = False
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        frame2 = frame.copy()
        cv2.imshow('img', frame)
        while first_frame:
            if button_down and moved and not button_up:
                clone = frame.copy()
                cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.imshow('img', clone)
                moved = False
            key = cv2.waitKey(1)
            if key == 13:
                first_frame = False
            elif key == ord('d'):
                digital_areas.append(DigitalArea(x1, y1, chosen_x2, chosen_y2, contour_area_thresh, height_thresh, debug=digital_debug))
                cv2.rectangle(frame, (x1, y1), (chosen_x2, chosen_y2), (255, 0, 0), 2)
                cv2.imshow('img', frame)
                x1, y1 = -1, -1
                x2, y2 = -1, -1
                button_down = False
                button_up = False
                moved = False
            elif key == ord('m'):
                curve_areas.append(CurveArea(x1, y1, chosen_x2, chosen_y2, mse_thresh, diff_thresh, ratio_thresh, debug=motion_debug))
                cv2.rectangle(frame, (x1, y1), (chosen_x2, chosen_y2), (255, 0, 0), 2)
                cv2.imshow('img', frame)
                x1, y1 = -1, -1
                x2, y2 = -1, -1
                button_down = False
                button_up = False
                moved = False
            elif key == ord('q'):
                exit(0)

        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        e1 = cv2.getTickCount()
        for index, area_iter in enumerate(digital_areas):
            area_iter.segment(thresh)
            area_iter.rnn_forward(model_proto, model_caffemodel)

            # debug...
            # roi = area_iter.roi
            # for box in area_iter.bounding_box:
            #     cv2.rectangle(roi,
            #                   (box[0], box[1]),
            #                   (box[2], box[3]),
            #                   (0, 0, 255), 1)
            # cv2.imshow('main_roi', roi)
            # if cv2.waitKey() == ord('q'):
            #     exit(0)

            cv2.rectangle(frame,
                          (area_iter.x1, area_iter.y1),
                          (area_iter.x2, area_iter.y2),
                          (0, 0, 255), 2)
            for idx, box in enumerate(area_iter.bounding_box):
                cv2.rectangle(frame,
                              (area_iter.x1 + box[0], area_iter.y1 + box[1]),
                              (area_iter.x1 + box[2], area_iter.y1 + box[3]),
                              (0, 0, 255), 2)
                cv2.putText(frame,
                            str(area_iter.predict[idx]),
                            (area_iter.x1 + box[2] + 10, area_iter.y1 + box[3] + 30),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.6, (255, 0, 0), 2)

        for index, area_iter in enumerate(curve_areas):
            detected = area_iter.detect(gray)
            cv2.rectangle(frame,
                          (area_iter.x1, area_iter.y1),
                          (area_iter.x2, area_iter.y2),
                          (0, 0, 255), 2)
            if detected:
                cv2.putText(frame,
                            str('WARNING'),
                            (area_iter.x1 + 20, area_iter.y1 + 20),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.6, (0, 0, 255), 2)
        e2 = cv2.getTickCount()
        time = (e2 - e1) / cv2.getTickFrequency() * 1000
        cv2.putText(frame, str(time) + "ms", (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.65, (0, 0, 255), 2)

        if output is not None:
            writer.write(frame)

        cv2.imshow('img', frame)
        key = cv2.waitKey(33)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()
    if output is not None:
        writer.release()


if __name__ == '__main__':
    main()
