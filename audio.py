#!/usr/bin/env python

from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np


def plot(wave, periods):
    plt.figure(1)
    plt.title('Signal Wave')
    plt.plot(wave)
    wave = np.abs(wave)
    if len(periods) > 0:
        for pair in periods:
            period = wave[pair[0]: pair[1]]
            maximum = np.max(period)
            plt.vlines(pair[0], -maximum, maximum, colors='r')
            plt.vlines(pair[1], -maximum, maximum, colors='r')
    plt.show()

def abnormal_check(data, threshold, duration, tolerance):
    if type(data) is np.ndarray:
        data = np.abs(data)
        periods = []
        start_point = -1
        drop_point = -1
        drop_time = 0
        for p in range(len(data)):
            if data[p] > threshold:
                if start_point == -1:
                    start_point = p
                else:
                    drop_point = -1
                    drop_time = 0
            else:
                if start_point != -1:
                    if drop_point != -1:
                        drop_time += 1
                    else:
                        drop_point = p
                    if drop_point - start_point > duration and drop_time > tolerance:
                        periods.append((start_point, drop_point))
                        start_point = -1
                        drop_time = 0
                        drop_point = -1
                else:
                    pass

    else:
        pass
    return periods

fs, data = wavfile.read('audio.wav')
peroids = abnormal_check(data, 10000, 1000, 2000)
plot(data, peroids)