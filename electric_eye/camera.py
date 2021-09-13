#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MIT License

Copyright (c) 2021 Julian Schwanbeck (julian.schwanbeck@med.uni-goettingen.de)
https://github.com/schwanbeck/UMG_Antigen_Evaluator

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import logging
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

tp = ThreadPoolExecutor(10)


def threaded(fn):
    # from concurrent.futures import ThreadPoolExecutor
    # thread_pool = ThreadPoolExecutor(10)
    # thread_pool.shutdown()
    # https://stackoverflow.com/questions/19846332/python-threading-inside-a-class/19846691
    def wrapper(*args, **kwargs):
        return tp.submit(fn, *args, **kwargs)  # returns Future object
    return wrapper


class Camera:
    def __init__(self, skip_video_capture_port=0, log=None, dims=None):
        self.log = log or logging.getLogger(__name__)
        self.cam_num = skip_video_capture_port
        self.fps = 0
        self.cap = self.get_cap()
        self.frame = np.zeros((1, 1, 3))
        if self.cap is None:
            self.log.critical('Could not set camera')
        self.dims = self.set_dims(dims)
        self.log.debug('Camera set up')

    @threaded
    def get_frame(self):
        ret, last_frame = self.cap.read()
        # cv2.waitKey(1)  # doesn't work with threading here
        # >> ElectricEye.set_frame()
        if ret:
            self.frame = last_frame
        else:
            self.close_camera()
        return ret, self.frame

    def get_cap(self):
        cap = None
        dshow = cv2.CAP_DSHOW
        # if platform.system() == 'Windows' and platform.release() == '7':
        #     dshow = cv2.CAP_DSHOW
        for i in range(self.cam_num, 20):
            cap = cv2.VideoCapture(i, dshow)
            if cap is None or not cap.isOpened():
                self.log.debug(f"Could not open video source: {i}")
                cap = None
                continue
            self.log.debug(f"Using video source: {i}")
            self.cam_num = i
            break
        return cap

    def set_dims(self, dims):
        if self.cap is None:
            return dims
        if dims is None:
            dims = (
                self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )
        else:
            assert len(dims) == 2
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, dims[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, dims[1])
        self.log.debug(f"Dimensions: {dims[0]} x {dims[1]}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.log.debug(self.fps)
        return dims

    def close_camera(self):
        tp.shutdown()
        if self.cap is not None:
            self.cap.release()

    def __str__(self):
        return f"OpenCV Camera {self.cam_num}"


def global_img_centering(c_img, upper_limit=255):
    if c_img.size < 100:
        return None
    c_img = np.asarray(c_img).astype('float64')  # float32
    mean = c_img.mean()
    std = c_img.std()
    if not std:
        return None
    c_img = (c_img - mean) / std
    c_img = (c_img + 1) / 2.
    c_img = np.clip(c_img, 0., 1.)  # clip before resaling around 0, 1 -> (-1, 1)
    # mean, std = c_img.mean(), c_img.std()
    # print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
    # print('Min: %.3f, Max: %.3f' % (c_img.min(), c_img.max()))
    c_img *= upper_limit
    if upper_limit == 255:
        c_img = c_img.astype('uint8')
    else:
        c_img = c_img.astype('float32')
    return c_img
