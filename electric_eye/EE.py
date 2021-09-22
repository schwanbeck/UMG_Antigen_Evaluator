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

import copy
import functools
import logging
import os
import pickle
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from shutil import copy2
from time import sleep

import cv2
import numpy as np
from pynput import keyboard
from pyzbar import pyzbar
from scipy.signal import find_peaks

from electric_eye.camera import Camera, global_img_centering
from electric_eye.constants import *
from electric_eye.graphs import LineManager, put_text, put_text_list, put_rectangle, convert_to_colormap
from electric_eye.helper_func import (
    makedir, npaverage_weights, safe_str, noneless_result,
    check_dict, get_strftime
)
from electric_eye.setup import Config

tp = ThreadPoolExecutor(10)


def threaded(fn):
    # from concurrent.futures import ThreadPoolExecutor
    # thread_pool = ThreadPoolExecutor(10)
    # thread_pool.shutdown()
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return tp.submit(fn, *args, **kwargs)
    return wrapper


class ElectricEye:
    def __init__(self, test_abbreviation=None, test=False, base_path='./', ext_path='./',
                 electric_evaluation=True, config_path=None):
        self.log = logging.getLogger(__name__)
        self.test = test
        self.running = True
        self.local_path = makedir(os.path.join(base_path, datetime.now().strftime('%Y%m%d')))
        self.ext_path = makedir(ext_path)
        self.backup_path = os.path.join(self.local_path, f"{datetime.now().strftime('%Y%m%d')}_backup.pickle")
        assert self.local_path != self.ext_path, 'Local and external paths are identical.'
        self.conf_parser = Config(file=config_path)
        self.conf = self.conf_parser.config
        if self.conf is None:
            self.close()
        self.test_abbreviation = test_abbreviation
        if self.test_abbreviation is None:
            self.test_abbreviation = self.conf[SAVE_VALUES_SECTION_HEADER][TEST_ABBR]
        self.e_evaluate = electric_evaluation
        self.auto_detect_test_bar = False  # can be changed by user during operatioon

        # Dicts, hard coded lists, constants
        self.key_dict = {
            self.conf[RESULT_VALUES_SECTION_HEADER][POSITIVE]: ['p', '1', ],
            self.conf[RESULT_VALUES_SECTION_HEADER][NEGATIVE]: ['n', '0', ],
            self.conf[RESULT_VALUES_SECTION_HEADER][UNCLEAR]: ['u', '7', ],
            HELP_KEY: ['h', keyboard.Key.f1, ],
            SAVE_KEY: ['s', ],  # keyboard.Key.enter,
            ESCAPE: [keyboard.Key.esc, ],
        }
        if self.e_evaluate:
            self.key_dict[SWITCH_AUTO_ROI] = ['a', ]
            self.key_dict[SET_ROI] = ['r', ]
        self.wat = False
        if self.test:
            self.key_dict[SWITCH_BRIGHTNESS_KEY] = ['q', ]

        self.used_keys = [item for sublist in [*self.key_dict.values()] for item in sublist]
        self.base_result_dict = {
            self.test_abbreviation: self.test_abbreviation,
            BARCODE: None,
            E_ABBREV: E_ABBREV,
            TRESULT: None,
            COUNTER: 0,
            IMGTIME: None,
            SAVE_OK: False,
            IMGPATH: None,
            E_RESULT: None,
            SAVEPATH: None,
        }
        self.decode_symbols = [
            pyzbar.ZBarSymbol.I25,
            pyzbar.ZBarSymbol.QRCODE,
            pyzbar.ZBarSymbol.CODE128,
        ]
        if self.test:
            self.decode_symbols = None
        self.result_dict = dict()
        self.backup_dict = dict()
        self.save_vals = [BARCODE, self.test_abbreviation, IMGTIME, TRESULT]
        self.save_vals_ = [i.replace(TEST_ABBR, self.test_abbreviation) for i in self.conf[
            SAVE_VALUES_SECTION_HEADER
        ][SAVE_VALS_ORDER]]
        # print(self.save_vals_)
        # print(self.save_vals)

        self.save_vals_electric = [BARCODE, E_ABBREV, IMGTIME, TRESULT, E_RESULT]

        self.barcode = None
        # self.result = None
        self.past_result = None
        self.e_result = None
        self.result_time = None
        self.img_path = None

        self.camera = Camera(dims=(1240, 1024), log=self.log)

        self.window_name = test_abbreviation
        self.frame = self.camera.frame.copy()
        self.display_frame = self.camera.frame.copy()
        self.kb_listener = keyboard.Listener(on_press=self.key_press)
        self.kb_listener.start()
        self.bbox_roi = None
        self.frame_roi_bw = None
        self.event_counter = 0
        self.event_counter_max = 30
        self.event_counter_big = 10**10
        self.counter = 0

        # Queues
        queue_length = 10
        self.barcode_queue = deque(maxlen=5)
        self.elec_pred_queue = deque(maxlen=queue_length)
        self.corner_queue = deque(maxlen=queue_length)
        # self.result_queue = deque(maxlen=queue_length)
        # self.graph_queue = deque(maxlen=5)

        self.queue_list = [
            self.barcode_queue,
            self.elec_pred_queue,
            self.corner_queue,
            # self.result_queue,
            # self.graph_queue,
        ]
        self.turn_image = False
        self.save_image_turned = False
        self.display_image_turned = False
        self.saved_barcode_history = []

        self.past_unsaved_list = []

        self.display_text = []
        self.display_boxes = []
        self.upper_text_list = []
        self.lower_text_list = []

        self.line_manager = LineManager()

        self.min_scale_fac = .2
        self.max_scale_fac = .6

        self.ln_length = 100
        self.ln_img_max = 255
        self.ln_deriv_max = 150
        self.vlines = []
        for c in ['tab:red', 'xkcd:grey', 'xkcd:grey', 'tab:red']:
            self.vlines.append(
                self.line_manager.add_line(axs=1, linestyle='--', c=c, zorder=1, )
            )
        self.hlines = []
        for c in ['tab:red', 'tab:red', 'xkcd:grey', 'xkcd:grey', ]:
            self.hlines.append(
                self.line_manager.add_line(axs=1, c=c, zorder=1, )
            )
        self.up_triangle = self.line_manager.add_line(
            axs=1,
            zorder=5,
            linestyle='None',
            marker='^',
        )
        self.down_triangle = self.line_manager.add_line(
            axs=1,
            zorder=5,
            linestyle='None',
            marker='v',
        )
        self.ln_img = self.line_manager.add_line(
            axs=0,
            zorder=100,
            coords=(np.linspace(0, self.ln_length, self.ln_length), np.linspace(0, self.ln_img_max, self.ln_length)),
            c='tab:blue',
        )
        self.ln_der = self.line_manager.add_line(
            axs=1,
            zorder=100,
            coords=(
                np.linspace(0, self.ln_length, self.ln_length),
                np.linspace(-self.ln_deriv_max, self.ln_deriv_max, self.ln_length)
            ),
            c='tab:blue',
            path_effects='line',
        )
        self.txt_im = self.line_manager.add_anotate(axs=0, txt='0')
        self.line_manager.add_anotate(axs=0, txt="  ", coords=(1, 1))  # right side padding
        self.c_fig = self.line_manager.update_figure()

        if self.camera.cap is None:
            self.close()

        # Logging infos
        self.log.info(
            f"Fuer hilfe: {', '.join(str(k).replace('Key.', '').capitalize() for k in self.key_dict[HELP_KEY])}"
        )
        try:
            import win32api
            win32api.SetConsoleCtrlHandler(self.close, True)
        except ImportError:
            pass

        # Self-start
        self.observe()

    def update_settings(self, settings_dict):
        pass

    # Graph
    def find_test_bar(self):
        if self.bbox_roi is None:
            self.log.debug('bbox_roi is None')
            return None
        x, y, w, h = self.bbox_roi
        if self.turn_image and w < 0:
            x += w
            w = abs(w)
        a, c = 0, 0
        edged = self.frame.copy()[y:y+h, x:x+w, :]

        if self.auto_detect_test_bar:
            edged = cv2.cvtColor(edged, cv2.COLOR_BGR2GRAY)
            edged = self.normalise_image(edged)
            if self.test:
                self.display(edged, 'wat1')
            # find automatic way to determine lower threshold
            kernel = np.ones((3, 3), np.uint8)
            edged = cv2.blur(edged, (3, 3))
            # edged = self.global_img_centering(edged, upper_limit=255)
            if self.test:
                self.display(edged, 'wat2')
            edged = cv2.erode(edged, kernel, iterations=1)
            if self.test:
                self.display(edged, 'wat3')
            edged = cv2.morphologyEx(edged, cv2.MORPH_OPEN, kernel)
            if self.test:
                self.display(edged[:, :], 'wat4')
            sigma = .33
            lower = int(max(0, (1.0 - sigma) * np.median(edged[:, :])))
            _, edged = cv2.threshold(edged, lower, self.ln_img_max, cv2.THRESH_BINARY_INV)
            if self.test:
                self.display(edged, 'wat5')
            cpix_contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not len(cpix_contours):
                self.log.debug('No contours')
                return None
            # select 5 largest contours by area
            cpix_contours = sorted(cpix_contours, key=lambda contours: cv2.contourArea(contours))[-5:]

            # select topmost contour
            # Reference:
            # https://www.pyimagesearch.com/2015/04/20/sorting-contours-using-python-and-opencv/
            cpix_contours = sorted(
                zip(cpix_contours, [cv2.boundingRect(c) for c in cpix_contours]),
                key=lambda contours: contours[1][1]
            )[0][0]
            rect = cv2.minAreaRect(cpix_contours)
            box = cv2.boxPoints(rect)
            ccorners = np.int0(box)

            bar_length = int(max(ccorners[:, 0]) * 1) - int(min(ccorners[:, 0]) * 1)

            a, b = int(min(ccorners[:, 1]) - .5 * bar_length), int(min(ccorners[:, 1]) + bar_length * 2.5),
            c, d = int(min(ccorners[:, 0]) * 1), int(max(ccorners[:, 0]) * 1)
            dc = int((d - c) * .3)
            c += dc
            d -= dc
            # cpix = None
            # # Limit to bbox size
            a, b, c, d = max(a, 0), max(b, 0), max(c, 0), max(d, 0)
            d = min(w, d)
            b = min(h, b)

            # ...................... [x, y, w, h]
            self.corner_queue.append([a, b, c, d])
            a, b, c, d = np.average(
                np.asarray(self.corner_queue),
                weights=npaverage_weights(self.corner_queue),
                axis=0,
            ).astype(int)
            self.display_boxes.append((self.display_frame, x+c, y+a, d-c, b-a,  (200, 50, 70), 1))
            edged = self.frame[y+a:y+b, x+c:x+d]
        if edged.size < 100:
            return None
        edged = cv2.cvtColor(edged, cv2.COLOR_BGR2GRAY)
        if self.test:
            self.display(self.normalise_image(edged), 'normalised')
            self.display(global_img_centering(edged, upper_limit=255), 'centered')
        if self.wat:
            edged = self.normalise_image(edged)
        else:
            edged = global_img_centering(edged, upper_limit=255)
        displ_edged = convert_to_colormap(edged)
        if displ_edged is not None:
            self.display_frame[y+a:y+a+edged.shape[0], x+c:x+c+edged.shape[1], :] = displ_edged
            pass
        return edged

    def scale_ret_frm(self, ret_frm):
        assert ret_frm.size > 100
        cpix_h, cpix_w = ret_frm.shape[:2]
        scale_w, scale_h = int(cpix_w * self.ln_length / cpix_h), self.ln_length
        try:
            img = cv2.resize(
                ret_frm,
                (scale_w, scale_h),
                interpolation=cv2.INTER_AREA,
            )
            # if self.transpose:
            #     return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            return img
        except cv2.error:
            return None

    def average_ret_frm(self, ret_frm):
        if ret_frm is None:
            return None
        assert ret_frm.shape[0] == self.ln_length
        if len(ret_frm.shape) > 2:
            ret_frm = cv2.cvtColor(ret_frm, cv2.COLOR_BGR2GRAY)
        ret_frm = np.average(ret_frm, axis=1).flatten()

        # ret_frm = rollavg(ret_frm, 5)
        # self.graph_queue.append(ret_frm)
        # ret_frm = np.average(
        #     np.asarray(self.graph_queue),
        #     weights=npaverage_weights(self.graph_queue),
        #     axis=0,
        # )
        return ret_frm  # np.interp(ret_frm, (ret_frm.min(), ret_frm.max()), (0, self.ln_img_max))

    def first_deriv(self, brightness_ln, n_diff=1):
        assert len(brightness_ln) == self.ln_length
        return np.diff(
            brightness_ln,
            n=n_diff,
            axis=0,
            prepend=brightness_ln[:n_diff],
            # append=0,
        ).flatten()

    def find_peaks(self, first_d_graph):
        # @todo optionally set to min/max of test peak
        ax_min = min(first_d_graph[:])  # int(self.ln_length / 2)
        ax_min_low = ax_min * self.min_scale_fac
        ax_minhigh = ax_min * self.max_scale_fac

        ax_max = max(first_d_graph[:])  # int(self.ln_length / 2)
        ax_max_low = ax_max * self.min_scale_fac
        ax_maxhigh = ax_max * self.max_scale_fac

        pos_peaks = find_peaks(
            first_d_graph[:],
            prominence=ax_max_low,
        )[0]
        # Invert for neg.
        neg_peaks = find_peaks(
            first_d_graph[:] * -1,
            prominence=ax_min_low * -1,
        )[0]

        triangle_offset = int(self.ln_deriv_max * .1)
        self.down_triangle.set_data(
            pos_peaks,
            [i if i < self.ln_deriv_max else self.ln_deriv_max for i in first_d_graph[pos_peaks] + triangle_offset],
        )
        self.up_triangle.set_data(
            neg_peaks,
            [i if i > -self.ln_deriv_max else -self.ln_deriv_max for i in first_d_graph[neg_peaks] - triangle_offset],
        )
        for line, val in zip(self.vlines, [ax_minhigh, ax_min_low, ax_max_low, ax_maxhigh]):
            line.set_data(
                [0, self.ln_length],
                [val, val]
            )

        # @todo set values in settings
        min_control_peak = 10
        max_control_peak = 40

        min_test_peak = 50
        max_test_peak = 90

        self.txt_im.set_text(
            f"Control band: Positive: {any([min_control_peak <= c < max_control_peak for c in pos_peaks])}, "
            f"Negative: {any([min_control_peak <= c < max_control_peak for c in neg_peaks])}\n"
            f"Test band: Positive: {any([min_test_peak <= c < max_test_peak for c in pos_peaks])}, "
            f"Negative: {any([min_test_peak <= c < max_test_peak for c in neg_peaks])}"
        )

        for line, val in zip(self.hlines, [min_control_peak, max_control_peak, min_test_peak, max_test_peak]):
            line.set_data(
                [val, val],
                [-self.ln_deriv_max, self.ln_deriv_max],
            )

    # @threaded
    def analyse_roi(self):
        try:
            if not self.e_evaluate or self.bbox_roi is None:
                return False
            # update self.frame_roi_bw / test_area_image
            test_area_image = self.find_test_bar()
            if test_area_image is None:
                return False
            self.frame_roi_bw = test_area_image

            if self.test:
                self.display(self.frame_roi_bw, 'wat 6')
            scaled_frame = self.scale_ret_frm(self.frame_roi_bw)
            if self.test:
                self.display(scaled_frame, 'wat 7')
            average_frame = self.average_ret_frm(scaled_frame)
            self.ln_img.set_ydata(average_frame)
            first_deriv = self.first_deriv(average_frame)
            self.find_peaks(first_deriv)
            self.ln_der.set_ydata(first_deriv)

        except (AssertionError, cv2.error) as ex:
            if self.test:
                self.log.exception(ex)
            return False
        return True

    # Barcode
    def barcode_check(self):
        barcodes = pyzbar.decode(self.frame, symbols=self.decode_symbols, )
        if not barcodes:
            self.barcode_queue.append(None)
            return
        rgb = (0, 60, 255)

        for barcode in barcodes:
            barcode_data = barcode.data.decode('utf-8')
            if not self.conf[
                       BARCODE_SECTION_HEADER][MIN_BARCODE_LENGTH] < len(barcode_data) <= self.conf[
                BARCODE_SECTION_HEADER][MAX_BARCODE_LENGTH]:
                continue
            # elif self.test:
            #     self.log.debug(barcode_data)
            if self.conf[BARCODE_SECTION_HEADER][BARCODE_ONLY_NUMERIC]:
                try:
                    _ = int(barcode_data)
                except ValueError:
                    continue
            (x, y, w, h) = barcode.rect
            self.display_boxes.append((self.display_frame, x, y, w, h, rgb))
            if self.test:
                self.display_text.append((self.display_frame, x, y - 10, f"{barcode_data}", rgb))
            text = f"Barcode: {barcode_data}"
            if self.test:
                text += f" {barcode.type}"
            self.barcode_queue.append(text)

    def unsaved_barcodes_list(self, max_len=5):
        res_text = []
        for k, rdict in self.result_dict.items():
            if not rdict[SAVE_OK]:
                res_text.append(f"{k}: {rdict[TRESULT]}")
        if res_text:
            res_text.append(
                f"Letzten {len(res_text[-max_len:])} von {len(res_text)} ungespeicherten Ergebnisse:"
            )
        return res_text[-max_len-1:]

    # Utility
    def annotate_image(self):
        if self.bbox_roi is None and self.e_evaluate:
            self.upper_text_list.append(f"No ROI defined - press {self.key_dict[SET_ROI][0].upper()} to set ROI")

        elif self.e_evaluate:
            x, y, w, h = self.bbox_roi
            self.display_boxes.append((self.display_frame, x, y, w, h, (230, 30, 20), 1))
            if self.test:
                self.display_boxes.append((self.display_frame, x, y, 1, 1, (230, 200, 20), 3))

        rgb = (0, 60, 255)
        if self.event_counter > 0:
            border_width = 20
            if self.past_result == self.conf[RESULT_VALUES_SECTION_HEADER][POSITIVE]:
                border_width += self.event_counter
                rgb = (255, 0, 0)
            self.display_boxes.append((
                self.display_frame, 0, 0, self.frame.shape[1], self.frame.shape[0], rgb, border_width
            ))

        for args in self.display_boxes:
            put_rectangle(*args)
        self.display_boxes = []

        res_text_frame = self.past_result
        self.check_orientation_for_display()
        img_height, img_width = self.display_frame.shape[:2]
        if self.past_result is not None:
            if self.past_result == self.conf[RESULT_VALUES_SECTION_HEADER][POSITIVE]:
                rgb = (255, 0, 0)
            if self.past_result == 'ERROR':
                if self.event_counter % 2:
                    rgb = (255, 0, 0)
                else:
                    rgb = (255, 80, 80)
                res_text_frame = 'Kein Barcode erkannt!'
            if self.past_result == 'Bereits hochgeladen':
                if self.event_counter % 2:
                    rgb = (255, 0, 0)
                else:
                    rgb = (255, 80, 80)
                res_text_frame = self.past_result
            self.display_text.append((
                self.display_frame,
                .2 * img_width,
                .5 * img_height,
                res_text_frame,
                rgb,
                2,
                8
            ))

        for args in self.display_text:
            put_text(*args)
        self.display_text = []

        self.upper_text_list.extend([i for i in set(self.barcode_queue) if i])
        put_text_list(self.display_frame, text_list=self.upper_text_list, rgb=(0, 60, 255))
        self.upper_text_list = []

        self.lower_text_list.extend(self.unsaved_barcodes_list())
        put_text_list(self.display_frame, text_list=self.lower_text_list, rgb=(0, 60, 255), count_up=False)
        self.lower_text_list = []

        self.update_figure()
        self.add_graph(own_window=False)

    @threaded
    def screensaver(self):
        self.line_manager.screen_saver()
        self.ln_der.set_ydata(
            np.sin(np.linspace(0, 2 * np.pi, self.ln_length) + (self.counter / 100) * np.pi) * self.ln_deriv_max
        )
        self.txt_im.set_text('Waiting...')
        self.ln_img.set_ydata(
            (np.cos(np.linspace(0, 2 * np.pi, self.ln_length) +
                    (self.counter / 100) * np.pi) + 1) / 2 * self.ln_img_max
        )

    def saved_barcode_history_update(self):
        for k, rdict in self.result_dict.items():
            if rdict[SAVE_OK]:
                self.saved_barcode_history.append(k)
        self.saved_barcode_history = list(set(self.saved_barcode_history))

    # Utility - Frame
    def check_orientation_for_display(self, check_save_frame=False):
        if self.display_image_turned:
            self.display_frame = cv2.rotate(self.display_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.display_image_turned = False
        if self.save_image_turned and check_save_frame:
            self.frame = cv2.rotate(self.frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.save_image_turned = False

    def set_roi(self):
        # self.bbox_roi = None
        if not self.e_evaluate:
            return
        self.reset_current_values()
        self.set_frame()  # repeated here, as otherwise cv2.selectROI misbehaves from time to time
        self.check_orientation_for_display()
        put_text_list(
            self.display_frame,
            ['Press "ENTER" or "SPACE" after selecting the ROI, "C" to cancel'],
            font_scale=.8,
        )
        bbox_roi = cv2.selectROI(
            self.window_name,
            self.display_frame,
            fromCenter=False,
            showCrosshair=True,
        )
        if bbox_roi[2] > 1.2 * bbox_roi[3]:
            if not self.turn_image:
                self.log.info('Test area set to horizontal')
            self.turn_image = True
            self.set_frame()
            bbox_roi = self.rotate_box(bbox_roi)
        else:
            if self.turn_image:
                self.log.info('Test area set to vertical')
            self.turn_image = False
        if sum(bbox_roi) > 0:
            self.bbox_roi = bbox_roi
            if self.test:
                cv2.destroyAllWindows()
        self.past_result = None
        self.event_counter = 0
        if self.test:
            self.log.debug(f"{self.bbox_roi}, {self.turn_image}")
        self.reset_current_values()

    def set_frame(self):
        self.camera.get_frame()
        cv2.waitKey(1)
        self.frame = self.camera.frame.copy()
        if self.turn_image:
            self.frame = cv2.rotate(self.frame, cv2.ROTATE_90_CLOCKWISE)
            self.display_image_turned = True
            self.save_image_turned = True
        self.display_frame = self.frame.copy()

    def display(self, frame, window_name=None):
        if window_name is None:
            window_name = self.window_name
        resize = cv2.WINDOW_AUTOSIZE
        if self.test:
            # resize = cv2.WINDOW_NORMAL
            pass
        try:
            cv2.namedWindow(window_name, resize)
            cv2.imshow(window_name, frame)
        except cv2.error:
            self.log.warning(f"Set frame error with window {window_name}")

    def display_help(self, texts=''):
        self.display_frame[:, ] = [240, 100, 90]
        help_text = ["Help:", '', ]
        for k, v in self.key_dict.items():
            help_text.append(
                f"Key(s) for {str(k).replace('Key.', '')}: "
                f"{', '.join(str(k).replace('Key.', '').capitalize() for k in v)}"
            )
        help_text.extend(['', 'In case of errors: tests with identical barcodes will be overwritten.'])
        if texts:
            help_text.extend(texts)
        put_text_list(
            frame=self.display_frame,
            text_list=help_text,
            rgb=(0, 0, 0)
        )

    def rotate_box(self, bb, img_height=None):
        if img_height is None:
            img_height = self.frame.shape[1]
        x, y, w, h = bb
        return [img_height - y, x, -h, w]

    def result_string(self, rdict, save_vals):
        res_str = self.conf[SAVE_VALUES_SECTION_HEADER][SAVE_VALUES_JOIN_CHAR].join(str(rdict[v]) for v in save_vals)
        return f"{res_str}\n\n"

    def reset_current_values(self):
        self.barcode = None
        # self.result = None
        self.e_result = None
        self.result_time = None
        self.img_path = None
        if self.test:
            # cv2.destroyAllWindows()
            pass
        for c_queue in self.queue_list:
            c_queue.clear()

    def normalise_image(self, image):
        return cv2.normalize(src=image, dst=None, alpha=0, beta=self.ln_img_max, norm_type=cv2.NORM_MINMAX)

    def close(self, *_):
        self.save_results()
        self.upload_results()
        if self.test:
            pass
        try:
            os.remove(self.backup_path)
            self.log.warning('Deleting Backup')
        except FileNotFoundError:
            pass
        tp.shutdown()
        self.log.info(f'Closing {self.window_name}')
        self.running = False
        self.camera.close_camera()
        cv2.destroyAllWindows()
        self.line_manager.close()
        self.kb_listener.stop()
        for handler in self.log.handlers[:]:
            handler.flush()
            handler.close()
            self.log.removeHandler(handler)
        sleep(.1)

    # Figure
    def add_graph(self, own_window=True):
        if self.c_fig is None or not self.e_evaluate:
            return
        if own_window:
            self.display(frame=self.c_fig, window_name=f"{self.window_name}_Graph")
            return
        if self.display_frame.size < 100:
            return
        dim_h, dim_w, dim_chanells = self.display_frame.shape
        img_h, img_w, _ = self.c_fig.shape

        dim_frac = dim_w / img_w

        resized = cv2.resize(
            self.c_fig,
            (int(dim_frac * img_w), int(dim_frac * img_h)),
            cv2.INTER_AREA
        )
        empty = np.ones(
            (int(dim_frac * img_h), dim_w, dim_chanells),
            dtype=self.display_frame.dtype
        ) * self.display_frame.max().max()
        empty[:, :, :] = resized[:, :dim_w, :]
        self.display_frame = np.vstack((self.display_frame, empty[:, :, :]))

    def update_figure(self):
        self.c_fig = self.line_manager.update_figure()

    def save_frame(self, result):
        now = get_strftime(msec=True)
        barcode = safe_str(self.barcode)
        result = safe_str(result)
        file_name = os.path.join(self.local_path, f"{now}_{barcode}_{result}.png")
        self.check_orientation_for_display(check_save_frame=True)
        try:
            cv2.imwrite(file_name, self.frame)
            self.log.info(f"Image saved {barcode}: {result} Path: {file_name}")
        except Exception as ex:
            self.log.exception(ex)
            self.log.warning(f"Image NOT saved: {barcode}")
        self.log.info(f"Timestamp: {now[:-6]}")
        return file_name, now

    # Results
    def key_press(self, key):
        # get key character
        try:
            key = key.char
        except AttributeError:
            key = key
        # Return if not used
        if key not in self.used_keys and not self.test:
            return
        # set key value for result & past result
        current_result = None
        past_result = self.past_result
        for key_val, key_list in self.key_dict.items():
            if key in key_list:
                # self.result = key_val
                current_result = key_val
                past_result = key_val
                break
        if self.test:
            self.log.debug(f"{key}, {current_result}")
        # Return if nothing was set
        if current_result is None:
            return
        # Something happened, set event counter to max
        self.event_counter = self.event_counter_max
        # Program ends/user input required: event counter can be endless
        if current_result in [ESCAPE, SET_ROI]:
            self.event_counter = self.event_counter_big
        if current_result == SWITCH_AUTO_ROI:
            self.reset_current_values()
            self.auto_detect_test_bar = not self.auto_detect_test_bar
        if current_result == SWITCH_BRIGHTNESS_KEY:
            self.wat = not self.wat
        try:
            if current_result in [
                self.conf[RESULT_VALUES_SECTION_HEADER][POSITIVE],
                self.conf[RESULT_VALUES_SECTION_HEADER][NEGATIVE],
                self.conf[RESULT_VALUES_SECTION_HEADER][UNCLEAR]
            ]:
                check_ok = self.set_results()
                self.img_path, self.result_time = self.save_frame(current_result)
                if self.barcode in self.saved_barcode_history:
                    past_result = 'Bereits hochgeladen'
                if check_ok:
                    # [self.barcode, self.result, self.e_result, self.result_time, self.img_path, ]
                    self.e_result = 1
                    self.update_result_dict(current_result)
                    self.save_results()
                    self.log.info(self.result_dict)
                    self.log.info(f"{self.barcode}, {len(self.barcode_queue)}")
                else:
                    past_result = 'ERROR'
            if current_result == SAVE_KEY:
                self.upload_results()
            self.past_result = past_result
        except Exception as ex:
            self.log.exception(ex)
        self.reset_current_values()

    def update_result_dict(self, result):
        check_list = [self.barcode, result, self.e_result, self.result_time, self.img_path, ]
        assert not any([i is None for i in check_list]), self.log.error(
            f"update result dict had None value in list: {', '.join(str(i) for i in check_list)}"
        )
        if self.barcode in self.result_dict:
            if self.result_dict[self.barcode][SAVE_OK]:
                self.log.warning(f"Barcode already uploaded: {self.barcode}")
                return
            c_counter = self.result_dict[self.barcode][COUNTER] + 1
            self.result_dict[self.barcode].update({
                TRESULT: result,
                COUNTER: c_counter,
                IMGTIME: self.result_time,
                IMGPATH: self.img_path,
                E_RESULT: self.e_result,
            })
            return
        new_entry = copy.deepcopy(self.base_result_dict)
        new_entry.update({
            BARCODE: self.barcode,
            TRESULT: result,
            IMGTIME: self.result_time,
            IMGPATH: self.img_path,
            E_RESULT: self.e_result,
        })
        self.saved_barcode_history_update()
        self.result_dict[self.barcode] = new_entry

    def set_results(self):
        self.barcode = noneless_result(self.barcode_queue)
        if self.e_evaluate:
            self.e_result = noneless_result(self.elec_pred_queue)
        else:
            self.e_result = True
        if self.test:
            self.e_result = True
        if self.barcode and self.e_result:
            return True
        return False

    def check_rdict_ok(self, rdict, barcode=None, checklist=None, check_save_ok=True):
        if not isinstance(rdict, dict):
            self.log.critical(f"No rdict")
            self.log.warning(f"{type(rdict)}")
            self.log.warning(f"{rdict}")
            return False
        if barcode is not None:
            assert str(barcode) == str(rdict[BARCODE]), self.log.critical(
                f"Assertion error with keys '{barcode}' and '{rdict[BARCODE]}', values:\n{rdict}"
            )
        else:
            assert rdict[BARCODE] is not None, self.log.critical(
                f"Assertion error with keys '{barcode}' and '{rdict[BARCODE]}', values:\n{rdict}"
            )
            barcode = rdict[BARCODE]
        if rdict[SAVE_OK] and check_save_ok:
            self.log.debug(f"{barcode} save ok True, not saving")
            return False
        if checklist is not None:
            if not check_dict(rdict, checklist):
                self.log.critical(
                    f"Value is None in dict of Barcode '{barcode}', values:\n{rdict}"
                )
                return False
        return True

    def save_results(self):
        self.backup()
        save_list = []
        for k, rdict in self.result_dict.items():
            if not self.check_rdict_ok(rdict, k, set(self.save_vals + self.save_vals_electric)):
                self.log.warning(f"Result not fully set: {rdict}")
                continue
            res_string = self.result_string(rdict, self.save_vals)
            save_path = os.path.join(
                self.local_path,
                f"{rdict[IMGTIME]}_{safe_str(k)}{self.conf[SAVE_VALUES_SECTION_HEADER][FILE_ENDING]}"
            )
            save_list.append((res_string, save_path))
            if self.e_evaluate:
                eye_res_str = self.result_string(rdict, self.save_vals_electric)
                save_list.append((
                    eye_res_str,
                    os.path.join(
                        self.local_path,
                        f"{rdict[IMGTIME]}_{safe_str(k)}_technical_evaluation"
                        f"{self.conf[SAVE_VALUES_SECTION_HEADER][FILE_ENDING]}"
                    )
                ))
            self.result_dict[k][SAVEPATH] = save_path
        for string, path in save_list:
            with open(path, 'w+') as f:
                f.write(string)

    def upload_results(self):
        self.backup()
        error_count = 100
        while error_count:
            c_dict = {
                k: rdict for k, rdict in self.result_dict.items() if self.check_rdict_ok(
                    rdict=rdict, barcode=k, checklist=self.save_vals
                )
            }
            if not c_dict:
                break
            for k, rdict in c_dict.items():
                if rdict[SAVEPATH] is not None:
                    try:
                        copy2(rdict[SAVEPATH], self.ext_path)
                        self.result_dict[k][SAVE_OK] = True
                        continue
                    except Exception as ex:
                        error_count -= 1
                        self.log.exception(ex)
                res_string = self.result_string(rdict, self.save_vals)
                try:
                    with open(os.path.join(
                            self.ext_path,
                            f"{safe_str(k)}{self.conf[SAVE_VALUES_SECTION_HEADER][FILE_ENDING]}"), 'w+') as f:
                        f.write(res_string)
                    self.result_dict[k][SAVE_OK] = True
                except Exception as ex:
                    error_count -= 1
                    self.log.exception(ex)
        self.backup()

    def backup(self):
        c_backup_dict = {}
        try:
            with open(self.backup_path, 'rb') as f:
                c_backup_dict = pickle.load(f)
        except (EOFError, FileNotFoundError):
            pass
        except Exception as ex:
            self.log.exception(ex)
        if len(c_backup_dict):
            self.log.debug(c_backup_dict)
            self.backup_dict.update(c_backup_dict)
        self.backup_dict.update(self.result_dict)
        try:
            if len(self.backup_dict):
                with open(self.backup_path, 'wb+') as f:
                    pickle.dump(self.backup_dict, f)
        except Exception as ex:
            self.log.exception(ex)
        for k, rdict in self.backup_dict.items():
            if k in self.result_dict:
                continue
            if self.check_rdict_ok(rdict, k, check_save_ok=False):
                self.result_dict[k] = rdict
        self.saved_barcode_history_update()
        if self.test:
            # self.log.debug('Backed up stuff')
            pass

    # Main method
    def observe(self):
        self.backup()
        while self.running:
            if self.past_result == ESCAPE:
                break
            self.set_frame()
            self.barcode_check()
            if self.past_result == SET_ROI:
                self.set_roi()
                continue
            if not self.analyse_roi():
                self.screensaver()
            self.annotate_image()
            # self.update_figure()
            # self.add_graph(own_window=False)
            if self.past_result == HELP_KEY:
                self.display_help()
            self.display(self.display_frame)

            self.counter += 1
            if self.counter > 199:
                self.counter = 0
            if self.event_counter > 0:
                self.event_counter -= 1
            if self.event_counter == 0:
                self.past_result = None
        self.close()
