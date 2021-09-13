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

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patheffects


class BlitManager:
    def __init__(self, canvas, animated_artists=()):
        """
        Parameters
        ----------
        canvas : FigureCanvasAgg
            The canvas to work with, this only works for sub-classes of the Agg
            canvas which have the `~FigureCanvasAgg.copy_from_bbox` and
            `~FigureCanvasAgg.restore_region` methods.

        animated_artists : Iterable[Artist]
            List of the artists to manage
        # Source:
        # https://matplotlib.org/stable/tutorials/advanced/blitting.html
        """
        self.canvas = canvas
        self._bg = None
        self._artists = []

        for a in animated_artists:
            self.add_artist(a)
        # grab the background on every draw
        self.cid = canvas.mpl_connect("draw_event", self.on_draw)

    def on_draw(self, event):
        """Callback to register with 'draw_event'."""
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def add_artist(self, art):
        """
        Add an artist to be managed.

        Parameters
        ----------
        art : Artist

            The artist to be added.  Will be set to 'animated' (just
            to be safe).  *art* must be in the figure associated with
            the canvas this class is managing.

        """
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)
        self._artists.append(art)

    def _draw_animated(self):
        """Draw all of the animated artists."""
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update(self):
        """Update the screen with animated artists."""
        cv = self.canvas
        fig = cv.figure
        # paranoia in case we missed the draw event,
        if self._bg is None:
            self.on_draw(None)
        else:
            # restore the background
            cv.restore_region(self._bg)
            # draw all of the animated artists
            self._draw_animated()
            # update the GUI state
            cv.blit(fig.bbox)
        # let the GUI event loop process anything it has to do
        cv.flush_events()


class LineManager(BlitManager):
    def __init__(self, dims=None):
        if dims is None:
            dims = (10, 4)
        self.fig, axs = plt.subplots(2, 1, figsize=dims, sharex='all', constrained_layout=True)
        # self.fig.canvas.draw()
        self.axs = axs.flatten()
        self.lines = []
        self.texts = []
        super().__init__(canvas=self.fig.canvas)

    def screen_saver(self):
        for line in self.lines:
            line.set_ydata(
                [None]
            )
        for text in self.texts:
            text.set_text('')

    def add_line(self, axs=0, zorder=50, marker=None, linestyle='solid',
                 lw=1, c='#2B2B2B', path_effects='', coords=None):
        if coords is None:
            coords = ([], [])
        if 'line' in path_effects:
            path_effects = [patheffects.withStroke(linewidth=3, foreground='w')]  # , patheffects.Normal()
        line, = self.axs[axs].plot(
            coords[0],
            coords[1],
            zorder=zorder,
            # facecolors='none',
            marker=marker,
            c=c,
            animated=True,
            linestyle=linestyle,
            path_effects=path_effects,
            lw=lw,
        )
        self.add_artist(line)
        self.lines.append(line)
        return line

    def add_anotate(self, axs=0, txt='', coords=None, animated=True):
        if coords is None:
            coords = (0, 1)
        annot = self.axs[axs].annotate(
            txt,
            coords,
            xycoords="axes fraction",
            xytext=(10, -10),
            textcoords="offset points",
            ha="left",
            va="top",
            animated=animated,
        )
        if animated:
            self.add_artist(annot)
            self.texts.append(annot)
        return annot

    def update_figure(self):
        plt.gcf().canvas.get_renderer()
        self.fig.canvas.draw()
        img = np.array(self.fig.canvas.get_renderer()._renderer)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def close(self):
        plt.close(self.fig)


def put_text(frame, x, y, text, rgb=None, font_scale=None, line_type=None, font=None, ):
    if font is None:
        font = cv2.FONT_HERSHEY_SIMPLEX
    if line_type is None:
        line_type = 2
    if font_scale is None:
        font_scale = 1
    if rgb is None:
        font_color_r = 255
        font_color_g = 0
        font_color_b = 0
    else:
        font_color_r, font_color_g, font_color_b = rgb
    cv2.putText(
        frame,
        text,
        (int(x), int(y)),  # bottomLeftCornerOfText
        font,
        font_scale,
        (font_color_b, font_color_g, font_color_r),
        line_type
    )


def put_text_list(frame, text_list, rgb=None, font_scale=None, line_type=None, font=None, count_up=True):
    img_height, img_width = frame.shape[:2]
    for i, text in enumerate(text_list):
        if count_up:
            y_base = .04 * img_height
            y_increase = img_height * (.05 * i)
        else:
            y_base = .96 * img_height
            y_increase = img_height * (.05 * i) * -1
        put_text(
            frame=frame,
            text=text,
            x=img_width * .005,
            y=y_increase + y_base,
            rgb=rgb,
            font_scale=font_scale,
            line_type=line_type,
            font=font
        )


def put_rectangle(frame, x, y, w, h, rgb=None, line_width=2):
    if line_width is None:
        line_width = 2
    if rgb is None:
        font_color_b = 0
        font_color_g = 0
        font_color_r = 255
    else:
        font_color_r, font_color_g, font_color_b = rgb
    cv2.rectangle(
        frame,
        (int(x), int(y)), (int(x + w), int(y + h)),
        (font_color_b, font_color_g, font_color_r),
        line_width
    )


def convert_to_colormap(image, cmap='viridis', convert_to_bgr=True, max_val=255):
    if image is None:
        return None
    # Source: Nathancy
    # https://stackoverflow.com/questions/59478962/how-to-convert-a-grayscale-image-to-heatmap-image-with-python-opencv
    cmap = plt.get_cmap(cmap)
    try:
        image = cmap(image)
    except TypeError:
        print(type(image), image.dtype)
        image = np.full_like(image, np.NAN)
    image = image * max_val
    image = image.astype(np.uint16)[:, :, :3]
    if convert_to_bgr:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image
