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
from configparser import ConfigParser
import os
from functools import wraps
from electric_eye.constants import (
    MIN_BARCODE_LENGTH, MAX_BARCODE_LENGTH, BARCODE_ONLY_NUMERIC, SAVE_VALS_ORDER,
    POSSIBLE_SAVE_VALUES_LIST, SAVE_VALUES_SECTION_HEADER, SAVE_VALUES_JOIN_CHAR,
    SAVE_VALUES_SECTION_HEADER,
    RESULT_VALUES_SECTION_HEADER,
    BARCODE_SECTION_HEADER,
    VARIOUS_SECTION_HEADER,
    LINE_LENGTH,
)
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.ini')

LOREM_IPSUM = [
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore "
    "et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut "
    "aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum "
    "dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui "
    "officia deserunt mollit anim id est laborum."
][0]


def decorator_factory(*funcs):
    def decorator(function):
        @wraps(function)
        def _wrapper(self, *args, **kwargs):
            for func in funcs:
                func(self)
            result = function(self, *args, **kwargs)
            return result
        return _wrapper
    return decorator


def as_dict(config):
    """
    Converts a ConfigParser object into a dictionary.

    The resulting dictionary has sections as keys which point to a dict of the
    sections options as key => value pairs.
    https://stackoverflow.com/questions/1773793/convert-configparser-items-to-dictionary
    """
    the_dict = {}
    for section in config.sections():
        the_dict[section] = {}
        for key, val in config.items(section):
            the_dict[section][key] = val
    return the_dict


class Config:
    def __init__(self, file=None):
        self.log = logging.getLogger(__name__)
        if file is None:
            file = BASE_PATH
        self.file = file
        assert os.path.isfile(self.file), f"settings.ini not found, path: {os.path.abspath(self.file)}"
        self.config_parser = ConfigParser()
        self.config = self.get_conf()

    def get_conf(self):
        try:
            self.config_parser.read(self.file)
            if SAVE_VALS_ORDER not in self.config_parser[SAVE_VALUES_SECTION_HEADER]:
                self.log.warning(f"{SAVE_VALS_ORDER} not set in .ini file; please set save values order")
                self.write_conf()
                return None
            # Flatten
            # config_dict = {kk: vv for _, v in self.config_parser.items() for kk, vv in v.items()}
            # convert to int/bool
            config_dict = as_dict(self.config_parser)
            config_dict[BARCODE_SECTION_HEADER][MIN_BARCODE_LENGTH] = int(
                config_dict[BARCODE_SECTION_HEADER][MIN_BARCODE_LENGTH]
            )
            config_dict[BARCODE_SECTION_HEADER][MAX_BARCODE_LENGTH] = int(
                config_dict[BARCODE_SECTION_HEADER][MAX_BARCODE_LENGTH]
            )
            config_dict[BARCODE_SECTION_HEADER][BARCODE_ONLY_NUMERIC] = self.config_parser.getboolean(
                BARCODE_SECTION_HEADER, BARCODE_ONLY_NUMERIC
            )
            config_dict[
                SAVE_VALUES_SECTION_HEADER
            ][SAVE_VALUES_JOIN_CHAR] = config_dict[SAVE_VALUES_SECTION_HEADER][SAVE_VALUES_JOIN_CHAR].strip()
            config_dict[
                SAVE_VALUES_SECTION_HEADER
            ][SAVE_VALS_ORDER] = [
                i.strip() for i in self.config_parser.get(SAVE_VALUES_SECTION_HEADER, SAVE_VALS_ORDER).split(
                    config_dict[
                        SAVE_VALUES_SECTION_HEADER
                    ][SAVE_VALUES_JOIN_CHAR]
                )
            ]
            for k, v in config_dict.items():
                print(f"{k}: {v}")
            return config_dict
        except Exception as ex:
            self.log.exception(ex)

    def write_conf(self):
        # write to file
        with open(self.file, 'w') as conf_file:
            self.config_parser.write(conf_file)


class BlankPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller


class Program(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.width = 600
        self.height = 400
        self.geometry(f"{self.width}x{self.height}")
        tk.Tk.iconbitmap(self, default="")
        tk.Tk.wm_title(self, "Eyo")
        self.frame_count = 0
        self.container = tk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        self.menu = tk.Menu(self.container)
        self.Add = Add
        self.BlankPage = BlankPage
        self.frames = {}
        self.change_func = None

    def add_frame(self, method, *args, **kwargs):
        frame = method(self.container, self, *args, **kwargs)
        frame.grid(row=0, column=0, sticky="nsew")
        # print(frame.__str__(), frame.title())
        self.frames[self.frame_count] = frame
        c = self.frame_count
        self.menu.add_command(
            label=f"Go to Page {self.frame_count + 1}",
            command=lambda: self.show_frame(c)
        )
        if self.frame_count > 0:
            self.config(menu=self.menu)
        self.frame_count += 1

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

    def close(self, event=None):
        print('Close called')
        # print(self.frames[0].position)
        self.destroy()


class Add(tk.Frame):
    # https://stackoverflow.com/questions/35616411/tkinter-going-back-and-forth-between-frames-using-buttons/35636241
    def __init__(self, parent, controller, page_count=3):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.position = 0
        self.page_count = page_count
        self.pages = []
        self.inner_frame = tk.Frame(self)
        self.inner_frame.place(relx=.5, rely=.5, anchor="c", relwidth=1.0, relheight=1.0)

        self.rowconf_n = 6
        for i in range(self.rowconf_n):
            self.inner_frame.grid_rowconfigure(i, weight=1)
            self.inner_frame.grid_columnconfigure(i, weight=1)

        self.nextButton = ttk.Button(self.inner_frame, text="Next")
        self.nextButton.grid(row=3, sticky='SE')

        self.backButton = ttk.Button(self.inner_frame, text="Back")
        self.backButton.grid(row=3, sticky='SW')

        for _ in range(self.page_count):
            self.new_page()

        # Initialise with no change in page to correct button placement & functionality
        self.change_page(change=0, index=0)

    def new_page(self):
        page = tk.Frame(self.inner_frame)
        page.grid(row=1, column=0, sticky="nsew")
        for i in range(self.rowconf_n):
            page.grid_rowconfigure(i, weight=1)
            page.grid_columnconfigure(i, weight=1)

        self.pages.append(page)

    def go_next(self, event=None):
        self.change_page(change=+1)

    def go_back(self, event=None):
        self.change_page(change=-1)

    def change_page(self, change, index=None):
        if self.controller.change_func is not None:
            rv = self.controller.change_func()
            if rv is not None:
                change = 0
                index = rv
        if index is not None:
            assert -len(self.pages) < index < len(self.pages), 'Index out of bounds'
            if index < 0:
                index = len(self.pages) + index
            self.position = index
        new_position = self.position + change
        if 0 <= new_position < len(self.pages):
            self.pages[new_position].tkraise()
            self.position = new_position

        if new_position + 1 >= len(self.pages):
            self.nextButton.config(text="Close", state=tk.NORMAL, command=self.controller.close)  # state=tk.DISABLED
        else:
            self.nextButton.config(text="Next", state=tk.NORMAL, command=self.go_next)

        if new_position - 1 < 0:
            self.backButton.config(text="First", state=tk.DISABLED, command=self.go_back)
        else:
            self.backButton.config(text="Back", state=tk.NORMAL, command=self.go_back)


class SetupTwo(Program):
    def __init__(self, config_dict):
        super().__init__()
        # self.change_func =
        self.join_char = config_dict[SAVE_VALUES_JOIN_CHAR]
        self.save_text = ''
        self.file_ending = ''
        self.number_of_peaks = 0
        titles = [
            'Start of setup',
            'Enter save variables in the order you want them to appear in the file\n'
            'Select them from the list below',
            'Set number of possible peaks/bars',
            'Set values for automatic assessment of peaks (0-100, whole numbers)',
            'Check entries',
        ]

        self.add_frame(self.Add, page_count=6)
        self.pages = self.frames[0].pages
        self.change_func = self.update_on_page_turn
        for page, title in zip(self.pages, titles):
            self.place_text_widget(
                tk.Message(page, text=title, width=int(self.width * .9)),
                sticky='NW',
            )
        # self.change_func_error = False

        # Page 0 - Intro
        cframe = self.pages[0]
        intro_text = ttk.Label(cframe, text=LOREM_IPSUM, wraplength=int(self.width * .9))
        intro_text.grid(sticky='nsew')

        # Page 1
        cframe = self.pages[1]
        # self.place_text_widget(
        #     tk.Message(
        #         cframe,
        #         text="Enter save variables in the order you want them to appear in the file\n"
        #              "Select them from the list below",
        #         width=int(self.width * .9)
        #     ),
        #     sticky="NW",
        #     row=0,
        # )
        self.save_variables_text_box = tk.Text(cframe, height=5)
        self.save_variables_text_box.grid(row=1, sticky='EW')

        self.csv_list = tk.Listbox(cframe, selectmode=tk.EXTENDED)
        self.csv_list.insert(0, *POSSIBLE_SAVE_VALUES_LIST)
        self.csv_list.grid(row=2, sticky='S')

        self.place_text_widget(ttk.Button(cframe, text='Insert text', command=self.insert_text), row=3, sticky='SE')
        self.place_text_widget(
            ttk.Button(cframe, text='Clear text', command=self.clear_save_variable_text), row=3, sticky='SW'
        )

        # width=len(text_save_file_ending),
        self.place_text_widget(ttk.Label(cframe, text='Save file ending:', ),  row=4, sticky='W')
        self.file_ending_box = self.place_text_widget(
            tk.Text(cframe, height=1, width=30), row=4, insert='.csv'
        )

        # Page 2
        cframe = self.pages[2]
        c_row = 3
        self.place_text_widget(
            ttk.Label(cframe, text='Number of peaks/possible bands (besides control band):', ),
            row=c_row, col=0, sticky='E'
        )
        self.number_of_peaks_text = self.place_text_widget(
            tk.Text(cframe, height=1, width=3),
            row=c_row, col=1, sticky='E', insert=1
        )
        self.place_text_widget(
            ttk.Label(cframe, text=f"{LINE_LENGTH}:", ),
            row=c_row, col=0, sticky='E'
        )
        self.ln_length = config_dict[LINE_LENGTH]
        self.ln_length_widget = self.place_text_widget(
            tk.Text(cframe, height=1, width=4),
            row=c_row, col=1, sticky='E', insert=self.ln_length
        )
        # self.number_of_peaks_text = self.place_text_widget(
        #     tk.Entry(cframe, validate='key', validatecommand=(self.validate_int, '%P')),
        #     row=c_row, col=1, sticky='E', insert=1
        # )

        # Page 3
        cframe = self.pages[3]
        c_row = 2
        self.place_text_widget(ttk.Label(cframe, text='Start/end of control peak:', ), row=c_row, col=0, sticky='E')
        self.start_control_peak = self.place_text_widget(
            tk.Text(cframe, height=1, width=3), row=c_row, col=1, sticky='E', insert=0
        )
        self.place_text_widget(ttk.Label(cframe, text='to', ), row=c_row, col=2, )
        self.end_control_peak = self.place_text_widget(
            tk.Text(cframe, height=1, width=3), row=c_row, col=3, sticky='W', insert=10
        )

        self.test_peaks_widget_list = []
        self.test_peak_value_list = []

        # Page 3
        cframe = self.pages[4]
        c_row = 2
        self.place_text_widget(ttk.Label(cframe, text='Minimal scale factor for peaks (0-1):', ), row=c_row, col=0, sticky='E')
        self.min_scale_fac = self.place_text_widget(
            tk.Text(cframe, height=1, width=3), row=c_row, col=1, sticky='W', insert=0.3
        )
        c_row = 3
        self.place_text_widget(ttk.Label(cframe, text='Maximal scale factor for peaks (0-1):', ), row=c_row, col=0, )
        self.max_scale_fac = self.place_text_widget(
            tk.Text(cframe, height=1, width=3), row=c_row, col=1, sticky='W', insert=0.6
        )

        # Page -1 - Check input
        cframe = self.pages[-1]
        self.end_text = ttk.Label(cframe, text=LOREM_IPSUM, width=int(self.width * .9), wraplength=int(self.width * .9))
        self.end_text.grid(sticky='nsew')
        # self.update_last_page()

        #
        self.show_frame(0)

    @staticmethod
    def place_text_widget(thing, row=None, col=None, sticky=None, insert=None, ):
        thing.grid(row=row, column=col, sticky=sticky, )
        if insert is not None:
            thing.insert('end', f"{insert}")
        return thing

    def setup_test_peak_frame(self):
        cframe = self.pages[3]
        for widget in self.test_peaks_widget_list:
            for w in widget:
                w.destroy()
        self.test_peaks_widget_list = []

        for i in range(self.number_of_peaks):
            c_row = 3 + i
            start = min(30 + i * 10, 90)
            end = min(40 + i * 10, 100)
            test_peak_start_end = ttk.Label(cframe, text=f"Start/end of test peak {i + 1}:", )
            test_peak_start_end.grid(row=c_row, column=0, sticky='E')
            start_test_peak = self.place_text_widget(
                tk.Text(cframe, height=1, width=3),
                row=c_row, col=1, sticky='E', insert=start
            )

            test_peak_end_label = ttk.Label(cframe, text="to", )  # width=len(text_save_file_ending),
            test_peak_end_label.grid(row=c_row, column=2, )

            end_test_peak = self.place_text_widget(
                tk.Text(cframe, height=1, width=3),
                row=c_row, col=3, sticky='W', insert=end
            )
            self.test_peaks_widget_list.append(
                (test_peak_start_end, start_test_peak, test_peak_end_label, end_test_peak)
            )

    def get_text_text_box(self):
        self.save_text = self.save_variables_text_box.get("1.0", "end-1c")
        return self.save_text

    def get_text_file_ending(self):
        self.file_ending = self.file_ending_box.get("1.0", "end-1c")
        # Add dot to file ending
        if self.file_ending[0] != '.' and '.' not in self.file_ending:
            self.file_ending = f".{self.file_ending}"
        return self.file_ending

    @staticmethod
    def error_window(warning, title=None,):
        if title is None:
            title = 'Error'
        messagebox.showerror(title, warning)

    def clear_save_variable_text(self):
        self.save_variables_text_box.delete("1.0", "end")
        self.get_text_text_box()

    def update_on_page_turn(self):
        ret_val = None
        self.get_text_text_box()
        self.get_text_file_ending()
        number_of_peaks = None
        try:
            number_of_peaks = int(self.number_of_peaks_text.get("1.0", "end-1c"))
        except ValueError:
            # self.change_func_error = True
            self.error_window('Number of peaks must be a whole number')
            ret_val = 2
        # number_of_peaks = self.number_of_peaks_text.get()
        if isinstance(number_of_peaks, int) and number_of_peaks != self.number_of_peaks:
            self.number_of_peaks = number_of_peaks
            self.setup_test_peak_frame()

        self.test_peak_value_list = []
        start_control_peak = None
        end_control_peak = None
        try:
            start_control_peak = int(self.start_control_peak.get("1.0", "end-1c"))
            end_control_peak = int(self.end_control_peak.get("1.0", "end-1c"))
            for i, (_, vala, _, valb) in enumerate(self.test_peaks_widget_list):
                start_val = int(vala.get("1.0", "end-1c"))
                end_val = int(valb.get("1.0", "end-1c"))
                self.test_peak_value_list.append((start_val, end_val))
        except ValueError:
            self.error_window(f"Start/End of peak(s) must be whole number")
            ret_val = 3

        test_band_fin_text = ''
        for i, (start, fin) in enumerate(self.test_peak_value_list):
            test_band_fin_text += f"\tTest band {i+1}:\t{start:2d} - {fin:2d}\n"
        # End text
        fin_text = [
            f"Saved values: {self.save_text}",
            f"Save file ending: {self.file_ending}\n",
            f"Number of peaks/bands besides control: {self.number_of_peaks}. Selected ranges",
            f"\tControl band:\t{start_control_peak:2d} - {end_control_peak:2d}",
            test_band_fin_text,
        ]
        self.end_text.configure(text='\n'.join(i for i in fin_text))

        # text.bind("<Configure>", lambda e: text.configure(wraplength=e.width-10))
        return ret_val

    def insert_text(self):
        text = self.join_char.join([self.csv_list.get(i) for i in self.csv_list.curselection()])
        if not len(text):
            return
        self.get_text_text_box()
        if self.save_text:
            if self.save_text[-1] != self.join_char.strip():
                text = self.join_char + text
        self.save_variables_text_box.insert('end', text)
        self.get_text_text_box()


if __name__ == '__main__':
    # setup = Setup()
    # setup.mainloop()
    config = Config().config
    app = SetupTwo(config)
    # app.state('zoomed')
    app.mainloop()
