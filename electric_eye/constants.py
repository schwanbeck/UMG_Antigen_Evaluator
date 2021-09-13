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

HELP_KEY = 'Hilfe'
SAVE_KEY = 'Ergebnisse speichern'
SWITCH_AUTO_ROI = 'Switching autom. ROI detection'
SWITCH_BRIGHTNESS_KEY = 'Switching brightness scaling'
LINE_LENGTH = 'Scale evaluation bar to size'
ESCAPE = 'Escape'
SET_ROI = 'Set area in which the test is'
BARCODE = 'Barcode'
TRESULT = 'Result'
COUNTER = 'Image save counter'
IMGTIME = 'Image Capture time'
SAVE_OK = 'save_ok'
IMGPATH = 'Latest image path'
E_ABBREV = 'E_EYE'
E_RESULT = 'Electric eye evaluation'
SAVEPATH = '.csv save path'
FILE_ENDING = 'file_ending'
MIN_BARCODE_LENGTH = 'min_barcode_length'
MAX_BARCODE_LENGTH = 'max_barcode_length'
BARCODE_ONLY_NUMERIC = 'only_numbers'
POSITIVE = 'positive'
NEGATIVE = 'negative'
UNCLEAR = 'unclear'
SAVE_VALS_ORDER = 'save_values_order'
TEST_ABBR = 'test_abbreviation'
SAVE_VALUES_JOIN_CHAR = 'save values separation character'
SAVE_VALUES_SECTION_HEADER = 'SAVE_VALUES'
RESULT_VALUES_SECTION_HEADER = 'RESULT_VALUES'
BARCODE_SECTION_HEADER = 'BARCODE'
VARIOUS_SECTION_HEADER = 'VARIOUS'

SECTION_HEADERS = [
    SAVE_VALUES_SECTION_HEADER,
    RESULT_VALUES_SECTION_HEADER,
    BARCODE_SECTION_HEADER,
    VARIOUS_SECTION_HEADER,
]

POSSIBLE_SAVE_VALUES_LIST = [
    BARCODE, TEST_ABBR, IMGTIME, TRESULT, IMGPATH, COUNTER, E_RESULT, SAVEPATH
]
