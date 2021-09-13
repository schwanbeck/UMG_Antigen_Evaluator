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

import os
import logging
import argparse
from electric_eye.EE import ElectricEye
from electric_eye.helper_func import logging_setup


if __name__ == '__main__':
    os.system('CLS')
    parser = argparse.ArgumentParser(
        description='Antigen test scanner of the Goettinger UMG'
    )
    parser.add_argument(
        '--savepath',
        '-s',
        nargs='?',
        default=os.getcwd(),
        const=os.getcwd(),
        help='Path where generated .csv files will be saved',
    )
    parser.add_argument(
        '--basepath',
        '-b',
        nargs='?',
        default=os.getcwd(),
        const=os.getcwd(),
        help='Local path where files, images, and backups will be saved',
    )
    parser.add_argument(
        '--logpath',
        '-l',
        nargs='?',
        default=os.getcwd(),
        const=os.getcwd(),
        help='Path where log files will be saved',
    )
    parser.add_argument(
        '--testabbreviation',
        '-t',
        nargs='?',
        default='ncovagna',
        const='ncovagna',
        help=f"Used test name/abbreviation",
    )
    args = parser.parse_args()
    logging_setup('EE', './log', test_abbreviation=str(args.testabbreviation))
    log = logging.getLogger(__name__)
    log.info('Session started - please wait')
    ElectricEye(
        test=True,
        base_path=args.savepath,
        ext_path=args.basepath,
        test_abbreviation=args.testabbreviation,
        electric_evaluation=True,
    )
