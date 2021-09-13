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
from logging.config import dictConfig
import os
import numpy as np
from socket import gethostname
from scipy.signal import convolve
from getpass import getuser
from datetime import datetime


def pathname(path):
    return os.path.abspath(os.path.realpath(os.path.expanduser(os.path.expandvars(path))))


def safe_str(s):
    return ''.join([c if c.isalnum() else '_' for c in str(s)])


def makedir(path):
    ret_path = './'
    try:
        path = pathname(path)
        os.makedirs(path, exist_ok=True)
        ret_path = path
    except Exception as ex:
        log = logging.getLogger(__name__)
        log.exception(ex)
        ret_path = pathname(ret_path)
    return ret_path


def rollavg(a, n):
    # https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-numpy-scipy
    assert n % 2 == 1
    return convolve(a, np.ones(n, dtype='float'), 'same') / convolve(np.ones(len(a)), np.ones(n), 'same')


def npaverage_weights(averaged_list):
    w = 1 / len(averaged_list)
    return [w * .1 + w * i for i in range(len(averaged_list))]


def noneless_result(res_list):
    noneless_res_queue = [i for i in res_list if i is not None]
    if not noneless_res_queue:
        return None
    return max(noneless_res_queue, key=noneless_res_queue.count)


def check_dict(dictionary, check_list):
    return all([True if dictionary[i] is not None else False for i in check_list])


def get_strftime(msec=False):
    timestr = '%Y%m%d%H%M%S'
    if msec:
        timestr = '%Y%m%d%H%M%S%f'
    return datetime.now().strftime(timestr)


def logging_setup(logger_name, log_folder_path, test_abbreviation='', stream_short=False):
    os.makedirs(log_folder_path, exist_ok=True)
    if test_abbreviation:
        test_abbreviation = f"_{safe_str(test_abbreviation)}"

    logger_name = safe_str(logger_name)
    user, host = safe_str(getuser()), safe_str(gethostname())
    long_format_logging = [
        '{asctime:} ' + f"{user}@{host} " + ' {name}\t' +
        '{filename}:{lineno}\t{funcName:8.8}\t{levelname:8.8}\t{process:>5}: {message}'
    ][0]
    short_format_logging = '{asctime:} {levelname:8.8}: {message}'
    stream_format = long_format_logging
    stream_level = 'DEBUG'

    if stream_short:
        stream_format = short_format_logging
        stream_level = 'INFO'

    log_config = {
        'version': 1,
        'disable_existing_loggers': True,
        'formatters': {
            'long': {
                'format': long_format_logging,
                'style': '{',
            },
            'stream': {
                'format': stream_format,
                'datefmt': "%H:%M:%S",
                'style': '{',
            },
        },
        'handlers': {
            'stream': {
                'level': stream_level,
                'formatter': 'stream',
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stdout',  # Default is stderr
            },
            'file': {
                'level': 'DEBUG',
                'class': 'logging.handlers.TimedRotatingFileHandler',
                'formatter': 'long',
                'filename': os.path.join(
                    log_folder_path, f"{host}_{logger_name}{test_abbreviation}.log"
                ),
                'when': 'midnight',
                'interval': 1,
                'backupCount': 50
            },
            'file_INFO': {
                'level': 'INFO',
                'class': 'logging.handlers.TimedRotatingFileHandler',
                'formatter': 'long',
                'filename': os.path.join(log_folder_path, f"{host}_info.log"),
                'when': 'midnight',
                'interval': 1,
                'backupCount': 50
            },
            'file_WARN': {
                'level': 'WARN',
                'class': 'logging.handlers.TimedRotatingFileHandler',
                'formatter': 'long',
                'filename': os.path.join(log_folder_path, f"{host}_WARN.log"),
                'when': 'midnight',
                'interval': 1,
                'backupCount': 50
            },
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['stream', 'file', 'file_INFO', 'file_WARN'],
                'level': 'DEBUG',
                'propagate': False,
            },
            'electric_eye.__main__': {
                'handlers': ['stream', 'file', 'file_INFO', 'file_WARN'],
                'level': 'DEBUG',
                'propagate': False,
            },
            'EE': {
                'handlers': ['stream', 'file', 'file_INFO', 'file_WARN'],
                'level': 'DEBUG',
                'propagate': True,
            },
            '__main__': {  # if __name__ == '__main__'
                'handlers': ['stream', 'file', 'file_INFO', 'file_WARN'],
                'level': 'DEBUG',
                'propagate': False,
            },
        },
    }
    if logger_name not in log_config['loggers']:
        log_config['loggers'].update(
            {
                f"{logger_name}": {
                    'handlers': ['stream', 'file', 'file_INFO', 'file_WARN'],
                    'level': 'DEBUG',
                    'propagate': True,
                },
            },
        )
        pass
    logging.config.dictConfig(log_config)
