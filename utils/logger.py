# -*- coding: utf-8 -*-
# @Time    : 2023/2/17
# @Author  : Xianda Zheng
# @File    : logger.py
# @Software: PyCharm

import logging


def get_logger(level='info', log_file=None):
    '''
    the logger module for printing
    use default parameters

    use case:
    init:
        logger = logger = get_logger()

    for error message:
        logger.error('error test')
        # expect output : [yyyy-mm-dd hh:mm:ss,ms] [ERROR] ERROR test

    for normal information (use as print):
        logger.info('the info')
        # expect output : [yyyy-mm-dd hh:mm:ss,ms] [INFO] info test
    '''

    head = '[%(asctime)-15s] [%(levelname)s] %(message)s'
    if level == 'info':
        logging.basicConfig(level=logging.INFO, format=head)
    elif level == 'debug':
        logging.basicConfig(level=logging.DEBUG, format=head)
    new_logger = logging.getLogger()
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        new_logger.addHandler(fh)
    return new_logger


if __name__ == '__main__':
    # test unit
    logger = get_logger()
    logger.info('info test')
    logger.error('error test')
