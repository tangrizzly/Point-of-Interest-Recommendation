#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:       Qiang Cui:  <cuiqiang1990[at]hotmail.com>
# Descripton:   
# Create Date:  2016-00-00 00:00:00
# Modify Date:  2016-00-00 00:00:00
# Modify Disp:

import time


def exe_time(func):
    def new_func(*args, **args2):
        t0 = time.time()
        print("-- @%s, {%s} start" % (time.strftime("%X", time.localtime()), func.__name__))
        back = func(*args, **args2)
        print("-- @%s, {%s} end" % (time.strftime("%X", time.localtime()), func.__name__))
        print("-- @%.3fs taken for {%s}" % (time.time() - t0, func.__name__))
        return back

    return new_func


def fun():
    pass


@exe_time  # 放到待调用函数的定义的上一行
def main():
    pass


if '__main__' == __name__:
    main()
