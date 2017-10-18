#!/usr/bin/env python
#coding=utf8

import time

f = open('poem.txt')
try:
    while True:
        line = f.readline()
        if len(line) == 0:
            break
        time.sleep(2)
        print(line)
except Exception as e:
    print('exception occurred: ' + str(e))
finally:
    f.close()
    print('Cleaning up...closed the file')
