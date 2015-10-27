#!/usr/bin/env python
#coding=utf8

'''
Created on 2015年10月27日

@author: daiguozhou
'''

class StaticMethod(object):
    '''A non-data descriptor used to define a "static method" in a class.

    '''
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        return self.func

    def __set__(self, name, value):
        raise AttributeError("read-only descriptor, __set__ not supported!!")

class SomeClass(object):
    def aStaticFunc():
        print("this is a static function")
    aStaticFunc = StaticMethod(aStaticFunc)

if __name__ == '__main__':
    SomeClass.aStaticFunc()
    obj = SomeClass()
    obj.aStaticFunc()