#!/usr/bin/env python
#coding=utf8

'''
Created on 2016年3月10日

@author: daiguozhou
'''

import string
import textwrap


def getText():
    return 'The quick brown fox jumped over the lazy dog.'


def capitalizeWords():
    text = getText()
    # capitalize each word
    print(' '.join([word.capitalize() for word in text.split()]))
    # capitalize each word using string.capwords(s, sep=None)
    print(string.capwords(text))


def simpleTranslate():
    text = getText()
    leet = string.maketrans('abegiloprstz', '463611092572')
    print(text.translate(leet))


def stringFormat():
    s = 'cstyle format: the price of %s is %d' % ("apple", 5)
    print(s)
    values = {"productName": 'apple', "price": 5}
    s = 'format with dict: the price of %(productName)s is %(price)d' % values
    print(s)
    t = string.Template(
        "format with string.Template: the price of ${productName} is $price")
    print(t.substitute(values))


def textWrapExample():
    text = '''\
        The textwrap module can be used to format text for output in
        situations where pretty-printing is desired. It offers
        programmatic functionality similar to the paragraph wrapping
        or filling features found in many text editors.
        '''
    dedentedText = textwrap.dedent(text)
    print('dedented:')
    print(dedentedText)
    print(
        'textwrap.fill(dedentedText, width=50, initial_indent=" "*2, subsequent_indent=""):')
    print(textwrap.fill(
        dedentedText, width=80, initial_indent=" " * 2, subsequent_indent=""))


def codecExample():
    print("str represents a byte stream")
    print("str.decode('codec') -> unicode")
    print("unicode.encode('codec') -> str")
    # s是str类型，str是字节流，字节流有编码，这里s的编码和文件头指定的编码一样
    s = "中华人民共和国"
    # str.decode('target-coding')将某种编码的字节流转换成unicode, 这里u是unicode类型
    u = s.decode("utf-8")
    # unicode.encode('target-coding')讲unicode转换为指定编码的字节流，这里gb是str类型
    gb = u.encode("gb2312")

if __name__ == '__main__':
    codecExample()
    capitalizeWords()
    simpleTranslate()
    stringFormat()
    textWrapExample()
