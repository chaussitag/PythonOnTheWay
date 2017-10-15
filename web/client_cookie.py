#!/usr/bin/env python
#coding=utf8

'''
Created on 2015年10月20日

@author: daiguozhou
'''

import cookielib
import urllib2

if __name__ == "__main__":
    cj = cookielib.CookieJar()
    opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj))
    ## open some sites to recieve cookies
    sites = ["http://www.douban.com", "http://www.baidu.com"]
    for site in sites:
        response = opener.open(site)
    ## print the cookies
    print("cookies:")
    for cookie in cj:
        print("%s:%s" % (cookie.name, cookie.value))