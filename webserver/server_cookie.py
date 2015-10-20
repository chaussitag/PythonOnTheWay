#!/usr/bin/env python
#coding=utf8

'''
Created on 2015年10月20日

@author: daiguozhou
'''

import Cookie

if __name__ == "__main__":
    cookie = Cookie.SimpleCookie()
    cookie['session-id'] = '123456789'
    cookie['session-id']['domain'] = '.github.com'
    cookie['session-id']['path'] = '/'
    cookie['acookie-key'] = 'acookie-值'

    print("iterate over the cookie object:")
    for cookie_key, cookie_value in cookie.iteritems():
        print(cookie_key)
        print("    value : %s" % cookie_value.value)
        print("    coded_value : %s" % cookie_value.coded_value)

    print("")
    print("<cookie.output()>")
    print(cookie.output())

    print("")
    print("<cookie.output(header='Cookie:')>")
    print(cookie.output(header='Cookie:'))

    print("")
    print("<cookie.output(attrs=['domain', 'path', 'version', 'expires', 'secure'])>")
    print(cookie.output(attrs=['domain', 'path', 'version', 'expires', 'secure']))

    print("")
    print("<cookie.js_output()>")
    print(cookie.js_output())