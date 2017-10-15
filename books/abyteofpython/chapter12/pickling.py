#!/usr/bin/env python
#coding=utf8

import cPickle as p
# import pickle as p

shopListFile = "shoplist.data"

shoplist = ['apple', 'mango', 'carrot']

with open(shopListFile, 'w') as f:
    p.dump(shoplist, f)

del shoplist

with open(shopListFile) as f:
    restoredShopList = p.load(f)
    print(restoredShopList)
