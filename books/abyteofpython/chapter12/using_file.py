#!/usr/bin/env python
#coding=utf8

poem = '''\
Programming is fun
When the work is done
if you wanna make your work also fun:
    use Python!
'''

with open('poem.txt', 'w') as f:
    f.write(poem)

with open('poem.txt') as f:
    while True:
        line = f.readline()
        if len(line) == 0:
            break
        print(line)
