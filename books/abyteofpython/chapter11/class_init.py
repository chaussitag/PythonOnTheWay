#!/usr/bin/python
#coding=utf8

class Person:
    def __init__(self, name, nick = ""):
        self.name = name
        self.nick = nick

    def info(self):
        print('name %s, nick %s' % (self.name, self.nick))

if __name__ == "__main__":
    p = Person("NicolosLee")
    p.info()
    p = Person("NicolosLee", "Nick")
    p.info()
    print(Person.__dict__)
    print(p.__dict__)
