#!/usr/bin/python
#coding=utf8

class Person:
    '''Represents a person'''
    population = 0

    def __init__(self, name):
        '''Initialize the person's data'''
        self.name = name
        print 'Initializing %s' % self.name
        Person.population += 1

    def __del__(self):
        '''I am dying'''
        print '%s say bye' % self.name
        Person.population -= 1
        if Person.population == 0:
            print 'I am the last person'
        else:
            print 'There are still %d people left' % Person.population

    def sayHi(self):
        ''' Greeting by the person

        Really, that's all it does'''
        print 'Hi, my name is %s' % self.name

    def howMany(self):
        '''Prints the current population'''
        if Person.population == 1:
            print 'I am the only person here'
        else:
            print 'We have %d persons here' % Person.population

if __name__ == '__main__':
    p1 = Person("p1")
    p1.sayHi()
    p1.howMany()

    p2 = Person("p2")
    p2.sayHi()
    p2.howMany()

    p1.sayHi()
    p1.howMany()
