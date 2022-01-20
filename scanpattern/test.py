# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 06:35:24 2022

@author: sstucker
"""

class Foo:
    
    def __init__(self, *args, **kwargs):
        
        print('len(args)', len(args))
        print('len(kwargs)', len(kwargs))
        
        if len(args) > 0:
            
            self.bar(*args, **kwargs)
    
        
    def bar(self, arg1, arg2, arg3, kwarg1=None, kwarg2=None):
        
        print('Function called:')
        print(arg1)
        print(arg2)
        print(arg3)
        print(kwarg1, kwarg2)


f = Foo(1, 2, 3, kwarg1=4)