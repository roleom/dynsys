#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 23:43:26 2018

@author: rlm
"""

from numpy import *
import pylab as p
import unittest
#import logging
#from scipy import integrate

#import loggersetup
#import os

import lv3 as lv


class Lv3FunctionShowcase:
    
    def __init__(self):
        pass
    
    def run(self):
        self.show_logistic()
        self.show_exp_decay()
        self.show_tanh_decay()
    
    def show_logistic(self):
        x = linspace(0, 5, 100)
        k = array([[1], [2.2], [3]])
        y = lv.logistic(x, k=k)
        p.figure()
        for i in range(size(y, 0)):
            p.plot(x, y[i,], label='k={}'.format(k[i, 0]))
        p.legend(loc='best')
        p.title('logistic growth')
        
    def show_exp_decay(self):
        x = linspace(0, 1, 100)
        d = array([[1.2], [4.5], [6]])
        y1 = lv.exp_decay(x, 5, d)
        y2 = lv.exp_decay(x, d, 3)
        p.figure()
        for i in range(size(y1, 0)):
            p.plot(x, y1[i,], label='start=5, decay={}'.format(d[i, 0]), linestyle=':')
        for i in range(size(y2, 0)):
            p.plot(x, y2[i,], label='decay=3, start={}'.format(d[i, 0]))
        p.legend(loc='best')
        p.title('exp decay')
        
    def show_tanh_decay(self):
        x = linspace(0, 1, 100)
        t = array([[0.5], [0.8]])
        d = array([[0.2], [0.5]])
        f = array([[0.], [0.2]])
        y1 = lv.tanh_decay(x, t, 0.3, 0)
        y2 = lv.tanh_decay(x, 0.5, d, 0)
        y3 = lv.tanh_decay(x, 0.5, 0.1, f)
        p.figure()
        for i in range(size(y1, 0)):
            p.plot(x, y1[i,], label='t={}, d=0.3, f=0'.format(t[i, 0]), linestyle=':')
        for i in range(size(y2, 0)):
            p.plot(x, y2[i,], label='t=0.5, d={}, f=0'.format(d[i, 0]), linestyle='--')
        for i in range(size(y3, 0)):
            p.plot(x, y3[i,], label='t=0.5, d=0.1 f={}'.format(f[i, 0]), linestyle='-')
        p.legend(loc='best')
        p.title('tanh decay')
        
        
class Lv3FunctionsTest(unittest.TestCase):
    
    def test_birthrate(self):
        p = linspace(0, 50, 100)
        t = 20
        d = 20
        f = 0.1
        self.assertListEqual(
                list(lv.birthrate_large_pop_penalty(p, t, d, f)),
                list(lv.tanh_decay(p, t, d, f)))
        
        
        
#%%
        
if __name__ == '__main__':
    unittest.main()
    
    
LFS = Lv3FunctionShowcase()
LFS.run()