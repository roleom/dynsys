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
from scipy import integrate

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
        
        
class Lv3SummingTest(unittest.TestCase):    
        
    def test_finite_sum(self):
        print('---test_finite_sum---')
        t = array([0, 1, 2, 2.1, 2.5])
        nx = 2
        x0 = arange(nx)
        def tmph(x, t=0):
            return ones(shape(x))
        xsum = lv.finite_sum(tmph, x0, t)
        xint = integrate.odeint(tmph, x0, t)
        print(xsum)
        # for function returning a fixed value, sum and integral should be the same:
        self.assertListEqual(xsum.tolist(), xint.tolist())
        t = array([0.4])
        nx = 2
        x0 = arange(0., nx, 1.0)  
        xsum = lv.finite_sum(tmph, x0, t)
        # if only one time value is passed, the output should consist of initial value in array:
        self.assertListEqual(xsum[0].tolist(), x0.tolist())
        
        
class Lv3SimpleSystemTest(unittest.TestCase):
    
    def test_build_system(self):
        print('---test_build_system---')
        # name pop mass hunger birth death sp-th, lp-th, lp-decay, lp-floor, death-decay, death-ceil
        A1 = Species('A1', 100, 10, 1, 2, 1, 0, 400, 2, 0.1)
        B1 = Species('B1', 10, 100, 10, 1.2, 0.4, 0, 2000, 2, 0)
        n_A1 = Nutrition_Pool(A1, [], [], 300)
        n_B1 = Nutrition_Pool(B1, [A1], [0.1], 0)
        s = System([A1, B1])
        self.assertEqual(s.Species[0].name, 'A1')
        self.assertEqual(s.Species[1].pop, 10)
        self.assertEqual(s.Species[0].PreyPool.Pred.name, 'A1')
        self.assertEqual(s.Species[0].PredPools[0].Prey[0].name, 'A1')
        self.assertEqual(s.Species[0].PreyPool.nu_encountered_autotroph, 300)
        self.assertEqual(s.Species[0].PredPools[0].encounter_probability[0], 0.1)
        self.assertEqual(s.Species[1].PreyPool.Pred.name, 'B1')
        self.assertEqual(s.Species[1].PreyPool.Prey[0].name, 'A1')
    

class Lv3SimpleSystemShowcase:
    
    def _build_system(self):        
        # name pop mass hunger birth death sp-th, lp-th, lp-decay, lp-floor, death-decay, death-ceil
        A1 = Species('A1', 100, 10, 1, 2, 1, 0, 400, 2, 0.1)
        B1 = Species('B1', 10, 100, 10, 1.2, 0.4) #, 0, 2000, 2, 0)
        n_A1 = Nutrition_Pool(A1, [], [], 300)
        n_B1 = Nutrition_Pool(B1, [A1], [0.1], 0)
        return System([A1, B1])
    
    def _plot_system(self, x, t, names):
        xt = x.T
        f1 = p.figure()
        for i in range(xt.shape[0]):
            p.plot(t, xt[i], label='{}: {}'.format(names[i], ceil(xt[i,-1])))
        p.grid()
        p.yscale('log')
    #    p.ylim([1e-1, 1e3])
        p.legend(loc='best')

    def show_summation_plot(self):
        print('---show_summation_plot---')
        t = linspace(0, 50, 10)
        s = self._build_system()
        delta_X = lv.build_derivative(s.get_pop(), s, 0)
        xsum = lv.finite_sum(delta_X, s.get_pop(), t)       
        self._plot_system(xsum, t, s.get_names())
        s = self._build_system()
        delta_X = lv.build_derivative(s.get_pop(), s, 0)
        xint = integrate.odeint(delta_X, s.get_pop(), t)        
        self._plot_system(xint, t, s.get_names())
        
        
    
#%%
        
if __name__ == '__main__':
    unittest.main()
    
    
#LFS = Lv3FunctionShowcase()
#LFS.run()
    
SC = Lv3SimpleSystemShowcase()
s = SC.show_summation_plot()