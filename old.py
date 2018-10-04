#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 23:43:26 2018

@author: rlm
"""
from numpy import *
import pylab as p
import logging
from scipy import integrate

import loggersetup
import os

from lv3 import System

#%%
print(os.path.splitext(os.path.basename(__file__))[0])
logger = loggersetup.getLogger(__name__, logging.DEBUG,
                               os.path.splitext(os.path.basename(__file__))[0] + '.log')

def build_derivative(X, System, t=0):
    def tmp(X, t=0):
        return System.set_pop_and_calc_growth(X, t)
    return tmp

def get_system(ind):
    if ind==0:
        logger.debug('building system 0')
        # name pop mass hunger birth death sp-th, lp-th, lp-decay, lp-floor, death-decay, death-ceil
        A1 = Species('A1', 100, 10, 1, 2, 1, 0, 400, 2, 0.1)
        B1 = Species('B1', 10, 100, 10, 1.2, 0.4, 0, 2000, 2, 0)
        n_A1 = Nutrition_Pool(A1, [], [], 300)
        n_B1 = Nutrition_Pool(B1, [A1], [0.1], 0)
        s = System([A1, B1])
        return s

#%%
if False:
    x = linspace(0,50,10000)
    f = p.figure()
    #p.plot(x, birthrate_small_pop_penalty(x, 2)*birthrate_large_pop_penalty(x, 15, 3, 0.5))
    p.plot(x, deathrate_malnutrition_penalty(2, x, 10, 3, 4))
    p.plot(x, birthrate_malnutrition_penalty(2, x, 10))

    p.plot(x, - deathrate_malnutrition_penalty(2, x, 10, 3, 4) + 3*birthrate_malnutrition_penalty(2, x, 10))
    #p.plot(x, birthrate(x, 2, 500, 1, 30, 400, 400/5, 0.5))
    p.grid()

    #x = linspace(1,100)
    #p.plot(x, kills(5, x, 0.1, 2))

if False:
    reset_data()

    s = get_system(0)
    t_end = 50
    dt = 1
    t = linspace(0, t_end, dt*t_end)
    X0 = s.get_pop()

    dX_dt = build_derivative(X0, s)
    X, infodict = integrate.odeint(dX_dt, X0, t, full_output=True)
    disp(infodict['message'])

    Xt = X.T
    f1 = p.figure()
    for i in range(Xt.shape[0]):
        p.plot(t, Xt[i], label=s.get_names()[i])
    p.grid()
#    p.yscale('log')
#    p.ylim([1e-1, 1e3])
    p.legend(loc='best')


if True:

    def get_and_log_stat():
        growth = s.growth()
        pop = s.get_pop()
        logger.debug('pop: %s', str(pop))
        logger.debug('growth: %s', str(growth))
        return pop, growth

    s = get_system(0)
    pop, growth = get_and_log_stat()

#%%
    logger.debug('iterate...')
    s.set_pop(pop + growth)
    pop, growth = get_and_log_stat()
    logger.debug('end')
