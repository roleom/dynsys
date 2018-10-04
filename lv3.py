#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 12:48:24 2018

@author: rlm
TODO Tests schreiben
TODO kommentieren
TODO iterativen Löser schreiben (nicht-infinitesimal)
TODO which effects in small and large population are effects of DENSITY rather than POPULATION?
"""
from numpy import *
import pylab as p
import logging
from scipy import integrate

#%%
def logistic(x, L=1, k=1, x0=0):
    return L / (1 + exp(-k * (x-x0) ) )

def exp_decay(x, start, decay):    
    res = (exp(-decay*x+decay)-1) / (exp(decay)-1) * (start-1) + 1
    return res

def tanh_decay(x, threshold, decay, floor):
    res = 0.5 * (1+floor - (1-floor)*tanh(1/decay*(x - threshold)))
    return res

#%%
def encounters(pop_pred, pop_prey, probability):
    res = pop_prey * pop_pred * probability
    return res

def nu_encountered(pop_prey, prey_nu, encounters):
    res = prey_nu * max(encounters, pop_prey)
    return res

def nu_used(nu_needed, nu_encountered):
    res = min(nu_needed, nu_encountered)
    return res

def nu_used_fraction(nu_encountered, nu_used):
    res = nu_used / nu_encountered
    return res

def satiety(nu_needed, nu_used):
    if nu_needed == 0:
        res = 1.
    else:
        res = nu_used / nu_needed
    return res

def birthrate_small_pop_penalty(pop, threshold):
    '''
    penalty factor on a birthrate for small populations: inbreeding depression, allee effect
    @param pop: size of population
    @param threshold: size of population at which penalty starts (more or less)
    @return factor between 0 and 1
    '''
    if threshold == 0:
        res = ones(shape(pop))
    else:
        res = tanh(2 * pop / threshold) # TODO extract constant
    return res


def birthrate_large_pop_penalty(pop, threshold, decay, floor):
    '''
    penalty factor on a birthrate for large populations: intraspecial competition, hormonal
    response causing lower birth rate...
    @param pop: size of population
    @param threshold: pop. size at which penalty has faded in by 50%
    @param decay: width (in terms of pop. indiv.) of birthrate fall-off
    @param floor: birthrate factor which serves as a lower boundary for fall-off, 0<=floor<1
    @return factor between 0 and 1
    '''
    if threshold == inf:
        res = ones(shape(pop))
    else:
        res = tanh_decay(pop, threshold, decay, floor)
    return res


def birthrate_low_satiety_penalty(satiety, decay=10):
    res = logistic(satiety, 1, decay, 0.5)
    return res


def birthrate(birthrate_max, 
              satiety, satiety_decay, 
              pop, smallpop_threshold, largepop_threshold, largepop_decay, largepop_floor):
    '''
    birthrate for a population under given circumstances of capacity and nu
    @return birthrate; 0 <= birthrate <= birthrate_max
    '''
    res = birthrate_max * \
            birthrate_low_satiety_penalty(satiety, satiety_decay) * \
            birthrate_small_pop_penalty(pop, smallpop_threshold) * \
            birthrate_large_pop_penalty(pop, largepop_threshold, largepop_decay, largepop_floor)
    return res


def deathrate_low_satiety_penalty(satiety, decay=3, ceil=5):
    res = exp_decay(satiety, ceil, decay)
    return res


def deathrate(deathrate_min, satiety, decay, ceil):
    res = deathrate_min * deathrate_low_satiety_penalty(satiety, decay, ceil)
    return res

#%%
class Nutrition_Pool:
    '''
    ein Räuber
    0 oder mehr Beute-Species
    autotroph kann auf inf gesetzt werden
    '''
    def __init__(self, Pred, Prey, encounter_probability, nu_encountered_autotroph=0):
        #
        self.Pred = Pred
        self.Prey = Prey # list
        self.encounter_probability = encounter_probability # list
        self.nu_encountered_autotroph = nu_encountered_autotroph
        #
        Pred.set_PreyPool(self)
        for i in range(len(Prey)):
            Prey[i].add_PredPool(self)

    def nu_encountered(self):
        tmp = 0
        for i in range(len(self.Prey)):
            tmp += nu_encountered(
                    self.Prey[i].pop, self.Prey[i].nu,
                    encounters(self.Pred.pop, self.Prey[i].pop, self.encounter_probability[i]))
        tmp = tmp + self.nu_encountered_autotroph
        return tmp

    def nu_used(self):
        res = nu_used(self.Pred.nu_needed, self.nu_encountered())
#        logger.debug('np preyed by %s, nu_used: %f, fraction: %f, satiety: %f',
#                     self.Pred.name, res, res/self.nu_encountered(), self.satiety())
        return res

    def nu_used_fraction(self):
        res = nu_used_fraction(self.nu_encountered(), self.nu_used())
        return res

    def satiety(self):
        res = satiety(self.Pred.nu_needed, self.nu_used())
        return res


#%%
class Species:

    def __init__(
            self, name, pop, nu, hunger, 
            birthrate_max, deathrate_min, 
            smallpop_threshold=0, largepop_threshold=inf, largepop_decay=0.5, largepop_floor=0.5,
            deathrate_satiety_decay=3., deathrate_satiety_ceil=5.):
        #
        self.PredPools = [] # list of NutritionPool in which this species acts as prey
        self.PreyPool = None # NutritionPool in which this species acts as predator
        #
        self.pop = pop
        #
        self.name = name
        self.nu = nu
        self.hunger = hunger # set to 0 to remove malnutrition penalty (rem. only for autotrophic)
        #
        self.birthrate_max = birthrate_max
        self.deathrate_min = deathrate_min
        self.smallpop_threshold = smallpop_threshold # set to 0 to remove smallpop penalty
        self.largepop_threshold = largepop_threshold # set to inf to remove largepop penalty
        self.largepop_decay = largepop_decay * largepop_threshold # TODO das ist unschön
        self.largepop_floor = largepop_floor
        self.deathrate_satiety_decay = deathrate_satiety_decay
        self.deathrate_satiety_ceil = deathrate_satiety_ceil

    def set_PreyPool(self, PreyPool):
        ''' PROTECTED '''
        self.PreyPool = PreyPool

    def add_PredPool(self, PredPool):
        ''' PROTECTED '''
        self.PredPools.append(PredPool)

    def set_pop(self, pop):
        self.pop = pop
    
    def satiety(self):
        res = self.PreyPool.satiety()
        return res 

    def birthrate(self):
        res = birthrate(
                self.birthrate_max, 
                self.satiety(), self.satiety_decay, 
                self.pop, self.smallpop_threshold, self.largepop_threshold, self.largepop_decay,
                self.largepop_floor)
        return res

    def deathrate(self):
        res = deathrate(
                self.deathrate_min, 
                self.satiety(), self.deathrate_satiety_decay, self.deathrate_satiety_ceil)
        return res

    def death_as_prey(self):
        tmp = 0
        for i in range(len(self.PredPools)):
            tmp = tmp + self.pop * self.PredPools[i].nu_used_fraction()
            # this can lead to errors if a prey species is preyed by multiplie predators (i.e. has 
            # more than one object in PredPools). nu_encountered and nu_used_fraction are calcula-
            # ted and capped per predator. That means that in some cases more prey might be preyed
            # than actually present. This will not be corrected for predators. However, in order to
            # rule out death numbers greater than pop, let's test:
        if tmp > self.pop: tmp = self.pop
        return tmp

    def growth(self):
        res = (self.birthrate() - self.deathrate()) * self.pop - self.death_as_prey()
#        logger.debug('growth of %s: %f', self.name, res)
#        logger.debug('dap of %s: %f', self.name, self.death_as_prey())
        return res


#%%
class System:

    def __init__(self, Species=[], Pools=[]):
        self.Species = Species # list
        self.Pools = Pools # list

    def add_Species(self, Species):
        self.Species.append(Species)

    def add_Pool(self, Pool):
        self.Pools.append(Pool)

    def set_pop(self, pop):
        for i in range(len(self.Species)):
            self.Species[i].set_pop(pop[i])

    def get_pop(self):
        tmp = []
        for i in range(len(self.Species)):
            tmp.append(self.Species[i].pop)
        return tmp

    def get_names(self):
        tmp = []
        for i in range(len(self.Species)):
            tmp.append(self.Species[i].name)
        return tmp

    def growth(self):
        tmp = []
        for i in range(len(self.Species)):
            tmp.append(self.Species[i].growth())
        return array(tmp)

    def set_pop_and_calc_growth(self, pop, t=0):
#        add_datum(t)
        self.set_pop(pop)
        return self.growth()

#%%

data = []
def reset_data():
    data = []

def add_datum(datum):
    data.append(datum)

