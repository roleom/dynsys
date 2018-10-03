#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 22:25:58 2018

@author: rlm
"""

import logging
import os

def getLogger(name, level, outfile):

    logger = logging.getLogger(name)
    closeLogger(logger)
    logger.setLevel(level)
    
    # create a file handlerimport os
    if os.path.exists(outfile):
      os.remove(outfile)
    handler = logging.FileHandler(outfile)
    handler.setLevel(level)
    
    # create a logging format
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # add the handlers to the logger
    logger.addHandler(handler)
    
    logger.info('Logger started')
    
    return logger

def closeLogger(logger):
    handlers = logger.handlers
    for i in range(len(handlers)-1,-1,-1):
        handlers[i].close
        logger.removeHandler(handlers[i])