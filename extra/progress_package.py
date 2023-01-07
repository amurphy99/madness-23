#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 03:15:22 2021

@author: andrewmurphy
"""

def progress_bar (current, total, bar_size=50):
    
    bar_size     = bar_size
    step         = total / bar_size
    progress     = int(current // step)
    percentage   = round(current/total*100, 2) 
    filled       = "#" * progress
    empty        = " " * (bar_size-progress)
    
    clear_line   = " " *  (2 * bar_size)
    bar          = "[{}{}] - {}%".format( filled, empty, percentage)
        
    
    print(clear_line, end="\r")
    if progress != bar_size:
        print(bar, end="\r")
    else:
        print(bar)
    

#for i in range(200):
#    progress_bar(i+1, 200)