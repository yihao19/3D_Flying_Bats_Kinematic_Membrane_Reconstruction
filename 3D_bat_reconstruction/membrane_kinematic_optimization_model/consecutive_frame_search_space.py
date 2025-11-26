# -*- coding: utf-8 -*-
"""
Created on Sun May 11 23:06:30 2025

@author: yihao
"""
from skopt.space import Real
searching_space_angular = [
    Real(0.00001,0.1, name='tension_stiffness'),
    #Real(0.0001, 10, name='compression_stiffness'),
    #Real(0.0001, 10, name='bend_stiffness'),
    #Real(0, 50, name='tension_damping'),
    #Real(0, 50, name='compression_damping'),
    #Real(0, 50, name='bend_damping')
    ]

searching_space_linear = [
    Real(0.0001, 0.015, name='structual_stiffness')
    ]
    
