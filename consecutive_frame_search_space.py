# -*- coding: utf-8 -*-
"""
Created on Sun May 11 23:06:30 2025

@author: yihao
"""
from skopt.space import Real, Integer
low_bound = -0.1824533
up_bound = 0.184533

position_adj_low_bound = -0.1
position_adj_up_bound = 0.1


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
    
