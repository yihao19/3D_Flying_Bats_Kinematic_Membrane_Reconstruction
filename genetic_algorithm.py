# -*- coding: utf-8 -*-
"""
Created on Sun May 11 07:56:43 2025

@author: yihao
"""
import numpy as np
import random
# Initialize Population
def initialize_population(pop_size):
    population = []
    for _ in range(pop_size):
        individual = np.random.uniform(low = 0.00001, high = 0.1)  # Random weights for linear model
        population.append(individual)
    return np.array(population)

# Crossover (average)
def crossover(parent1, parent2):
    offspring= (parent1 + parent2) / 2
    return offspring

# Mutation (Gaussian Mutation)
def mutate(individual, mutation_rate=0.30):
    if random.random() < mutation_rate:
        individual += np.random.normal(0, 0.01)  # Add small random noise
        if(individual < 0): 
            individual = 0.00001
        elif(individual > 1): 
            individual = 0.1
    return individual
