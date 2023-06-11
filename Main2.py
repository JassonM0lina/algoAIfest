# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 12:21:46 2021

@author: jasso
"""
import numpy as np
import ga
from time import process_time
import matplotlib.pyplot as plt 
import csv
import json
import pandas as pd
from collections import OrderedDict
import math

dataset = pd.read_csv("players_22.csv", dtype='unicode')

""" input variables for optimization decision"""
managment = dataset[['short_name', 'player_positions', 'overall', 'potential', 'value_eur', 'wage_eur', 'mentality_composure']] 
postion_required = OrderedDict({
    'RW': 1,
    'ST':1,
    'CF': 1,
    'ST':1,
    'LW':1,
    'CAM':1,
    'CM':1,
    'GK':1
})

no_variables = len(list(postion_required.values())) # each variable is a position
""" variables for genetic algoritm (GA) """
pop_size = 100 
crossover_rate = 50
mutation_rate = 50
no_generations = 80
step_size = 5
rate = 10

""" restriccions for optimization decision"""
max_budget = 3000000 # Budget in dollars

dict_managment = managment.to_dict('records')

dict_statistics = {}
total_position = {pos:[] for pos in postion_required.keys()}
for statistics_candidates in dict_managment:
    posible_positions = [position.strip() for position in statistics_candidates['player_positions'].split(',')]
    for current_position in posible_positions:
        if current_position in postion_required:
            
            prob = {
                'overall': statistics_candidates['overall'],
                'potential': statistics_candidates['potential'],
                'value_eur': statistics_candidates['value_eur'],
                'wage_eur': statistics_candidates['wage_eur'],
                'mentality_composure': statistics_candidates['mentality_composure'],
            }
            noneList = []
            for stat in list(prob.values()):
                if math.isnan(float(stat)): 
                    noneList.append(1)
            if len(noneList) >0 : continue
            total_position[current_position].append(statistics_candidates['short_name'])
            dict_statistics[statistics_candidates['short_name']] = {
                'overall': statistics_candidates['overall'],
                'potential': statistics_candidates['potential'],
                'value_eur': statistics_candidates['value_eur'],
                'wage_eur': statistics_candidates['wage_eur'],
                'mentality_composure': statistics_candidates['mentality_composure'],
            }

lower_bounds = []
upper_bounds = []
for ordered_position in postion_required.keys():
    upper_bounds.append(len(total_position[ordered_position]))
    lower_bounds.append(0)

pop = np.zeros((pop_size,no_variables))
for s in range(pop_size):
    for h in range(no_variables):
        pop[s,h] = np.random.uniform(lower_bounds[h],upper_bounds[h])
        
extended_pop = np.zeros((pop_size+crossover_rate+mutation_rate+2*no_variables*rate,pop.shape[1]))       

A = []
a = 5
g = 0
global_best = pop
k= 0
while g<= no_generations:
    for i in range(no_generations):
        offspring1 = ga.crossover(pop,crossover_rate)
        offspring2 = ga.mutation(pop, mutation_rate)
        fitness = ga.objective_function(pop, total_position, dict_statistics, postion_required, max_budget)
        offspring3 = ga.local_search(pop,fitness, lower_bounds,upper_bounds,step_size,rate)
        step_size = step_size*0.98
        if step_size < 1:
            step_size = 1
        extended_pop[0:pop_size]=pop
        extended_pop[pop_size:pop_size+crossover_rate] =  offspring1
        extended_pop[pop_size+crossover_rate:pop_size+crossover_rate+mutation_rate] = offspring2
        extended_pop[pop_size+crossover_rate+mutation_rate:pop_size+crossover_rate+mutation_rate+2*no_variables*rate] = offspring3
        fitness = ga.objective_function(extended_pop, total_position, dict_statistics, postion_required, max_budget)
        pop = ga.selection(extended_pop, fitness, pop_size)
        rt = np.argmax(fitness)
        print(extended_pop[rt])
        print("Generations: ", g, ", Current fitness value: ", 10e6-max(fitness))
        A.append(10e6 - max(fitness))
        g += 1
        if i>=a:
            if sum(abs(np.diff(A[g-a:g]))) <=0.05:
                index =np.argmax(fitness)
                current_best = extended_pop[index]
                pop = np.zeros((pop_size, no_variables))
                for s in range(pop_size):
                    for h in range(no_variables):
                        pop[s, h] = np.random.uniform(lower_bounds[h], upper_bounds[h])
                step_size = 5
                global_best[k] = current_best
                k +=1
                break
            if g > no_generations:
                break
        if g > no_generations:
            break
fitness = ga.objective_function(global_best, total_position, dict_statistics, postion_required, max_budget)
index = np.argmax(fitness)

total_candidates = []
selection_final = global_best[index]
order_position = list(postion_required.keys())
for pos, candidate in enumerate(selection_final):
    candidate = math.floor(candidate)
    position = order_position[pos]
    nameCandidate = total_position[position][candidate]
    total_candidates.append(nameCandidate)

print("Best solution= ", global_best[index])
print("Best fitness value= ", 10e6-max(fitness))
plt.show()

final_result = {}
for i in range(len(order_position)):
    final_result.setdefault(order_position[i], []).append(total_candidates[i])

print("---> Final Result <----")
print(final_result)

fig = plt.figure()
ax = fig.add_subplot()
fig.show()
plt.title('Evolutionary process of the objective function value')
plt.xlabel('iteration')
plt.ylabel('objetive function value')
plt.plot(A, '*', markersize=2, color='red')
plt.show()



