#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 09:53:18 2018

@author: pritish.jadhav
"""
import os
import sys
import re

import pandas as pd
import numpy as np
import pulp

curr_dir = os.path.dirname(os.path.realpath(__file__))
max_players = 11
max_batsman = 5
max_allrounders = 3
max_bowlers = 5
max_keepers = 1
max_cost = 100
max_team1_players = 7
max_team2_players = 7

data = pd.DataFrame({
                     'player_id': ['dhoni', 'raina', 'kishan', 
                                    'rohit_sharma', 'lewis', 
                                    'rayadu', 'surya', 'watson',
                                    'hardik', 'krunal', 'bravo',
                                    'bumrah', 'rahman', 'markande', 
                                    'thakur', 'chahar'],
                    'points': [25,30,20, 
                               15, 20, 
                               45, 30, 35, 
                               25, 32, 27, 
                               18, 10, 10, 
                               16, 15],
                    'is_team1': [1,1,0,0,0,1,0,1,0,0,1,0,0,0,1,1],
                    'is_team2': [0,0,1,1,1,0,1,0,1,1,0,1,1,1,0,0],
                    'is_keeper': [1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    'is_batsman':[0,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0],
                    'is_allrounder':[0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0],
                    'is_bowler':[0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1],
                    'cost':[9,10,8.5,10.5,9.5,9,8.5,10.5,9,9,9,9,8.5,8.5,8.5,8]
                     })

data.reset_index(inplace = True)
prob = pulp.LpProblem('Dreamteam', pulp.LpMaximize)


decision_variables = []
for rownum, row in data.iterrows():
    # variable = set('x' + str(rownum))
    variable = str('x' + str(row['index']))
    variable = pulp.LpVariable(str(variable), lowBound = 0, upBound = 1, cat = 'Integer') # make variable binary
    decision_variables.append(variable)
    
print('Total number of decision variables: ' + str(len(decision_variables)))

# Create optimization Function
total_points = ''
for rownum, row in data.iterrows():
    for i, p_select in enumerate(decision_variables):
        if rownum == i:
            formula = row['points'] * p_select
            total_points += formula
            
prob += total_points

#set constrainst for keeper
total_keepers = ''
for rownum, row in data.iterrows():
    for i,  p_select in enumerate(decision_variables):
        if rownum == i:
            formula = row['is_keeper']*p_select
            total_keepers += formula
            
prob += (total_keepers == max_keepers)

total_batsman = ''
for rownum, row in data.iterrows():
    for i,  p_select in enumerate(decision_variables):
        if rownum == i:
            formula = row['is_batsman']*p_select
            total_batsman += formula
            
prob += (total_batsman <= max_batsman)

total_allrounder = ''
for rownum, row in data.iterrows():
    for i,  p_select in enumerate(decision_variables):
        if rownum == i:
            formula = row['is_allrounder']*p_select
            total_allrounder += formula
            
prob += (total_allrounder <= max_allrounders)


total_bowler = ''
for rownum, row in data.iterrows():
    for i,  p_select in enumerate(decision_variables):
        if rownum == i:
            formula = row['is_bowler']*p_select
            total_bowler += formula
            
prob += (total_bowler <= max_bowlers)

total_players = ''
for rownum, row in data.iterrows():
    for i,  p_select in enumerate(decision_variables):
        if rownum == i:
            formula = p_select
            total_players += formula
            
prob += (total_players == max_players)

total_cost = ''
for rownum, row in data.iterrows():
    for i,  p_select in enumerate(decision_variables):
        if rownum == i:
            formula = row['cost']*p_select
            total_cost += formula
            
prob += (total_cost <= max_cost)

total_team1 = ''
for rownum, row in data.iterrows():
    for i,  p_select in enumerate(decision_variables):
        if rownum == i:
            formula = row['is_team1']*p_select
            total_team1 += formula
            
prob += (total_team1 <= max_team1_players)

total_team2 = ''
for rownum, row in data.iterrows():
    for i,  p_select in enumerate(decision_variables):
        if rownum == i:
            formula = row['is_team2']*p_select
            total_team2 += formula
            
prob += (total_team2 <= max_team2_players)

print(prob)
prob.writeLP('Dreamteam.lp')

optimization_result = prob.solve()

assert optimization_result == pulp.LpStatusOptimal
print('Status:', LpStatus[prob.status])
print('Optimal Solution to the problem: ', value(prob.objective))
print('Individual decision variables: ')
for v in prob.variables():
    print(v.name, '=', v.varValue)
    
# reorder results
variable_name = []
variable_value = []

for v in prob.variables():
    variable_name.append(v.name)
    variable_value.append(v.varValue)
    
df = pd.DataFrame({'index': variable_name, 'value': variable_value})
for rownum, row in df.iterrows():
    value = re.findall(r'(\d+)', row['index'])
    df.loc[rownum, 'index'] = int(value[0])
    
df = df.sort_values(by = 'index')
result = pd.merge(data, df, on = 'index')
result = result[result['value'] == 1].sort_values(by = 'points', ascending = False)
selected_cols_final = ['player_id', 'is_team1', 'is_team2', 'points']
final_set_of_talks_to_watch = result[selected_cols_final]

