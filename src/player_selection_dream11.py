#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 09:53:18 2018

@author: pritish.jadhav
"""
import os
from typing import List
import logging

import pandas as pd
import numpy as np
from ast import literal_eval
import pulp as p

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger()

MANDATORY_COLUMNS = {"player_name", "player_category", "team_name", "last_5_matches_points", "cost"}

# Dream11 constraints
MAX_PLAYERS = 11
MAX_BATSMEN = 5
MAX_ALLROUNDERS = 3
MAX_BOWLERS = 5
MAX_KEEPERS = 2
MAX_COST = 100
MAX_PLAYER_IN_EACH_TEAM = 7
HUMANFRIENDLY_COLUMNS = ["player_name", "weighted_player_points", "is_selected", "cost","last_5_matches_points"]

def read_data(abs_filename: str) -> pd.DataFrame:
    data = pd.read_csv(abs_filename, converters={"last_5_matches_points": literal_eval})
    assert set(data.columns) == MANDATORY_COLUMNS

    return data

def compute_weighted_points(points_list: List, alpha: int = 0.20):
    weights = np.exp(list(reversed(np.array(range(1, len(points_list)+1))*alpha * -1)))
    exponential_weighted_average = np.average(np.array(points_list), weights = weights)
    return exponential_weighted_average

def _get_decision_variables(all_data: pd.DataFrame) -> pd.DataFrame:
    # define decision variables for each row in the input dataframe

    decision_variables = []

    for rownum, row in all_data.iterrows():
        variable = str('x_{}'.format(str(rownum)))
        variable = p.LpVariable(variable, lowBound=0, upBound=1, cat='Integer')
        decision_variables.append({"player_name": row["player_name"], "pulp_variable": variable})

    return pd.DataFrame(decision_variables)

def _get_optimization_function(player_df: pd.DataFrame) -> p.LpProblem:
    # Create optimization Function
    prob = p.LpProblem('Dreamteam', p.LpMaximize)

    total_points = ''
    for index, row in player_df.iterrows():
        formula = row['weighted_player_points'] * row["pulp_variable"]
        total_points += formula
    prob += total_points
    return prob

def _add_constraints(player_df: pd.DataFrame, optimization_prob: p.LpProblem):
    total_keepers = ''
    total_batsman = ''
    total_allrounder = ''
    total_bowler = ''
    total_players = ''
    total_cost = ''
    total_team2 = ''
    total_team1 = ''

    for rownum, row in player_df.iterrows():
        keeper_formula = row['player_category_wicket_keeper'] * row["pulp_variable"]
        total_keepers += keeper_formula

        batsman_formula = row['player_category_batsman'] * row["pulp_variable"]
        total_batsman += batsman_formula

        allrounder_formula = row['player_category_all_rounder'] *row["pulp_variable"]
        total_allrounder += allrounder_formula

        bowler_formula = row['player_category_bowler'] * row["pulp_variable"]
        total_bowler += bowler_formula

        total_players_formula = row["pulp_variable"]
        total_players += total_players_formula

        total_cost_formula = row['cost'] * row["pulp_variable"]
        total_cost += total_cost_formula

        formula = row['team_name_CSK'] * row["pulp_variable"]
        total_team1 += formula

        formula = row['team_name_MI'] * row["pulp_variable"]
        total_team2 += formula

    optimization_prob += (total_keepers <= MAX_KEEPERS)
    optimization_prob += (total_batsman <= MAX_BATSMEN)
    optimization_prob += (total_allrounder <= MAX_ALLROUNDERS)
    optimization_prob += (total_bowler <= MAX_BOWLERS)
    optimization_prob += (total_players == MAX_PLAYERS)
    optimization_prob += (total_cost <= MAX_COST)
    optimization_prob += (total_team1 <= MAX_PLAYER_IN_EACH_TEAM)
    optimization_prob += (total_team2 <= MAX_PLAYER_IN_EACH_TEAM)

    print(optimization_prob)
    optimization_prob.writeLP('Dreamteam.lp')
    return optimization_prob

def caller(base_dir: str= os.path.join(".", "data")) -> pd.DataFrame:
    raw_data = read_data(os.path.join(base_dir, "dream11_performance_data.csv"))

    processed_player_data = pd.get_dummies(raw_data, columns=["player_category", "team_name"])
    processed_player_data['weighted_player_points'] = processed_player_data['last_5_matches_points'].apply(
        compute_weighted_points)

    decision_variables_df = _get_decision_variables(processed_player_data)
    assert len(decision_variables_df) == len(processed_player_data), f"Number of Decision Variables must be equal to the" \
                                                                  f"number of rows in the input file. Expected {len(processed_player_data)}" \
                                                                  f"Received {len(decision_variables_df)}"

    merged_processed_players_df = pd.merge(processed_player_data,decision_variables_df, on = "player_name")
    merged_processed_players_df["pulp_variable_name"] = merged_processed_players_df["pulp_variable"].apply(lambda x: x.name)

    optimization_prob = _get_optimization_function(merged_processed_players_df)
    optimization_prob = _add_constraints(merged_processed_players_df, optimization_prob)

    optimization_result = optimization_prob.solve()

    assert optimization_result != p.LpStatusNotSolved

    solution_df = pd.DataFrame([{"pulp_variable_name": v.name,
                                 "is_selected": v.varValue} for v in optimization_prob.variables()])
    optimized_players_df = pd.merge(merged_processed_players_df, solution_df, on = "pulp_variable_name")

    dream_team_df = optimized_players_df.loc[optimized_players_df.is_selected == 1, HUMANFRIENDLY_COLUMNS]

    assert len(dream_team_df) == MAX_PLAYERS, f"there should be {MAX_PLAYERS} in the team but there are " \
                                              f"{len(dream_team_df)} players. Something went wrong."
    logger.debug(f"This Team can earn you an estimated of {dream_team_df['weighted_player_points'].sum()}")
    logger.debug(dream_team_df)
    return dream_team_df

if __name__ == "__main__":
    caller()