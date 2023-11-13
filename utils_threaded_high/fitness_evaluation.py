# Project
from config import ea_config
from utils_threaded_high import unity_stuff

# Python modules
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import os
import multiprocessing

def evaluate_population(population, unity_interface) -> None:
	"""
	Splits the population in groups, passing them to unity for evaluation.
		Arg: population (the whole)
		Ret: list containing tuples containing fitness of an individual
				why tuple? https://deap.readthedocs.io/en/master/overview.html
	"""
	num_agents = unity_interface.get_agents()
	fitnesses = [0]*len(population)
	for i in range(0, len(population), num_agents):
		subset = population[i:i+num_agents]
		fitnesses[i:i+num_agents] = evaluate_group(subset, unity_interface)
	return fitnesses

def evaluate_group(group, unity_interface=None) -> tuple:
	"""
	Evaluates the fitness of an individuals genes. Used by DEAP.
		Arg: individual
		Ret: list containing tuples containing fitness of an individual
	"""
	all_movements = []
	for i in range(len(group)):
		movement = compute_movement(group[i])
		all_movements.append(movement)

	if unity_interface != None:
		coordinates = unity_interface.send_actions_to_unity(all_movements)
	else:
		coordinates = unity_stuff.evaluate_in_unity(all_movements)
	fitness = []
	for j in range(len(coordinates)):
		if ea_config['fitness_one_axis']:
			fit = max(coordinates[j][2], 0) # rewards walking far in one direction
		else:
			fit = np.sqrt(coordinates[j][0]**2 + coordinates[j][2]**2) # rewards walking far

		fitness.append(fit) # must add tuple to list! why: https://deap.readthedocs.io/en/master/overview.html

	fitnesses_as_tuple = []

	for i in range(len(fitness)):
		penalty_accumulated = 0
		for entry in group[i]: # for entry in individual
			if entry < 0:
				penalty_accumulated += entry
		fitnesses_as_tuple.append((fitness[i] + penalty_accumulated,))
	return fitnesses_as_tuple
#[[array([ 3.6873782, -0.7602028,  0.4305477], dtype=float32)], [array([ 3.8278027 , -0.57440406,  1.8399124], dtype=float32)], ]


def compute_movement(individual, num_move_directions=12) -> np.array(np.array):
	"""
	Fetches a sinusoid for each limb and returns a 2d list
	containing sinusoid for all limbs walking directions.
		Arg: an individual (the individual must inherit from python list).
		Ret: np array containing movements for all moving directions of all limbs. 
	"""
	individual = np.array(individual)
	if ea_config['equal_frequency_all_limbs']:
		freq = individual[-1]
		limbs = list(np.array_split(individual[:-1],num_move_directions))
		movement = [0]*len(limbs)
		for i in range(len(limbs)):
			sinusoid = my_sin(*(limbs[i]), f=freq)
			movement[i] = sinusoid
		return np.asarray(movement)

	else:
		limbs = list(np.array_split(individual,num_move_directions))
		movement = [0]*len(limbs)
		for i in range(len(limbs)):
			sinusoid = my_sin(*(limbs[i]))
			movement[i] = sinusoid
		return np.asarray(movement)


def my_sin(A, phase, f):
    t = np.linspace(0, 4, 2000, endpoint=False) #periode på 500 - det vi vil ha
    return A*np.sin(2*np.pi*f*t+phase)[:500]


def simulate_best(group: np.array, unity_interface) -> None:
	"""Simulate a singe individ with one crwaler 

	Args:
		individual (np.array): individual for the crawler (one individ from a population) 
		repetitions (int): how many steps to do/ how many repititions of the sampled list
		unity_interface (unity): unity interface 
	Returns:
		None 
	"""
	all_movements = []
	for i in range(len(group)):
		movement = compute_best_movement(group[i][0])
		all_movements.append(movement)

	coordinates = unity_interface.send_actions_to_unity(all_movements)


def compute_best_movement(individual, num_move_directions=12):

	individual = np.array(individual)
	if ea_config['equal_frequency_all_limbs']:
		freq = individual[-1]
		limbs = list(np.array_split(individual[:-1],num_move_directions))
		movement = [0]*len(limbs)
		for i in range(len(limbs)):
			sinusoid = my_sin(*(limbs[i]), freq)
			movement[i] = sinusoid
		return np.asarray(movement)
	else:
		limbs = list(np.array_split(individual,num_move_directions))
		movement = [0]*len(limbs)
		for i in range(len(limbs)):
			sinusoid = my_sin(*(limbs[i]))
			movement[i] = sinusoid
		return np.asarray(movement)
