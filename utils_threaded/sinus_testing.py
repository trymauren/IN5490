# Project
# from config import ea_config
# from utils_threaded import unity_stuff

# Python modules
import numpy as np
from tqdm import tqdm
import time
import os
import multiprocessing

# def evaluate_population(population, unity_interface) -> None:
# 	"""
# 	Splits the population in groups, passing them to unity for evaluation.
# 		Arg: population (the whole)
# 		Ret: list containing tuples containing fitness of an individual
# 				why tuple? https://deap.readthedocs.io/en/master/overview.html
# 	"""
# 	num_agents = unity_interface.get_agents()
# 	fitnesses = [0]*len(population)
# 	for i in range(0, len(population), num_agents):
# 		subset = population[i:i+num_agents]
# 		fitnesses[i:i+num_agents] = evaluate_group(subset, unity_interface)
# 	return fitnesses

# def evaluate_group(group, unity_interface=None, repetitions=ea_config['num_mov_repeat'], sim_best=True) -> tuple:
# 	"""
# 	Evaluates the fitness of an individuals genes. Used by DEAP.
# 		Arg: individual
# 		Ret: list containing tuples containing fitness of an individual
# 	"""
# 	all_movements = []
# 	for i in range(len(group)):
# 		movement = compute_movement(group[i])
# 		movement_rep = repeat_movement(movement, repetitions)
# 		all_movements.append(movement_rep)

# 	if unity_interface != None:
# 		coordinates = unity_interface.send_actions_to_unity(all_movements)
# 	else:
# 		coordinates = unity_stuff.evaluate_in_unity(all_movements)
# 	fitness = []
# 	for j in range(len(coordinates)):
# 		if ea_config['fitness_one_axis']:
# 			fit = max(coordinates[j][2], 0) # rewards walking far in one direction
# 		else:
# 			fit = np.sqrt(coordinates[j][0]**2 + coordinates[j][2]**2) # rewards walking far

# 		fitness.append((fit,)) # must add tuple to list! why: https://deap.readthedocs.io/en/master/overview.html
# 	return fitness 
# #[[array([ 3.6873782, -0.7602028,  0.4305477], dtype=float32)], [array([ 3.8278027 , -0.57440406,  1.8399124], dtype=float32)], ]


def compute_movement(individual, num_move_directions=12) -> np.array(np.array):
	"""
	Fetches a sinusoid for each limb and returns a 2d list
	containing sinusoid for all limbs walking directions.
		Arg: an individual (the individual must inherit from python list).
		Ret: np array containing movements for all moving directions of all limbs. 
	"""
	individual = np.array(individual)
	# if ea_config['equal_frequency_all_limbs']:
	if False:
		freq = individual[-1]
		limbs = list(np.array_split(individual[:-1],num_move_directions))
		movement = [0]*len(limbs)
		for i in range(len(limbs)):
			sinusoid = compute_sin(*(limbs[i]), freq)
			movement[i] = sinusoid
		return np.asarray(movement)

	else:
		limbs = list(np.array_split(individual,num_move_directions))
		movement = [0]*len(limbs)
		for i in range(len(limbs)):
			sinusoid = compute_sin(*(limbs[i]))
			movement[i] = sinusoid
		return np.asarray(movement)

def repeat_movement(movement, repetitions, num_move_directions=12, num_movements=10):
	"""Repeat the movement from a given list [1,2,3]*2 --> [1,2,3,1,2,3] 
	Args:
		movement (np.array): movement to repeat
		repetitions (int): num rep
		num_move_directions (int, optional): _description_. Defaults to 12.
		num_movements (int, optional): _description_. Defaults to 10.
	Returns:
		_type_: _description_
	"""
	movement_rep = np.zeros((num_move_directions,num_movements*repetitions))
	for i in range(num_move_directions):
		movement_rep[i] = np.tile(movement[i], repetitions)	
	return movement_rep

def compute_sin(A=1, phase=0, f=2) -> np.array:
	"""
	Computes a sinusoid for the given parameters.
		Arg: A: amplitude, f: frequency, phase: phase
		Ret: sinusoid-signal
	"""
	# t = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], dtype=float)
	t = np.linspace(0,2*np.pi*f,500)
	sinusoid = A*np.sin(t + phase)
	plt.plot(sinusoid)
	return sinusoid

def simulate_best(individual: np.array, repetitions : int, unity_interface) -> None:
	"""Simulate a singe individ with one crwaler 

	Args:
		individual (np.array): individual for the crawler (one individ from a population) 
		repetitions (int): how many steps to do/ how many repititions of the sampled list
		unity_interface (unity): unity interface 
	Returns:
		None 
	"""
	evaluate_group(individual, unity_interface, repetitions)

import matplotlib.pyplot as plt

ind = [1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1]

sinus = compute_sin(1,0,1)
# sinus = np.tile(sinus,10)
plt.plot(sinus)
plt.show()
exit()

ret = compute_movement(ind)
repeated_ret = repeat_movement(ret,10)

plt.plot(repeated_ret[0])
# plt.xlim(0,10)
plt.show()