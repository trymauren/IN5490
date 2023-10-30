import numpy as np
import time
from tqdm import tqdm

def evaluate_population(population, unity_interface) -> None:
	"""
	Splits the population in groups, passing them to unity for evaluation.
		Arg: population (the whole)
		Ret: list containing tuples containing fitness of an individual
				why tuple? https://deap.readthedocs.io/en/master/overview.html
	"""
	num_agents = unity_interface.get_agents()
	fitnesses = [0]*len(population)
	counter = 0
	for i in range(0, len(population), num_agents):
		if len(population)-counter*num_agents >= num_agents:
			counter += 1
			subset = population[i:i+num_agents] 
			fitnesses[i:i+num_agents] = evaluate_group(subset, unity_interface)
		else:
			subset = population[i:i+len(population)-(num_agents)]
			fitnesses[i:i+len(population)-num_agents] = evaluate_group(subset, unity_interface)


	return fitnesses

def evaluate_group(group, unity_interface, repetitions = 20) -> tuple:
	"""
	Evaluates the fitness of an individuals genes. Used by DEAP.
		Arg: individual
		Ret: list containing tuples containing fitness of an individual
	"""
	all_movmets = []
	for i in range(len(group)):
		movement = compute_movement(group[i])
		movement_rep = repeat_movment(movement, repetitions) # 5 should be switched to config file?
		all_movmets.append(movement_rep)

	coordinates = unity_interface.send_actions_to_unity(all_movmets)

	fitness = []
	for j in range(len(coordinates)):
		fit = max(coordinates[j][2], 0)
		# fit = np.sqrt(coordinates[j][0]**2 + coordinates[j][2]**2)
		fitness.append((fit,)) # must add tuple to list! why: https://deap.readthedocs.io/en/master/overview.html

	return fitness 
#[[array([ 3.6873782, -0.7602028,  0.4305477], dtype=float32)], [array([ 3.8278027 , -0.57440406,  1.8399124], dtype=float32)], ]


def compute_movement(individual, num_move_directions=12) -> np.array(np.array):
	"""
	Fetches a sinusoid for each limb and returns a 2d list
	containing sinusoid for all limbs walking directions.
		Arg: an individual (the individual must inherit from python list).
		Ret: np array containing movements for all moving directions of all limbs. 
	"""
	limbs = list(np.array_split(individual,num_move_directions))
	movement = [0]*len(limbs)
	for i in range(len(limbs)):
		sinusoid = compute_sin(*(limbs[i]))
		movement[i] = sinusoid

	return np.asarray(movement)

def repeat_movment(movement, repetitions, num_move_directions=12, num_movements=10):
	"""Repeat the movment from a given list [1,2,3]*2 --> [1,2,3,1,2,3] 
	Args:
		movement (np.array): movment to repeat
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

def compute_sin(A=1, f=2, phase=0) -> np.array:
	"""
	Computes a sinusoid for the given parameters.
		Arg: A: amplitude, f: frequency, phase: phase
		Ret: sinusoid-signal
	"""
	t = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], dtype=float)
	sinusoid = A*np.sin(t * f + phase)
	return sinusoid

def simulate_best( movment: np.array, repetitions : int, unity_interface) -> None:
	"""Simulate a singe individ with one crwaler 

	Args:
		movment (np.array): movment for the crawler (one individ from a population) 
		repetitions (int): how many steps to do/ how many repititions of the sampled list
		unity_interface (unity): unity interface 
	Returns:
		None 
	"""
	evaluate_group([movment], unity_interface ,repetitions)