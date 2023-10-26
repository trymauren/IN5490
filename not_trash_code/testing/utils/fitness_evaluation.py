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
	fitnesses = [0]*len(population)
	for i in range(0, len(population), 30):
		subset = population[i:i+30] # fix this slice
		fitnesses[i:i+30] = evaluate_group(subset, unity_interface)

	return fitnesses

def evaluate_group(group, unity_interface, verbose=False) -> tuple:
	"""
	Evaluates the fitness of an individuals genes. Used by DEAP.
		Arg: individual
		Ret: list containing tuples containing fitness of an individual
	"""
	all_movmets = []
	for i in range(len(group)):
		movement = compute_movement(group[i])
		movement_rep = repeat_movment(movement, 20) # 5 should be switched to config file?
		all_movmets.append(movement_rep)

	coordinates = unity_interface.send_actions_to_unity(all_movmets)

	fitness = []
	for j in range(len(coordinates)):
		fit = np.sqrt(coordinates[j][1][0]**2 + coordinates[j][1][2]**2)
		# fit = np.sqrt(coordinates[0][1][0][2]**2 + coordinates[0][1][0][0]**2)
		fitness.append((fit,)) # must add tuple to list! why: https://deap.readthedocs.io/en/master/overview.html
	
	if verbose:
		print("---------------fitness----------")

	return fitness 
# [array([1.3351440e-05, 1.3962054e-01, 1.1444092e-05], dtype=float32), start 
#  array([ 0.2965622 , -0.13715017,  0.31038666], dtype=float32)]       slutt


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
	"""
	ØYYYYYSTEIN dokumenteeeeeer 
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