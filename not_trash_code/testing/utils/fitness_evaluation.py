import numpy as np
import time

def evaluate_population(population, unity_interface) -> None:
	# return ([0]*len(population),) # testmode
	fitnesses = [0]*len(population)
	for i in range(len(population)):
		fitnesses[i] = evaluate_individual(population[i], unity_interface)

	return fitnesses

def evaluate_individual(individual, unity_interface, verbose=False) -> tuple:
	"""
	Evaluates the fitness of an individuals genes. Used by DEAP.
		Arg: individual
	"""
	movement = compute_movement(individual)
	movement_rep = repeat_movment(movement, 5)
	coordinates = unity_interface.send_actions_to_unity([movement_rep])
	fitness = np.sqrt(coordinates[0][1][0][2]**2 + coordinates[0][1][0][0]**2)
	
	if verbose: print("---------------fitness----------")
	print(fitness)

	return (fitness,) # must return tuple!

def compute_movement(individual) -> np.array(np.array):
	"""
	Fetches a sinusoid for each limb and returns a 2d list
	containing sinusoid for all limbs.
		Arg: an individual (the individual must inherit from python list).
	"""
	limbs = list(np.array_split(individual,12))
	movement = [0]*len(limbs)
	for i in range(len(limbs)):
		sinusoid = compute_sin(*(limbs[i]))
		movement[i] = sinusoid

	return np.asarray(movement)

def repeat_movment(movement, repetitions):

	movement_rep = np.zeros((12,10*repetitions))
	for i in range(12):
		movement_rep[i] = np.tile(movement[i], repetitions)	

	return movement_rep

def compute_sin(A=1, f=2, phase=0) -> np.array:
	"""
	Computes a sinusoid for the given parameters.
		Arg: A: amplitude, f: frequency, phase: -.-
	"""
	t = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], dtype=float)
	sinusoid = A*np.sin(t * f + phase)
	return sinusoid