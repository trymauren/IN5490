import numpy as np
import unity_interface
import time

def evaluate_population(population, env) -> None:
	# return ([0]*len(population),) # testmode
	fitnesses = [0]*len(population)
	for i in range(len(population)):
		#unity_interface.reset_env(env)
		var = evaluate_individual(population[i], env)
		fitnesses[i] = var[0]
		best_movment = var[1]

	wait = input("Traning done, hit enter when redy for best result")
	env_use = unity_interface.start_env(executable_file="C:/Users/oyo12/3D Objects/exe_filer/UnityEnvironment.exe", graphics=False)
	print(best_movment)
	coordinates = unity_interface.send_actions_to_unity(env_use, [best_movment])

	return fitnesses

def evaluate_individual(individual, env) -> tuple:
	"""
	Evaluates the fitness of an individuals genes. Used by DEAP.
		Arg: individual
	"""
	best_fitness = 0  
	best_movment = 0
	movement = compute_movement(individual)
	env_use = unity_interface.start_env(executable_file="C:/Users/oyo12/3D Objects/exe_filer/UnityEnvironment.exe", graphics=True)
	# print(np.shape(movement))
	movement_rep = repeat_movment(movement, 5)
	# print(np.shape(movement_rep))
	coordinates = unity_interface.send_actions_to_unity(env_use, [movement_rep])
	fitness = np.sqrt(coordinates[0][1][0][2]**2 + coordinates[0][1][0][0]**2)
	print("---------------fitness----------")
	print(fitness)

	if fitness > best_fitness:
			best_fitness = fitness
			best_movment = movement_rep

	return (fitness,), best_movment # must return tuple!

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


# mat =  [1,1,0.25, 1,1,0.25, 1,1,0.25, 1,1,0.25,
# 		1,1,0.5, 1,1,0.5, 1,1,0.5, 1,1,0.5,
# 		2,1,0.5, 2,1,0.5, 2,1,0.5, 2,1,0.5]
# print(evaluate_population([mat,mat], test_mode=True))