import numpy as np

def evaluate_individual(individual) -> tuple:
	"""
	Evaluates the fitness of an individuals genes. Used by DEAP.
		Arg: individual
		Ret: --
	"""
	
	limbs = list(np.array_split(individual,12))
	for i in limbs:
		actions = sin_func(*i)
	fitness = evaluate(unity)
	return (fitness,) # must return tuple!

def sin_func(A, f, phase):
	# Create an array of linearly spaced values for x
    x_values = np.linspace(0, 2 * np.pi, 12)

    # Compute the sine values
    sin_values = (A*np.sin(f * x_values + phase)+A)/(2*A)
    continuous_action = sin_values.reshape(1, 12).astype(np.float32)
    return continuous_action
