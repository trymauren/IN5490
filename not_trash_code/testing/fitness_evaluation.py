def evaluate_individual(individual) -> tuple:
	"""
	Evaluates the fitness of an individuals genes. Used by DEAP.
		Arg: individual
		Ret: --
	"""
	sum_ = 0
	for i in individual:
		sum_ += i
	return (sum_,) # must return tuple!

