from deap import base
from deap import tools
from deap import creator
import random
import fitness_evaluation
toolbox = base.Toolbox()

def train():
	IND_SIZE = 4 # size of individuals (size of genome)
	NGEN = 100 # number of generations. Move to config-file later
	POP_SIZE = 50
	CXPB = 0.1
	MUTPB = 0.1
	creator.create('FitnessMin', base.Fitness, weights=(1.0,))
	# this means creating a new base class Individual that ...
	# ... behave like a python list (inherits from list) 
	creator.create('Individual', list, fitness=creator.FitnessMin)

	# register all stages of the EA in DEAPs toolbox
	toolbox = base.Toolbox()
	toolbox.register('attr_float', random.random)
	toolbox.register('individual', tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=IND_SIZE)
	

	toolbox.register('mate', tools.cxTwoPoint)
	toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
	toolbox.register('select', tools.selTournament, tournsize=3)
	toolbox.register('evaluate', fitness_evaluation.evaluate_individual)
	toolbox.register('population', tools.initRepeat, list, toolbox.individual)
	population = toolbox.population(n=POP_SIZE)

	# evolving
	for g in range(NGEN):
		# Select the next generation individuals
	    offspring = toolbox.select(population, len(population))
	    # Clone the selected individuals
	    offspring = list(map(toolbox.clone, offspring))

	    # Apply crossover on the offspring
	    for child1, child2 in zip(offspring[::2], offspring[1::2]):
	        if random.random() < CXPB:
	            toolbox.mate(child1, child2)
	            del child1.fitness.values
	            del child2.fitness.values

	    # Apply mutation on the offspring
	    for mutant in offspring:
	        if random.random() < MUTPB:
	            toolbox.mutate(mutant)
	            del mutant.fitness.values

	    # Evaluate the individuals with an invalid fitness
	    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
	    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
	    for ind, fit in zip(invalid_ind, fitnesses):
	        ind.fitness.values = fit

	    # The population is entirely replaced by the offspring
	    population[:] = offspring
	print(population)


train()
