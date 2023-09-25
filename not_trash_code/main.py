# our stuff
from testing import algorithm
from testing import fitness_evaluation
# deap stuff
from deap import base
from deap import tools
from deap import creator

import random
# general stuff


# test-main
def main():
	algorithm.train()

main()

# def main():
# 	creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# 	creator.create("Individual", list, fitness=creator.FitnessMin)

# 	toolbox = base.Toolbox()
# 	toolbox.register("evaluate", benchmarks.rastrigin)

#     numpy.random.seed(128)

#     strategy = cma.Strategy(centroid=[5.0]*N, sigma=5.0, lambda_=20*N)
#     toolbox.register("generate", strategy.generate, creator.Individual)
#     toolbox.register("update", strategy.update)

#     hof = tools.HallOfFame(1)
#     stats = tools.Statistics(lambda ind: ind.fitness.values)
#     stats.register("avg", numpy.mean)
#     stats.register("std", numpy.std)
#     stats.register("min", numpy.min)
#     stats.register("max", numpy.max)
#     # The CMA-ES algorithm converge with good probability with those settings
# main()