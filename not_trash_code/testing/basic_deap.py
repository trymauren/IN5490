from deap import base
from deap import tools
from deap import creator
import random
import fitness_evaluation
import unity_interface

toolbox = base.Toolbox()

def train():
    IND_SIZE = 36  # size of individuals (size of genome)
    NGEN = 100  # number of generations. Move to config-file later
    POP_SIZE = 50
    CXPB = 0.1
    MUTPB = 0.1
    #staring env
    # env = unity_interface.start_env(executable_file="C:/Users/oyo12/3D Objects/exe_filer/UnityEnvironment.exe")

    creator.create('FitnessMin', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMin)

    toolbox.register('attr_float', random.random)
    toolbox.register('individual', tools.initRepeat, creator.Individual,
                     toolbox.attr_float, n=IND_SIZE)

    toolbox.register('mate', tools.cxTwoPoint)
    toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register('select', tools.selTournament, tournsize=3)
    toolbox.register('evaluate_population', fitness_evaluation.evaluate_population)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    population = toolbox.population(n=POP_SIZE)

    for g in range(NGEN):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        # fitnesses = toolbox.map(toolbox.evaluate, [env, invalid_ind])
        # for ind, fit in zip(invalid_ind, fitnesses):
        #     ind.fitness.values = fit

        # for ind, fit in zip(invalid_ind, fitnesses):
        #     ind.fitness.values = fit

        env = None
        toolbox.evaluate_population(invalid_ind, env)
        population[:] = offspring

    # print(population)

    print(population)
    #unity_interface.stop_env(env)

train()
