from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
import random
from utils import fitness_evaluation
from utils import functions
from config import ea_config
import numpy as np
# http://deap.gel.ulaval.ca/doc/default/overview.html

def train(unity_interface, runs_path, verbose=True):
    print(' -- Strategy: basic ea')
    lambda_ = ea_config['pop_size']
    N = ea_config['genome_len']
    lower_start_limit = ea_config['lower_start_limit']
    upper_start_limit = ea_config['upper_start_limit']
    np.random.seed(ea_config['seed']+r)

    def signal_handler(signal, frame):
        functions.dump_data(logbooks, halloffame, runs_path)
        exit()

    import signal
    signal.signal(signal.SIGINT, signal_handler)

    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attribute", np.random.uniform, lower_start_limit, upper_start_limit)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=N)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=lambda_)
    toolbox.register('evaluate', fitness_evaluation.evaluate_population)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    def amplitude(ind):
        return np.mean(np.array([ind[i] for i in range(0,len(ind),3)]))
    def frequency(ind):
        return np.abs(np.mean(np.array([ind[i] for i in range(1,len(ind),3)])))
    def phase_shift(ind):
        return np.mean(np.array([ind[i] for i in range(2,len(ind),3)]))

    stats_fitness       = tools.Statistics(lambda ind: ind.fitness.values)
    stats_amplitude     = tools.Statistics(amplitude)
    stats_frequency     = tools.Statistics(frequency)
    stats_phase_shift   = tools.Statistics(phase_shift)
    stats = tools.MultiStatistics(  fitness=stats_fitness, amplitude=stats_amplitude, 
                                    frequency=stats_frequency, phase_shift=stats_phase_shift)
    stats.register('avg', lambda x: np.mean(x))
    stats.register('std', lambda x: np.std(x))
    stats.register('min', lambda x: np.min(x))
    stats.register('max', lambda x: np.max(x))

    halloffame = tools.HallOfFame(1)
    logbooks = list()

    CXPB, MUTPB = 0.5, 0.2
    
    for r in range(ea_config['num_restarts']):
        pop = toolbox.population(n=ea_config['pop_size'])
        np.random.seed(ea_config['seed']+r)
        print(np.random.seed(ea_config['seed']+r))
        logbooks.append(tools.Logbook())
        logbooks[-1].header = 'gen', 'run', 'pop_size', 'fitness', 'amplitude', 'frequency', 'phase_shift'
        logbooks[-1].chapters['fitness'].header = 'std', 'min', 'avg', 'max'
        logbooks[-1].chapters['amplitude'].header = 'std', 'min', 'avg', 'max'
        logbooks[-1].chapters['frequency'].header = 'std', 'min', 'avg', 'max'
        logbooks[-1].chapters['phase_shift'].header = 'std', 'min', 'avg', 'max'

        for g in range(ea_config['num_generations']):
            
            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))

            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the entire population
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

            fitnesses = toolbox.evaluate(invalid_ind, unity_interface) # very nice
            # Assign the computed fitness to individuals
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            # The population is entirely replaced by the offspring
            pop[:] = offspring
            record = stats.compile(pop)
            logbooks[-1].record(gen=g, run=r, pop_size=len(pop), **record)
            if verbose:
                print(logbooks[-1].stream)
            halloffame.update(pop)

    return logbooks, halloffame
