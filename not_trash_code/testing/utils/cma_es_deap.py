#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.
# 
#    Implementation of the BI-Population CMA-ES algorithm. As presented in
#    *Hansen, 2009, Benchmarking a BI-Population CMA-ES on the BBOB-2009 Function
#    Testbed* with the exception of the modifications to the original CMA-ES
#    parameters mentioned at the end of section 2's first paragraph.



# See documentation:
#   https://deap.readthedocs.io/en/master/api/algo.html#module-deap.cma
#
#   Statistics:
#   https://deap.readthedocs.io/en/master/tutorials/basic/part3.html



from utils import fitness_evaluation
from collections import deque

import pickle
import time
import numpy as np

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import cma
from deap import creator
from deap import tools
from config import ea_config
# Problem size (genome length)



def train(unity_interface, verbose=True):

    lambda_ = ea_config['pop_size']
    sigma = ea_config['std_dev']
    N = ea_config['genome_len']

    creator.create('FitnessMin', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMin)

    np.random.seed(128)

    toolbox = base.Toolbox()

    # Register custom fitness evaluation function
    toolbox.register('evaluate', fitness_evaluation.evaluate_population)

    # Create the hall of fame (the container storing the best individuals)
    halloffame = tools.HallOfFame(1)


    # Create the statistics functions
    def amplitude(ind):
        return np.mean(np.array([ind[i] for i in range(0,len(ind),3)]))
    def frequency(ind):
        return np.mean(np.array([ind[i] for i in range(1,len(ind),3)]))
    def phase_shift(ind):
        return np.mean(np.array([ind[i] for i in range(2,len(ind),3)]))

    stats_fitness       = tools.Statistics(lambda ind: ind.fitness.values)
    stats_amplitude     = tools.Statistics(amplitude)
    stats_frequency     = tools.Statistics(frequency)
    stats_phase_shift   = tools.Statistics(phase_shift)
    
    # Register and create statistics functions
    stats = tools.MultiStatistics(  fitness=stats_fitness, amplitude=stats_amplitude, 
                                    frequency=stats_frequency, phase_shift=stats_phase_shift)
    stats.register('avg', lambda x: round(np.mean(x), 4))
    stats.register('std', lambda x: round(np.std(x), 4))
    stats.register('min', lambda x: round(np.min(x), 4))
    stats.register('max', lambda x: round(np.max(x), 4))

    logbooks = list()

    strategy = cma.Strategy(centroid=np.random.uniform(-1, 1, N), lambda_=lambda_, sigma=sigma) # 
    toolbox.register('generate', strategy.generate, creator.Individual)
    toolbox.register('update', strategy.update)

    logbooks.append(tools.Logbook())
    logbooks[-1].header = 'gen', 'evals', 'fitness', 'amplitude', 'frequency', 'phase_shift'
    logbooks[-1].chapters['fitness'].header = 'std', 'min', 'avg', 'max'
    logbooks[-1].chapters['amplitude'].header = 'std', 'min', 'avg', 'max'
    logbooks[-1].chapters['frequency'].header = 'std', 'min', 'avg', 'max'
    logbooks[-1].chapters['phase_shift'].header = 'std', 'min', 'avg', 'max'

    MAXITER = 100
    i = 0
    while i < MAXITER:
        # Generate a new population
        population = toolbox.generate()

        # Evaluate the individuals:
        # The following commented out line is how DEAP suggests to calculate fitness
        # fitnesses = toolbox.map(toolbox.evaluate, population, unity_interface) # fuck this

        # Here, it is done with the following to make evaluation parallell
        # (evaluate many individuals) in Unity.
        fitnesses = toolbox.evaluate(population, unity_interface) # very nice

        # Assign the computed fitness to individuals
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # update hall of fame with current populations best k individuals
        halloffame.update(population)

        # This creates stats of current population using the stats.register('') above
        record = stats.compile(population)
        logbooks[-1].record(gen=i, evals=lambda_, **record)

        if verbose:
            print(logbooks[-1].stream)

        # Update the strategy with the evaluated individuals
        toolbox.update(population)
        i += 1

    return logbooks, halloffame



# def train(unity_interface, verbose=True):
#     np.random.seed(128)
#     NRESTARTS = 10  # Initialization + 9 I-POP restarts
#     SIGMA0 = 2.0    # 1/5th of the domain [-5 5]

#     toolbox = base.Toolbox()

#     # Register custom fitness evaluation function
#     toolbox.register('evaluate', fitness_evaluation.evaluate_population)

#     # Create the hall of fame (the container storing the best individuals)
#     halloffame = tools.HallOfFame(1)

#     # TODO: move these to functions.py or something else
#     def amplitude(ind):
#         return np.mean(np.array([ind[i] for i in range(0,len(ind),3)]))
#     def frequency(ind):
#         return np.mean(np.array([ind[i] for i in range(1,len(ind),3)]))
#     def phase_shift(ind):
#         return np.mean(np.array([ind[i] for i in range(2,len(ind),3)]))

#     # Create the statistics functions
#     stats_fitness       = tools.Statistics(lambda ind: ind.fitness.values)
#     stats_amplitude     = tools.Statistics(amplitude)
#     stats_frequency     = tools.Statistics(frequency)
#     stats_phase_shift   = tools.Statistics(phase_shift)
#     # Register statistics functions
#     stats = tools.MultiStatistics(  fitness=stats_fitness, amplitude=stats_amplitude, 
#                                     frequency=stats_frequency, phase_shift=stats_phase_shift)
#     stats.register('avg', lambda x: round(np.mean(x), 2))
#     stats.register('std', lambda x: round(np.std(x), 2))
#     stats.register('min', lambda x: round(np.min(x), 2))
#     stats.register('max', lambda x: round(np.max(x), 2))

#     # Create a list to store all the logbooks. One logbook is created for each run.
#     logbooks = list()

#     nsmallpopruns = 0
#     smallbudget = list()
#     largebudget = list()
#     # lambda0 = 4 + int(3 * np.log(N)) # population size - dynamic, dependent on individual size
#     lambda0 = 60 # population size - static
#     regime = 1
    
#     i = 0
#     while i < (NRESTARTS + nsmallpopruns):
#         # The first regime is enforced on the first and last restart
#         # The second regime is run if its allocated budget is smaller than the allocated
#         # large population regime budget
#         if i > 0 and i < (NRESTARTS + nsmallpopruns) - 1 and sum(smallbudget) < sum(largebudget):
#             lambda_ = int(lambda0 * (0.5 * (2**(i - nsmallpopruns) * lambda0) / lambda0)**(np.random.rand()**2))
#             sigma = 2 * 10**(-2 * np.random.rand())
#             nsmallpopruns += 1
#             regime = 2
#             smallbudget += [0]
#         else:
#             lambda_ = 2**(i - nsmallpopruns) * lambda0
#             sigma = SIGMA0
#             regime = 1
#             largebudget += [0]

#         t = 0

#         # Set the termination criterion constants
#         if regime == 1:
#             MAXITER = 100 + 50 * (N + 3)**2 / np.sqrt(lambda_)
#         elif regime == 2:
#             MAXITER = 0.5 * largebudget[-1] / lambda_

#         TOLHISTFUN = 10**-12
#         TOLHISTFUN_ITER = 10 + int(np.ceil(30. * N / lambda_))
#         EQUALFUNVALS = 1. / 3.
#         EQUALFUNVALS_K = int(np.ceil(0.1 + lambda_ / 4.))
#         TOLX = 10**-12
#         TOLUPSIGMA = 10**20
#         CONDITIONCOV = 10**14
#         STAGNATION_ITER = int(np.ceil(0.2 * t + 120 + 30. * N / lambda_))
#         NOEFFECTAXIS_INDEX = t % N

#         equalfunvalues = list()
#         bestvalues = list()
#         medianvalues = list()
#         mins = deque(maxlen=TOLHISTFUN_ITER)

#         # We start with a centroid in [-4, 4]**D
#         strategy = cma.Strategy(centroid=np.random.uniform(-1, 1, N), sigma=sigma, lambda_=lambda_)
#         toolbox.register('generate', strategy.generate, creator.Individual)
#         toolbox.register('update', strategy.update)

#         # Create a new toolbox for each restart
#         logbooks.append(tools.Logbook())
#         logbooks[-1].header = 'gen', 'evals', 'restart', 'regime', 'fitness', 'amplitude', 'frequency', 'phase_shift'
#         logbooks[-1].chapters['fitness'].header = 'std', 'min', 'avg', 'max'
#         logbooks[-1].chapters['amplitude'].header = 'std', 'min', 'avg', 'max'
#         logbooks[-1].chapters['frequency'].header = 'std', 'min', 'avg', 'max'
#         logbooks[-1].chapters['phase_shift'].header = 'std', 'min', 'avg', 'max'

#         # Create stopping criteria dict
#         conditions = {'MaxIter' : False, 'TolHistFun' : False, 'EqualFunVals' : False,
#                       'TolX' : False, 'TolUpSigma' : False, 'Stagnation' : False,
#                       'ConditionCov' : False, 'NoEffectAxis' : False, 'NoEffectCoor' : False}


#         # Run the current regime until one of the following is true:
#         ## Note that the algorithm won't stop by itself on the optimum (0.0 on rastrigin).
#         while not any(conditions.values()):
            
#             # Generate a new population
#             population = toolbox.generate()

#             # Evaluate the individuals:
#             # The following commented out line is how DEAP suggests to calculate fitness
#             # fitnesses = toolbox.map(toolbox.evaluate, population, unity_interface) # fuck this

#             # Here, it is done with the following to make evaluation parallell
#             # (evaluate many individuals) in Unity.
#             fitnesses = toolbox.evaluate(population, unity_interface) # very nice

#             # Assign the computed fitness to individuals
#             for ind, fit in zip(population, fitnesses):
#                 ind.fitness.values = fit

#             # update hall of fame with current populations best k individuals
#             halloffame.update(population)

#             # This creates stats of current population using the stats.register('') above
#             record = stats.compile(population)
#             logbooks[-1].record(gen=t, evals=lambda_, restart=i, regime=regime, **record)

#             if verbose:
#                 print(logbooks[-1].stream)

#             # Update the strategy with the evaluated individuals
#             toolbox.update(population)

#             # Count the number of times the k'th best solution is equal to the best solution
#             # At this point the population is sorted (method update)
#             if population[-1].fitness == population[-EQUALFUNVALS_K].fitness:
#                 equalfunvalues.append(1)

#             # Log the best and median value of this population
#             bestvalues.append(population[-1].fitness.values)
#             medianvalues.append(population[int(round(len(population)/2.))].fitness.values)

#             # First run does not count into the budget
#             if regime == 1 and i > 0:
#                 largebudget[-1] += lambda_
#             elif regime == 2:
#                 smallbudget[-1] += lambda_

#             # Here comes lots of stopping-criteria stuff
#             t += 1
#             STAGNATION_ITER = int(np.ceil(0.2 * t + 120 + 30. * N / lambda_))
#             NOEFFECTAXIS_INDEX = t % N

#             if t >= MAXITER:
#                 # The maximum number of iteration per CMA-ES ran
#                 conditions['MaxIter'] = True

#             min_ = logbooks[-1].chapters['fitness'].select('min')[-1]
#             mins.append(min_)
#             if (len(mins) == mins.maxlen) and max(mins) - min(mins) < TOLHISTFUN:
#                 # The range of the best values is smaller than the threshold
#                 conditions['TolHistFun'] = True

#             if t > N and sum(equalfunvalues[-N:]) / float(N) > EQUALFUNVALS:
#                 # In 1/3rd of the last N iterations the best and k'th best solutions are equal
#                 conditions['EqualFunVals'] = True

#             if all(strategy.pc < TOLX) and all(np.sqrt(np.diag(strategy.C)) < TOLX):
#                 # All components of pc and sqrt(diag(C)) are smaller than the threshold
#                 conditions['TolX'] = True

#             # Need to transfor strategy.diagD[-1]**2 from pyp/np.float64 to python
#             # float to avoid OverflowError
#             if strategy.sigma / sigma > float(strategy.diagD[-1]**2) * TOLUPSIGMA:
#                 # The sigma ratio is bigger than a threshold
#                 conditions['TolUpSigma'] = True

#             if len(bestvalues) > STAGNATION_ITER and len(medianvalues) > STAGNATION_ITER and \
#                np.median(bestvalues[-20:]) >= np.median(bestvalues[-STAGNATION_ITER:-STAGNATION_ITER + 20]) and \
#                np.median(medianvalues[-20:]) >= np.median(medianvalues[-STAGNATION_ITER:-STAGNATION_ITER + 20]):
#                 # Stagnation occurred
#                 conditions['Stagnation'] = True

#             if strategy.cond > 10**14:
#                 # The condition number is bigger than a threshold
#                 conditions['ConditionCov'] = True

#             if all(strategy.centroid == strategy.centroid + 0.1 * strategy.sigma * strategy.diagD[-NOEFFECTAXIS_INDEX] * strategy.B[-NOEFFECTAXIS_INDEX]):
#                 # The coordinate axis std is too low
#                 conditions['NoEffectAxis'] = True

#             if any(strategy.centroid == strategy.centroid + 0.2 * strategy.sigma * np.diag(strategy.C)):
#                 # The main axis std has no effect
#                 conditions['NoEffectCoor'] = True

#         stop_causes = [k for k, v in conditions.items() if v]
#         print('Stopped because of condition%s %s' % ((':' if len(stop_causes) == 1 else 's:'), ','.join(stop_causes)))
#         i += 1

#     return logbooks, halloffame


