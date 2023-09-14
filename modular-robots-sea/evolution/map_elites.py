import random
import itertools
import numpy as np
import multiprocessing as mp
from deap import base
from deap import tools

from simulation.unity_evaluator import UnityEvaluator as Evaluator

UNITY_EVALUATOR = None

def get_unity_evaluator(id, steps, editor_mode, headless):
    global UNITY_EVALUATOR
    if UNITY_EVALUATOR is None:
        UNITY_EVALUATOR = Evaluator(steps, editor_mode=editor_mode, headless=headless, worker_id=id)

def evaluate(individual, id, steps, editor_mode, headless):
    get_unity_evaluator(id, steps, editor_mode, headless)
    return UNITY_EVALUATOR.evaluate(individual)

class MapElites:
    def __init__(self, args, Individual, Encoding, Controller):
        self.steps = 0
        self.crossover_probability = args.crossover_probability
        self.args = args
        self.elite = None

        # Evaluation setup
        self.evaluation_steps = args.evaluation_steps
        self.editor_mode = args.editor_mode
        self.headless = args.headless

        # Rng setup
        self.start_seed = args.seed
        self.seed = np.random.SeedSequence(self.start_seed)
        self.rng = np.random.default_rng(self.seed)
        master_seed = self.rng.integers(2**32 - 1)
        random.seed(master_seed)
        np.random.seed(master_seed)

        self.writetime()

        # Deap setup
        toolbox = base.Toolbox()
        toolbox.register("individual", Individual, Encoding=Encoding, Controller=Controller, max_modules=args.max_modules, max_depth=args.max_depth, num_symbols=args.symbol_size, self_adaptive=args.self_adaptive, module_types=args.module_types)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evaluate_parallel)
        toolbox.register("mutate", Individual.mutate, mutation_probability=args.mutation_parameters[0], mutation_sigma=args.mutation_parameters[1], mutation_probability_controller=args.mutation_parameters[2], mutation_sigma_controller=args.mutation_parameters[3])
        toolbox.register("init_robot", Individual.init_random_robot)
        toolbox.register("select", tools.selRandom)
        #toolbox.register("select", tools.selTournament, tournsize=3)
        pool = mp.Pool(args.threads)
        chunksize = int(np.ceil(float(args.population_size)/float(args.threads)))
        toolbox.register("map", pool.starmap, chunksize=chunksize)
        self.toolbox = toolbox

        # Population setup
        self.population_size = args.population_size
        self.population = self.toolbox.population(n=self.population_size)
        for individual in self.population:
            individual.rng = np.random.default_rng(self.seed.spawn(1)[0])
            self.toolbox.init_robot(individual, individual.rng)
        self.population = self.toolbox.evaluate(self.population)

        # Map setup
        self.map_sizes = [args.map_resolution, args.map_resolution]
        self.map_dimensions = args.map_dimensions
        self.map_shape = tuple([self.map_sizes[x] for x in range(self.map_dimensions)])
        self.map = np.full(self.map_shape, None)
        # Place population in map
        for individual in self.population:
            map_index = tuple([int(map_position*map_size) for map_position, map_size in zip(individual.map_position, self.map_sizes)])
            if self.map[map_index] is None or individual.fitness > self.map[map_index].fitness:
                self.map[map_index] = individual
        # Replace population with map inhabitants
        self.population = []
        map_indexes = [tuple([x, y]) for x in range(self.map_sizes[0]) for y in range(self.map_sizes[1])]
        for map_index in map_indexes:
            if self.map[map_index] is not None:
                self.population.append(self.map[map_index])

    def evaluate_parallel(self, population):
        results = self.toolbox.map(evaluate, zip(population, [x + self.args.socket_offset*self.args.threads for x in range(len(population))], [self.evaluation_steps for _ in range(len(population))], [self.editor_mode for _ in range(len(population))], [self.headless for _ in range(len(population))]))
        for individual, result in zip(population, results):
            individual.map_position = result[1]
            individual.realised_genome = result[2]
            individual.energy_consumption = result[3]
            individual.distance_moved = result[0]
            realised_genome = result[2]

            motor_energy_offset = 0
            for val in realised_genome:
                if val == 1 or val == 2 or val == 3 or val == 4:
                    motor_energy_offset += 10
            if result[0] == 0 or result[3] == 0 or motor_energy_offset == 0:
                fitness = -1000
            else:
                fitness = ((-sum([abs(x) for x in result[3]])) - (motor_energy_offset))/result[0]
            if self.args.fitness == 'd' or self.args.fitness == 'distance':
                individual.fitness = result[0]
            else:
                individual.fitness = fitness
        return population

    def step(self):
        # Select
        parents = self.toolbox.select(self.population, self.population_size)
        # Mutate
        offspring = [self.toolbox.clone(parent) for parent in parents]
        to_be_evaluated = []
        not_evaluated = []
        for individual_1, individual_2 in zip(offspring[::2], offspring[1::2]):
            individual_1.rng = np.random.default_rng(self.seed.spawn(1)[0])
            individual_2.rng = np.random.default_rng(self.seed.spawn(1)[0])
            self.toolbox.mutate(individual_1, individual_1.rng)
            to_be_evaluated.append(individual_1)
            self.toolbox.mutate(individual_2, individual_2.rng)
            to_be_evaluated.append(individual_2)
        # Evaluate
        offspring = self.toolbox.evaluate(to_be_evaluated)
        # Place in map
        for individual in offspring:
            map_index = tuple([int(map_position*map_size) for map_position, map_size in zip(individual.map_position, self.map_sizes)])
            if self.map[map_index] is None or individual.fitness > self.map[map_index].fitness:
                self.map[map_index] = individual
        # Replace population with map inhabitants
        self.population = []
        map_indexes = [tuple([x, y]) for x in range(self.map_sizes[0]) for y in range(self.map_sizes[1])]
        for map_index in map_indexes:
            if self.map[map_index] is not None:
                self.population.append(self.map[map_index])
        self.steps += 1

    def stat(self, i):
        fitnesses = [individual.fitness for individual in self.population]
        with open(self.args.path + "loggfile" + str(self.start_seed) + ".txt", "a+") as f:
            f.write("Generation: " + str(i+1) + "\tHighest fitness: " + str(max(fitnesses)) + "\tAverage fitness: " + str(sum(fitnesses)/len(fitnesses)) + "\tPopsize: " + str(len(self.population)) + "\n")
        if self.elite is None or max(fitnesses) > self.elite.fitness:
            self.elite = self.toolbox.clone(self.population[fitnesses.index(max(fitnesses))])

    def writetime(self):
        from datetime import datetime
        with open(self.args.path + "loggfile" + str(self.start_seed) + ".txt", "a+") as f:
            now = datetime.now()
            f.write("Time: " + now.strftime("%H:%M:%S") + "\n")
            f.write("Seed: " + str(self.start_seed) + "\n")

    def save_map(self):
        fitness_map = np.full(self.map_shape, np.nan)
        map_indexes = [tuple([x, y]) for x in range(self.map_sizes[0]) for y in range(self.map_sizes[1])]
        for map_index in map_indexes:
            if self.map[map_index] is not None:
                fitness_map[map_index] = self.map[map_index].fitness
        import matplotlib.pyplot as plt
        fitness_map = fitness_map.reshape((20,20))
        plt.matshow(fitness_map)
        plt.savefig("loggmap" + str(self.start_seed) + ".png")

    def save_data(self, i):
        import pickle
        pickle_data = {'individual_map': self.map, 'args': self.args, 'elite': self.elite}
        with open(self.args.path + "pickle_logg_map_" + str(i+1) + "_" + str(self.start_seed) + ".pkl", "wb") as f:
            pickle.dump(pickle_data, f)

    def load_data(args, seed):
        import pickle
        with open(args.path + "pickle_logg_map_" + str(args.generations) + "_" + str(seed) + ".pkl", "rb") as f:
            pickle_data = pickle.load(f)
        individual_map = pickle_data['individual_map']
        map_sizes = [args.map_resolution, args.map_resolution]
        map_shape = tuple([map_sizes[x] for x in range(args.map_dimensions )])
        fitness_map = np.full(map_shape, np.nan)
        elite = None
        elite_dist = None

        map_indexes = [tuple([x, y]) for x in range(map_sizes[0]) for y in range(map_sizes[1])]
        for map_index in map_indexes:
            if individual_map[map_index] is not None and individual_map[map_index].fitness > -1000:
                fitness_map[map_index] = individual_map[map_index].fitness
                #fitness_map[map_index] = individual_map[map_index].distance_moved
                if elite_dist is None or individual_map[map_index].distance_moved > elite_dist:
                    elite_dist = individual_map[map_index].distance_moved
                    elite = individual_map[map_index]
        return fitness_map

