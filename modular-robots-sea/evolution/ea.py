import random
import numpy as np
import multiprocessing as mp
from deap import base
from deap import tools
import math

from simulation.unity_evaluator import UnityEvaluator as Evaluator

UNITY_EVALUATOR = None

def get_unity_evaluator(id, steps, editor_mode, headless):
    global UNITY_EVALUATOR
    if UNITY_EVALUATOR is None:
        UNITY_EVALUATOR = Evaluator(steps, editor_mode=editor_mode, headless=headless, worker_id=id)

def evaluate(individual, id, steps, editor_mode, headless):
    get_unity_evaluator(id, steps, editor_mode, headless)
    return UNITY_EVALUATOR.evaluate(individual)

class StandardEvolutionaryAlgorithm:
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
        #toolbox.register("select", tools.selRandom)
        toolbox.register("select_parents", tools.selTournament, tournsize=args.tournament_size)
        toolbox.register("select_survivors", tools.selTournament, tournsize=args.tournament_size)
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

    def evaluate_parallel(self, population):
        results = self.toolbox.map(evaluate, zip(population, [x + self.args.socket_offset*self.args.threads for x in range(len(population))], [self.evaluation_steps for _ in range(len(population))], [self.editor_mode for _ in range(len(population))], [self.headless for _ in range(len(population))]))
        for individual, result in zip(population, results):
            realised_genome = result[2]
            motor_energy_offset = 0
            for val in realised_genome:
                if val == 1 or val == 2 or val == 3 or val == 4:
                    motor_energy_offset += 10
            if result[0] == 0 or result[3] == 0 or motor_energy_offset == 0:
                fitness = -1000
            else:
                fitness = (-sum([abs(x) for x in result[3]]) - (motor_energy_offset))/result[0]
            if self.args.fitness == 'd' or self.args.fitness == 'distance':
                individual.fitness = result[0]
            else:
                individual.fitness = fitness
            individual.map_position = result[1]
            individual.realised_genome = result[2]
            individual.energy_consumption = result[3]
            individual.distance_moved = result[0]
        return population

    def step(self):
        # Select
        parents = self.toolbox.select_parents(self.population, self.population_size)
        # Mutate
        offspring = [self.toolbox.clone(parent) for parent in parents]
        to_be_evaluated = []
        for individual in offspring:
            individual.rng = np.random.default_rng(self.seed.spawn(1)[0])
            self.toolbox.mutate(individual, individual.rng)
            to_be_evaluated.append(individual)

        # Evaluate
        offspring = self.toolbox.evaluate(to_be_evaluated)

        # Replace
        self.population[:] = offspring
        self.steps += 1

    def stat(self, i):
        fitnesses = [individual.fitness for individual in self.population]
        with open(self.args.path + "loggfile" + str(self.start_seed) + ".txt", "a+") as f:
            f.write("Generation: " + str(i+1) + "\tHighest fitness: " + str(max(fitnesses)) + "\tAverage fitness: " + str(sum(fitnesses)/len(fitnesses))+ "\t St. dev.: " + str(np.std(fitnesses))  + "\n")
        if self.elite is None or max(fitnesses) > self.elite.fitness:
            self.elite = self.toolbox.clone(self.population[fitnesses.index(max(fitnesses))])

    def writetime(self):
        from datetime import datetime
        with open(self.args.path + "loggfile" + str(self.start_seed) + ".txt", "a+") as f:
            now = datetime.now()
            f.write("Time: " + now.strftime("%H:%M:%S") + "\tSeed: " + str(self.start_seed) + "\n")

    def save_data(self, i):
        import pickle
        pickle_data = {'population': self.population, 'args': self.args, 'elite': self.elite}
        with open(self.args.path + "pickle_logg_" + str(i+1) + "_" + str(self.start_seed) + ".pkl", "wb") as f:
            pickle.dump(pickle_data, f)

    def load_data(args):
        import pickle
        with open(args.path + "pickle_logg_" + str(args.generations) + "_" + str(args.seed) + ".pkl", "rb") as f:
            pickle_data = pickle.load(f)
        population = pickle_data['population']
        fitnesses = []
        for ind in population:
            fitnesses.append(ind.fitness)
        ind = population[fitnesses.index(max(fitnesses))]
        #ind = pickle_data['elite']
        print("-----" + str(args.seed) + "-----")
        print(pickle_data['args'])
        result = evaluate(ind, 17, args.evaluation_steps, args.editor_mode, args.headless)
        print("Fitness" + str(result[0]))
        print("Supposed fitness" + str(max(fitnesses)))

    def get_fitness_over_time(args, seed):
        import pickle
        max_fitness = []
        avg_fitness = []
        for i in range(args.generations):
            with open(args.path + "pickle_logg_" + str(i+1) + "_" + str(seed) + ".pkl", "rb") as f:
                pickle_data = pickle.load(f)
            population = pickle_data['population']
            fitnesses = []
            for ind in population:
                fitnesses.append(ind.fitness)
            max_fitness.append(max(fitnesses))
            avg_fitness.append(sum(fitnesses)/len(fitnesses))
        return max_fitness, avg_fitness
    
    def get_elite_fitness(args, seed):
        import pickle
        with open(args.path + "pickle_logg_" + str(args.generations) + "_" + str(seed) + ".pkl", "rb") as f:
            pickle_data = pickle.load(f)
        population = pickle_data['population']
        fitnesses = []
        for ind in population:
            fitnesses.append(ind.fitness)
        ind = population[fitnesses.index(max(fitnesses))]
        ind = pickle_data['elite']
        return ind.fitness
    
    def get_elite_cot(args, seed):
        import pickle
        with open(args.path + "pickle_logg_" + str(args.generations) + "_" + str(seed) + ".pkl", "rb") as f:
            pickle_data = pickle.load(f)
        population = pickle_data['population']
        fitnesses = []
        for ind in population:
            fitnesses.append(ind.fitness)
        ind = population[fitnesses.index(max(fitnesses))]
        ind = pickle_data['elite']
        return (-sum([abs(x) for x in ind.energy_consumption]))/ind.fitness
    
    def get_elite_joint_count(args, seed):
        import pickle
        with open(args.path + "pickle_logg_" + str(args.generations) + "_" + str(seed) + ".pkl", "rb") as f:
            pickle_data = pickle.load(f)
        population = pickle_data['population']
        fitnesses = []
        for ind in population:
            fitnesses.append(ind.fitness)
        ind = population[fitnesses.index(max(fitnesses))]
        ind = pickle_data['elite']
        module_count = 0
        for val in ind.realised_genome:
            if val == 3:
                module_count += 1
        return module_count

    def get_elite_module_count(args, seed):
        import pickle
        with open(args.path + "pickle_logg_" + str(args.generations) + "_" + str(seed) + ".pkl", "rb") as f:
            pickle_data = pickle.load(f)
        population = pickle_data['population']
        fitnesses = []
        for ind in population:
            fitnesses.append(ind.fitness)
        ind = population[fitnesses.index(max(fitnesses))]
        ind = pickle_data['elite']
        module_count = 0
        for val in ind.realised_genome:
            if val != 0:
                module_count += 1
        return module_count
    
    def get_elite_module_weight(args, seed):
        import pickle
        with open(args.path + "pickle_logg_" + str(args.generations) + "_" + str(seed) + ".pkl", "rb") as f:
            pickle_data = pickle.load(f)
        module_count = 0
        population = pickle_data['population']
        fitnesses = []
        for ind in population:
            fitnesses.append(ind.fitness)
        ind = population[fitnesses.index(max(fitnesses))]
        ind = pickle_data['elite']
        for val in ind.realised_genome:
            if val == 3:
                module_count += 1
            if val == 5:
                module_count += 0.33
            if val == 6:
                module_count += 1
        return module_count
    
    def get_elite_module_volume(args, seed):
        import pickle
        with open(args.path + "pickle_logg_" + str(args.generations) + "_" + str(seed) + ".pkl", "rb") as f:
            pickle_data = pickle.load(f)
        module_count = 0
        population = pickle_data['population']
        fitnesses = []
        for ind in population:
            fitnesses.append(ind.fitness)
        ind = population[fitnesses.index(max(fitnesses))]
        ind = pickle_data['elite']
        for val in ind.realised_genome:
            if val == 3:
                module_count += 1
            if val == 5:
                module_count += 1
            if val == 6:
                module_count += 3
        return module_count
    
    def get_elite_depth(args, seed):
        import pickle
        with open(args.path + "pickle_logg_" + str(args.generations) + "_" + str(seed) + ".pkl", "rb") as f:
            pickle_data = pickle.load(f)
        population = pickle_data['population']
        fitnesses = []
        for ind in population:
            fitnesses.append(ind.fitness)
        ind = population[fitnesses.index(max(fitnesses))]
        ind = pickle_data['elite']
        realised_nodes = ind.encoding.get_realised_nodes(ind.realised_genome)
        depth = 0
        for node in realised_nodes:
            if node.depth > depth:
                depth = node.depth
        return depth
    
    def get_elite_width(args, seed):
        import pickle
        with open(args.path + "pickle_logg_" + str(args.generations) + "_" + str(seed) + ".pkl", "rb") as f:
            pickle_data = pickle.load(f)
        population = pickle_data['population']
        fitnesses = []
        for ind in population:
            fitnesses.append(ind.fitness)
        ind = population[fitnesses.index(max(fitnesses))]
        ind = pickle_data['elite']
        realised_nodes = ind.encoding.get_realised_nodes(ind.realised_genome)
        depth = 0
        for node in realised_nodes:
            if node.depth > depth:
                depth = node.depth
        widths = [0 for _ in range(depth+1)]
        for node in realised_nodes:
            widths[node.depth] += 1
        return float(max(widths))

    def get_elite_all_springyness(args, seed):
        import pickle
        with open(args.path + "pickle_logg_" + str(args.generations) + "_" + str(seed) + ".pkl", "rb") as f:
            pickle_data = pickle.load(f)
        population = pickle_data['population']
        fitnesses = []
        for ind in population:
            fitnesses.append(ind.fitness)
        ind = population[fitnesses.index(max(fitnesses))]
        ind = pickle_data['elite']
        module_count = 0
        total_springyness = []
        genome_length, genome, springyness, rotation = ind.get_unity_data()
        for module, springyness in zip(ind.realised_genome, springyness):
            if module == 3:
                module_count += 1
                total_springyness.append(((((math.e**(math.e*springyness-math.e))*1000)-65)/(1000-65))*1000)
                #total_springyness += springyness
        return total_springyness
    
    def get_elite_avg_springyness(args, seed):
        import pickle
        with open(args.path + "pickle_logg_" + str(args.generations) + "_" + str(seed) + ".pkl", "rb") as f:
            pickle_data = pickle.load(f)
        population = pickle_data['population']
        fitnesses = []
        for ind in population:
            fitnesses.append(ind.fitness)
        ind = population[fitnesses.index(max(fitnesses))]
        ind = pickle_data['elite']
        module_count = 0
        total_springyness = []
        genome_length, genome, springyness, rotation = ind.get_unity_data()
        for module, springyness in zip(ind.realised_genome, springyness):
            if module == 3:
                module_count += 1
                total_springyness += ((((math.e**(math.e*springyness-math.e))*1000)-65)/(1000-65))*1000
                #total_springyness += springyness
        return total_springyness/module_count

    def get_springyness_over_time(args, seed):
        import pickle
        bins = 80
        mat = []
        for i in range(0, args.generations):
            row = [0 for _ in range(bins)]
            with open(args.path + "pickle_logg_" + str(i+1) + "_" + str(seed) + ".pkl", "rb") as f:
                pickle_data = pickle.load(f)
            #pickle_data['population']
            population = [pickle_data['elite']]
            for ind in population:
                spring_proportions = []
                genome_length, genome, springyness, _ = ind.get_unity_data()
                for module_i in range(len(ind.realised_genome)):
                    if ind.realised_genome[module_i] == 3 or ind.realised_genome[module_i] == 4:
                        spring_proportions.append((((math.e**(math.e*springyness[module_i]-math.e))*1000)-65)/(1000-65))  
                for spring_proportion in spring_proportions:
                    index = int(spring_proportion*(bins-1))
                    row[index] += 1
            mat.append(row)
        mat = np.array(mat).T.tolist()
        return mat

