from deap import base

class _Fitness(base.Fitness):
    def __init__(self):
        self.weights = (1.0, 1.0)

class Individual:
    def __init__(self, Encoding, Controller, max_modules, max_depth, num_symbols, self_adaptive, module_types):
        self.rng = None
        self.encoding = Encoding(Controller, max_modules, max_depth, num_symbols, self_adaptive, module_types)
        self.fitness = _Fitness()
        self.distance_moved = None
        self.energy_consumption = None
        self.realised_genome = None
        self.map_position = None

    def get_unity_data(self):
        genome, springyness, rotation = self.encoding.get_base_encoding()
        genome_length = len(genome)
        print(rotation)
        return genome_length, genome, springyness, rotation

    def get_actions(self, time, observations=None):
        return self.encoding.get_actions(time, observations=observations)

    def mutate(individual, rng, mutation_probability, mutation_sigma, mutation_probability_controller, mutation_sigma_controller):
        return individual.encoding.mutate(mutation_probability, mutation_sigma, mutation_probability_controller, mutation_sigma_controller, rng)

    def crossover(individual, other, rng):
        individual.encoding.crossover(other.encoding, rng)

    def init_random_robot(individual, rng):
        individual.encoding.init_random_robot(rng)
