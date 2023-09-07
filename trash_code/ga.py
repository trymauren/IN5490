import random
import string

TARGET_STRING = "Hello, World!"
GENES = string.ascii_letters + ' ,!'
POPULATION_SIZE = 1000

class Individual:
    def __init__(self, chromosome=None):
        if chromosome:
            self.chromosome = chromosome
        else:
            self.chromosome = [random.choice(GENES) for _ in range(len(TARGET_STRING))]
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self):
        return sum(1 for c1, c2 in zip(self.chromosome, TARGET_STRING) if c1 == c2)

    @staticmethod
    def mutate():
        return random.choice(GENES)

    @staticmethod
    def mate(parent1, parent2):
        child_chromosome = []
        for gp1, gp2 in zip(parent1.chromosome, parent2.chromosome):
            if random.random() < 0.5:
                child_chromosome.append(gp1)
            else:
                child_chromosome.append(gp2)
            
        # Random mutation
        if random.random() < 0.05:
            index = random.randint(0, len(child_chromosome)-1)
            child_chromosome[index] = Individual.mutate()
        
        return Individual(child_chromosome)
