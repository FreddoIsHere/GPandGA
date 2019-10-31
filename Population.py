from Chromosomes import PolyChrom, Tree_Chrom
import numpy as np
import copy


class Population:

    def __init__(self, fitness_functions, population_size, test_interval):
        self.test_interval = test_interval
        self.fitness_functions = fitness_functions
        self.polynomials = np.empty(population_size)

    def introduce_new_generation(self, death_rate=0.95, epsilon=1):
        print("New gen")
        for p in self.polynomials:
            print(p.eval_polynomial("x"))

        scores = np.array(
            [p.primes_fitness(self.test_interval, self.fitness_functions) for p in self.polynomials]) + epsilon
        probabilities = scores / np.sum(scores)
        breeders_1 = np.random.choice(self.polynomials, size=int(self.polynomials.size * death_rate), replace=True,
                                      p=probabilities)
        breeders_2 = np.random.choice(copy.deepcopy(self.polynomials), size=int(self.polynomials.size * death_rate), replace=True,
                                      p=probabilities)
        self.polynomials = np.array([breeders_1[i].discrete_crossover(breeders_2[i]) for i in range(breeders_1.size)])

    def merge_populations(self, population):
        self.polynomials = np.append(self.polynomials, population.polynomials)


class Poly_Population(Population):

    def __init__(self, fitness_functions, population_size, test_interval):
        super().__init__(fitness_functions, population_size, test_interval)
        self.polynomials = np.array([PolyChrom() for _ in range(population_size)])


class Tree_Population(Population):

    def __init__(self, fitness_functions, population_size, test_interval):
        super().__init__(fitness_functions, population_size, test_interval)
        self.polynomials = np.array([Tree_Chrom() for _ in range(population_size)])
