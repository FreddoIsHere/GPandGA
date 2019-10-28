from enum import Enum

from Chromosomes import PolyChrom
import numpy as np


class Poly_Population:

    def __init__(self, fitness_functions, population_size, test_interval):
        self.test_interval = test_interval
        self.fitness_functions = fitness_functions
        self.polynomials = np.array([PolyChrom() for _ in range(population_size)])

    def introduce_new_generation(self, death_rate=0.95, epsilon=5):
        scores = np.array([p.primes_fitness(self.test_interval, self.fitness_functions) for p in self.polynomials]) + epsilon
        probabilities = scores / np.sum(scores)
        breeders_1 = np.random.choice(self.polynomials, size=int(self.polynomials.size * death_rate), replace=True,
                                      p=probabilities)
        breeders_2 = breeders_1
        np.random.shuffle(breeders_2)
        self.polynomials = np.array([breeders_1[i].discrete_crossover(breeders_2[i]) for i in range(breeders_1.size)])

    def merge_populations(self, poly_population):
        self.polynomials = np.append(self.polynomials, poly_population.polynomials)
