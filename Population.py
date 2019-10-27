from Chromosomes import PolyChrom
import numpy as np


class Poly_Population:

    def __init__(self, population_size, test_interval, seed):
        np.random.seed(seed)
        self.seed = seed
        self.test_interval = test_interval
        self.polynomials = np.array([PolyChrom(seed) for _ in range(population_size)])

    def introduce_new_generation(self, birth_rate=0.95):
        scores = np.array([p.num_primes_fitness(self.test_interval) for p in self.polynomials])
        probabilities = scores / np.sum(scores)
        new_polynomials = np.random.choice(self.polynomials, size=int(self.polynomials.size * birth_rate), replace=True,
                                           p=probabilities)
        self.polynomials = new_polynomials
