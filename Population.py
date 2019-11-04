from Chromosomes import Poly_Chrom, Tree_Chrom
import numpy as np
import copy


class Population:

    def __init__(self, fitness_functions, population_size, test_interval, birth_rate, mutation):
        self.test_interval = test_interval
        self.fitness_functions = fitness_functions
        self.polynomials = np.empty(population_size)
        self.birth_rate = birth_rate
        self.mutation = mutation
        self.average_fitness_over_time = []
        self.fitness_over_time = []

    def introduce_new_generation(self):
        num_deaths = self.selection()
        self.replenish(num_deaths)
        return num_deaths != 0

    def selection(self):
        fitnesses = [p.primes_fitness(self.test_interval, self.fitness_functions) for p in self.polynomials]
        num_polynomials = len(fitnesses)
        mean = np.mean(fitnesses)
        self.fitness_over_time.append(fitnesses)
        self.average_fitness_over_time.append(mean)
        indices = [i for i in range(num_polynomials) if (fitnesses[i] < mean)]
        self.polynomials = np.delete(self.polynomials, indices)
        return num_polynomials - self.polynomials.size

    def replenish(self, num_deaths):
        fitnesses = [p.primes_fitness(self.test_interval, self.fitness_functions) for p in self.polynomials]
        scores = np.array(fitnesses) + 0.1 * self.polynomials.size
        probabilities = scores / np.sum(scores)
        breeders_1 = np.random.choice(self.polynomials, size=int(num_deaths * self.birth_rate), replace=True,
                                      p=probabilities)
        breeders_2 = np.random.choice(copy.deepcopy(self.polynomials),
                                      size=int(num_deaths * self.birth_rate), replace=True,
                                      p=probabilities)
        self.polynomials = np.array(np.append(copy.deepcopy(self.polynomials), [breeders_1[i].discrete_crossover(breeders_2[i], mutation=self.mutation) for i in range(breeders_1.size)]))

    def merge_populations(self, population):
        self.polynomials = np.append(self.polynomials, population.polynomials)


class Poly_Population(Population):

    def __init__(self, fitness_functions, population_size, test_interval, birth_rate, mutation, coeffs_bound, constraint, operator_functions=None):
        super().__init__(fitness_functions, population_size, test_interval, birth_rate, mutation)
        self.polynomials = np.array([Poly_Chrom(coeffs_bound=coeffs_bound, order_upper_bound=constraint) for _ in range(population_size)])


class Tree_Population(Population):

    def __init__(self, fitness_functions, population_size, test_interval, birth_rate, mutation, coeffs_bound, constraint, operator_functions=None):
        super().__init__(fitness_functions, population_size, test_interval, birth_rate, mutation)
        self.polynomials = np.array([Tree_Chrom(coeffs_bound=coeffs_bound, depth_limit=constraint, operator_functions=operator_functions) for _ in range(population_size)])
