from Chromosomes import Poly_Chrom, Tree_Chrom, Tree_Terminal
import numpy as np
import copy
import math


class Population:

    def __init__(self, fitness_functions, population_size, test_interval, birth_rate, mutation, constraint):
        self.test_interval = test_interval
        self.fitness_functions = fitness_functions
        self.polynomials = np.empty(population_size)
        self.birth_rate = birth_rate
        self.mutation = mutation
        self.average_fitness_over_time = []
        self.fitness_over_time = []
        self.constraint = constraint

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
        self.polynomials = np.array(np.append(copy.deepcopy(self.polynomials),
                                              [breeders_1[i].discrete_crossover(breeders_2[i], mutation=self.mutation)
                                               for i in range(breeders_1.size)]))

    def merge_populations(self, population):
        self.polynomials = np.append(self.polynomials, population.polynomials)


class Poly_Population(Population):

    def __init__(self, fitness_functions, population_size, test_interval, birth_rate, mutation, coeffs_bound,
                 constraint, operator_functions=None):
        super().__init__(fitness_functions, population_size, test_interval, birth_rate, mutation, constraint)
        self.polynomials = np.array(
            [Poly_Chrom(coeffs_bound=coeffs_bound, order_upper_bound=constraint) for _ in range(population_size)])


class Tree_Population(Population):

    def __init__(self, fitness_functions, population_size, test_interval, birth_rate, mutation, coeffs_bound,
                 constraint, operator_functions=None):
        super().__init__(fitness_functions, population_size, test_interval, birth_rate, mutation, constraint)
        self.polynomials = np.array(
            [Tree_Chrom(coeffs_bound=coeffs_bound, depth_limit=constraint, operator_functions=operator_functions) for _
             in range(population_size)])


class Tree_SimAnnealing_Population(Population):

    def __init__(self, fitness_functions, population_size, test_interval, birth_rate, mutation, coeffs_bound,
                 constraint, operator_functions=None):
        self.coeffs_bound = coeffs_bound
        super().__init__(fitness_functions, population_size, test_interval, birth_rate, mutation, constraint)
        self.polynomials = np.array(
            [self.simulated_annealing(Tree_Chrom(coeffs_bound=coeffs_bound, depth_limit=constraint, operator_functions=operator_functions)) for _
             in range(population_size)])

    def simulated_annealing(self, original_polynomial, max_iterations=900, max_temp=100000, temp_change=0.98):
        if isinstance(original_polynomial, Tree_Chrom):
            current_polynomial = copy.deepcopy(original_polynomial)
            best_polynomial = original_polynomial
            i_polynomial = copy.deepcopy(current_polynomial)
            terminals = i_polynomial.return_terminals()
            for _ in range(max_iterations):
                terminals = np.array([t.go_to_neighbour(int((self.coeffs_bound-self.coeffs_bound[0])/4)) for t in terminals])
                max_temp = max_temp * temp_change
                i_cost = self.annealing_cost(i_polynomial)
                current_cost = self.annealing_cost(current_polynomial)
                prob = math.exp((current_cost - i_cost) / max_temp)
                if i_cost <= current_cost:
                    current_polynomial = copy.deepcopy(i_polynomial)
                    if i_cost <= self.annealing_cost(best_polynomial):
                        best_polynomial = copy.deepcopy(i_polynomial)
                elif np.random.choice([True, False], p=[prob, 1 - prob]):
                    current_polynomial = copy.deepcopy(i_polynomial)
            return best_polynomial
        return original_polynomial

    def annealing_cost(self, polynomial):
        return 1 / (1 + polynomial.primes_fitness(self.test_interval, self.fitness_functions))
