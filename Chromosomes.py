import random
from enum import IntEnum

import numpy as np
from gmpy import is_prime, mpz
import math


class Fitness(IntEnum):
    number_of_primes_in_interval = 0
    number_of_consecutive_primes_in_interval = 1


class PolyChrom():

    def __init__(self, values=None, coeffs_bound=50, order_upper_bound=7):
        self.coeffs_bound = coeffs_bound
        self.order_upper_bound = order_upper_bound
        if values is None:
            self.value = np.array(
                random.sample(range(-coeffs_bound, coeffs_bound), random.randint(1, order_upper_bound)))
        else:
            self.value = values

    def discrete_crossover(self, polychrom, mutation=True):
        """
        PolyChrom:param polychrom: Polychrom this Polychrom breeds with
        PolyChrom:return: Child of this Polychrom and polychrom
        """
        max_order = max(polychrom.value.size, self.value.size)
        min_order = min(polychrom.value.size, self.value.size)
        a = np.append(self.value, np.zeros(self.order_upper_bound - min_order))
        b = np.append(polychrom.value, np.zeros(self.order_upper_bound - min_order))
        child_values = np.zeros(max_order)
        for i in range(max_order):
            child_values[i] = random.choice((a[i], b[i]))
        new_values = self.mutate(mutation, child_values)
        return PolyChrom(values=new_values)

    def mutate(self, mutation, values):
        if mutation and np.random.choice([False, True], size=1, p=[0.99, 0.01])[0]:
            values[random.randint(0, values.size - 1)] = random.randint(-self.coeffs_bound, self.coeffs_bound)
            return np.trim_zeros(values, trim='b')
        else:
            return np.trim_zeros(values, trim='b')

    def smooth_crossover(self):
        pass

    def eval_polynomial(self, x):
        """
        int or array of ints:param x: Integers to evaluate on
        int or array of ints:return: x evaluated on the polynomial
        """
        order = self.value.size
        X = np.tile(x, (order, 1))
        power = np.array(range(0, order)).reshape((order, 1))
        results = np.power(X, power)
        if self.value.size == 1:
            return [self.value[0]]
        return np.dot(self.value, results)

    def primes_fitness(self, interval, fitness_functions):
        functions = [self.num_primes_fitness, self.num_consecutive_primes_fitness]
        fitness = 0
        for i in fitness_functions:
            function = functions[int(i)]
            fitness += function(interval)
        return fitness

    def num_primes_fitness(self, interval):
        """
        tuple (l, u):param interval: Interval to test on [l, u)
        int:return: fitness score
        """
        results = self.eval_polynomial(np.array(range(interval[0], interval[1])))
        return np.sum([0.5 * is_prime(int(x)) for x in results])

    def num_consecutive_primes_fitness(self, interval):
        results = self.eval_polynomial(np.array(range(interval[0], interval[1])))
        longest_run = 0
        current_run = 0
        for x in results:
            if is_prime(int(x)):
                current_run += 1
            else:
                if current_run > longest_run:
                    longest_run = current_run
                current_run = 0
        if current_run > longest_run:
            longest_run = current_run
        return longest_run


class ReLUChrom():

    def __init__(self):
        pass
