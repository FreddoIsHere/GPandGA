import random
import numpy as np
from gmpy import is_prime
import math


class PolyChrom():

    def __init__(self, seed, values=None, coeffs_upper_bound=100, order_upper_bound=10):
        random.seed(seed)
        self.seed = seed
        if values is None:
            self.value = random.sample(range(0, coeffs_upper_bound), random.randint(1, order_upper_bound))
        self.value = values

    def discrete_crossover(self, polychrom):
        """
        PolyChrom:param polychrom: Polychrom this Polychrom breeds with
        PolyChrom:return: Child of this Polychrom and polychrom
        """
        max_order = max(polychrom.value.size, self.value.size)
        min_order = min(polychrom.value.size, self.value.size)
        a = np.append(self.value, np.zeros(max_order - min_order))
        b = np.append(polychrom.value, np.zeros(max_order - min_order))
        child_values = np.zeros(max_order)
        for i in range(max_order):
            child_values[i] = random.choice((a[i], b[i]))
        return PolyChrom(seed=self.seed, values=np.trim_zeros(child_values, trim='b'))

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
        return np.dot(self.value, results)

    def num_primes_fitness(self, interval):
        """
        tuple (l, u):param interval: Interval to test on [l, u)
        int:return: fitness score
        """
        results = self.eval_polynomial(np.array(range(interval[0], interval[1])))
        return np.sum([1 * is_prime(x) for x in results])


class ReLUChrom():

    def __init__(self):
        pass
