import random
from enum import IntEnum
from abc import ABC, abstractmethod
import numpy as np
from gmpy import is_prime


class Fitness(IntEnum):
    number_of_primes_in_interval = 0
    number_of_consecutive_primes_in_interval = 1
    number_of_consecutive_primes = 2


class Chromosome(ABC):

    def __init__(self):
        super().__init__()

    def primes_fitness(self, interval, fitness_functions):
        actual_functions = [self.num_primes_fitness_in_interval,
                            self.num_consecutive_primes_fitness_in_interval,
                            self.num_consecutive_primes_fitness]
        fitness = 0
        for i in fitness_functions:
            function = actual_functions[int(i)]
            fitness += function(interval)
        return fitness

    def num_primes_fitness_in_interval(self, interval):
        """
        tuple (l, u):param interval: Interval to test on [l, u)
        int:return: fitness score
        """
        results = self.eval_polynomial(np.array(range(interval[0], interval[1]))))
        return np.sum([0.5 * is_prime(int(x)) for x in results])

    def num_consecutive_primes_fitness_in_interval(self, interval):
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

    def num_consecutive_primes_fitness(self, interval):
        i = interval[0]
        last_x = 0
        x = self.eval_polynomial(i)[0]
        while is_prime(int(x)) == 2 and last_x != x:
            i += 1
            last_x = x
            x = self.eval_polynomial(i)[0]
        return i

    @abstractmethod
    def discrete_crossover(self, polychrom, mutation=True):
        pass

    @abstractmethod
    def eval_polynomial(self, x):
        pass

    @abstractmethod
    def print_gp_polynomial(self):
        pass


class PolyChrom(Chromosome):

    def __init__(self, values=None, coeffs_bound=50, order_upper_bound=7):
        super().__init__()
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
        if mutation and np.random.choice([False, True], p=[0.99, 0.01]) and self.value.size > 0:
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

    def print_gp_polynomial(self):
        poly_string = "(" + str(self.value[0]) + ")"
        for i in range(1, self.value.size):
            if self.value[i] != 0:
                poly_string = "(" + str(self.value[i]) + ")" + "x^" + str(i) + " + " + poly_string
        return poly_string


# ----------------------------------------------------------------------------------------------------------------------

def add(x, y):
    if isinstance(x, str) or isinstance(y, str):
        return str(x) + " + " + str(y)
    return x + y


def multiply(x, y):
    if isinstance(x, str) or isinstance(y, str):
        return str(x) + " * " + str(y)
    return np.multiply(x, y)


functions = [add, multiply]


class Tree_Chrom(Chromosome):

    def __init__(self, depth_limit=3, first_subtree=None, operator=None, second_subtree=None):
        super().__init__()
        self.depth_limit=depth_limit
        if operator is None:
            self.first_subtree = Tree_Chrom_Leaf(
                np.random.choice([random.randint(-50, 50), 0], p=[0.7, 0.3]))
            self.operator = np.random.choice(functions, p=[0.7, 0.3])
            self.second_subtree = Tree_Chrom_Leaf(
                np.random.choice([random.randint(-50, 50), 0], p=[0.7, 0.3]))
        else:
            self.first_subtree = first_subtree
            self.operator = operator
            self.second_subtree = second_subtree

    def discrete_crossover(self, treechrom, mutation=True):
        sub_subtree = np.random.choice([Tree_Chrom(), np.random.choice(treechrom.max_list_subtrees())],
                                   p=[mutation * 0.01, 0.99 + (not mutation) * 0.01])
        np.random.choice(self.min_list_subtrees()).set_subtree(sub_subtree, np.random.choice([True, False]))
        return self

    def eval_polynomial(self, x):
        return self.operator(self.first_subtree.eval_polynomial(x), self.second_subtree.eval_polynomial(x))

    def max_list_subtrees(self):
        if self.max_depth() > self.depth_limit:
            return np.append(self.first_subtree.max_list_subtrees(), self.second_subtree.max_list_subtrees())
        return self.min_list_subtrees()

    def max_depth(self, d=0):
        return max(self.first_subtree.max_depth(d + 1), self.second_subtree.max_depth(d + 1))

    def min_list_subtrees(self, depth=0):
        if depth >= self.depth_limit:
            return self
        return np.append(np.append([self], self.first_subtree.min_list_subtrees(depth+1)),
                         self.second_subtree.min_list_subtrees(depth+1))

    def set_subtree(self, subtree, first_subtree):
        if first_subtree:
            self.first_subtree = subtree
        else:
            self.second_subtree = subtree

    def print_gp_polynomial(self):
        return self.eval_polynomial("x")


class Tree_Chrom_Leaf:

    def __init__(self, x):
        self.value = x

    def eval_polynomial(self, x):
        if not(self.value == 0):
            return self.value
        return x

    def max_depth(self, d):
        return d

    def max_list_subtrees(self):
        return self

    def min_list_subtrees(self, depth):
        return []
