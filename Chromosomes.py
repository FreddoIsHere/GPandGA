import random
from enum import IntEnum
from abc import ABC, abstractmethod
import numpy as np
from gmpy import is_prime
import copy


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
        results = np.unique(self.eval_polynomial(np.array(range(interval[0], interval[1]))))
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
        x = self.eval_polynomial(np.array([i]))[0]
        while is_prime(int(x)) == 2 and last_x != x:
            i += 1
            last_x = x
            x = self.eval_polynomial(np.array([i]))[0]
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

    def __init__(self, values=None, coeffs_bound=50, order_upper_bound=3):
        super().__init__()
        self.coeffs_bound = coeffs_bound
        self.order_upper_bound = order_upper_bound
        if values is None:
            self.value = np.array(
                random.sample(range(-coeffs_bound, coeffs_bound), random.randint(1, order_upper_bound)))
        else:
            self.value = values

    def discrete_crossover(self, polychrom, mutation=0.01):
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
        if np.random.choice([False, True], p=[1 - mutation, mutation]) and self.value.size > 0:
            values[random.randint(0, values.size - 1)] = random.randint(-self.coeffs_bound, self.coeffs_bound)
            return np.trim_zeros(values, trim='b')
        else:
            return np.trim_zeros(values, trim='b')

    def smooth_crossover(self, poly_chrom, mutation=0.01):
        max_order = max(poly_chrom.value.size, self.value.size)
        min_order = min(poly_chrom.value.size, self.value.size)
        a = np.append(self.value, np.zeros(self.order_upper_bound - min_order))
        b = np.append(poly_chrom.value, np.zeros(self.order_upper_bound - min_order))
        child_values = np.zeros(max_order)
        for i in range(max_order):
            child_values[i] = int((a[i] + b[i]) / 2)
        new_values = self.mutate(mutation, child_values)
        return PolyChrom(values=new_values)

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

class Operator:

    def __init__(self):
        functions = [self.add, self.multiply]
        self.function = np.random.choice(functions, p=[1 / 2, 1 / 2])

    def add(self, x, y):
        return np.add(x, y)

    def multiply(self, x, y):
        return np.multiply(x, y)

    def mod(self, x, y):
        return np.mod(x, y)

    def print_operator(self, x, y):
        switch = {
            "add": "({0} + {1})".format(x, y),
            "multiply": "{0} * {1}".format(x, y),
            "mod": "({0} % {1})".format(x, y)
        }
        return switch.get(self.function.__name__, " Invalid operator ")


class Tree_Chrom(Chromosome):

    def __init__(self, depth=0, depth_limit=2):
        super().__init__()
        self.depth_limit = depth_limit
        if depth == 2 * depth_limit - 1:
            self.first_subtree = np.random.choice([Tree_Terminal(), Tree_Non_Terminal()], p=[0.5, 0.5])
            self.operator = Operator()
            self.second_subtree = np.random.choice([Tree_Terminal(), Tree_Non_Terminal()], p=[0.5, 0.5])
        else:
            self.first_subtree = Tree_Chrom(depth + 1)
            self.operator = Operator()
            self.second_subtree = Tree_Chrom(depth + 1)

    def discrete_crossover(self, tree_chrom, mutation=0.01):
        bottom_up_subtrees = np.random.choice(
            [Tree_Chrom(), np.random.choice(copy.deepcopy(tree_chrom).bottom_up_list_subtrees())],
            p=[mutation, 1 - mutation])
        np.random.choice(self.top_down_list_subtrees()).set_subtree(bottom_up_subtrees, np.random.choice([True, False]))
        return copy.deepcopy(self)

    def eval_polynomial(self, x):
        return self.operator.function(self.first_subtree.eval_polynomial(x), self.second_subtree.eval_polynomial(x))

    def bottom_up_list_subtrees(self):
        if self.max_depth() > self.depth_limit:
            return np.append(self.first_subtree.bottom_up_list_subtrees(),
                             self.second_subtree.bottom_up_list_subtrees())
        return self.top_down_list_subtrees()

    def max_depth(self, depth=0):
        if depth > self.depth_limit:
            return depth
        return max(self.first_subtree.max_depth(depth + 1), self.second_subtree.max_depth(depth + 1))

    def top_down_list_subtrees(self, depth=0):
        if depth >= self.depth_limit:
            return self
        return np.append(np.append([self], self.first_subtree.top_down_list_subtrees(depth + 1)),
                         self.second_subtree.top_down_list_subtrees(depth + 1))

    def set_subtree(self, subtree, first_subtree):
        subtree = self.collapse_subtree(subtree)
        if first_subtree:
            self.first_subtree = subtree
        else:
            self.second_subtree = subtree

    def collapse_subtree(self, subtree):
        if isinstance(subtree, Tree_Chrom):
            if isinstance(subtree.first_subtree, Tree_Terminal) and isinstance(subtree.second_subtree, Tree_Terminal):
                subtree = Tree_Terminal(subtree.operator.function(subtree.first_subtree.value, subtree.second_subtree.value))
        return subtree

    def print_gp_polynomial(self):
        return self.operator.print_operator(self.first_subtree.print_gp_polynomial(),
                                            self.second_subtree.print_gp_polynomial())


class Tree_Terminal:

    def __init__(self, x=None):
        if x is None:
            self.value = random.choice([i for i in range(-50, 50) if x != 0])
        else:
            self.value = x

    def eval_polynomial(self, x):
        return np.tile(self.value, x.size)

    def max_depth(self, d):
        return d

    def bottom_up_list_subtrees(self):
        return self

    def top_down_list_subtrees(self, depth):
        return []

    def print_gp_polynomial(self):
        return str(self.value)


class Tree_Non_Terminal:

    def eval_polynomial(self, x):
        return x

    def max_depth(self, d):
        return d

    def bottom_up_list_subtrees(self):
        return self

    def top_down_list_subtrees(self, depth):
        return []

    def print_gp_polynomial(self):
        return "x"
