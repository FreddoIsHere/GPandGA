from enum import IntEnum
from abc import ABC, abstractmethod
import numpy as np
from gmpy2 import is_prime
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
        return np.sum([is_prime(int(x)) and x > 0 for x in results])

    def num_consecutive_primes_fitness_in_interval(self, interval):
        results = np.unique(self.eval_polynomial(np.array(range(interval[0], interval[1]))))
        longest_run = 0
        current_run = 0
        for x in results:
            if is_prime(int(x)) and x > 0:
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
        while is_prime(int(x)) and last_x != x and x > 0:
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


class Poly_Chrom(Chromosome):

    def __init__(self, values=None, order_upper_bound=2, coeffs_bound=(-50, 50)):
        super().__init__()
        self.coeffs_bound = coeffs_bound
        self.order_upper_bound = order_upper_bound+2
        if values is None:
            self.value = np.array(
                np.random.choice(range(coeffs_bound[0], coeffs_bound[1]), size=np.random.randint(1, self.order_upper_bound), replace=True))
        else:
            self.value = values

    def discrete_crossover(self, polychrom, mutation=0.01):
        min_order = min(polychrom.value.size, self.value.size)
        a = np.append(self.value, np.zeros(self.order_upper_bound - min_order))
        b = np.append(polychrom.value, np.zeros(self.order_upper_bound - min_order))
        child_values = np.zeros(self.order_upper_bound)
        for i in range(self.order_upper_bound):
            child_values[i] = np.random.choice([a[i], b[i]], p=[0.5, 0.5])
        new_values = self.mutate(mutation, child_values)
        return Poly_Chrom(values=new_values, order_upper_bound=self.order_upper_bound, coeffs_bound=self.coeffs_bound)

    def mutate(self, mutation, values):
        if np.random.choice([False, True], p=[1 - mutation, mutation]) and self.value.size > 0:
            values[np.random.randint(0, values.size - 1)] = np.random.randint(self.coeffs_bound[0], self.coeffs_bound[1])
            return np.trim_zeros(values, trim='b')
        else:
            return np.trim_zeros(values, trim='b')

    def eval_polynomial(self, x):
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
    return np.add(x, y)


def multiply(x, y):
    return np.multiply(x, y)


class Operator:

    def __init__(self, operator_functions):
        self.function = np.random.choice(operator_functions)

    def print_operator(self, x, y):
        return "{0}({1}, {2})".format(self.function.__name__, x, y)


class Tree_Chrom(Chromosome):

    def __init__(self, depth=0, depth_limit=2, coeffs_bound=(-50, 50), operator_functions=None):
        super().__init__()
        if operator_functions is None:
            operator_functions = [add, multiply]
        self.coeffs_bound = coeffs_bound
        self.depth_limit = depth_limit
        if depth == 2 * depth_limit - 1:
            self.first_subtree = np.random.choice([Tree_Terminal(coeffs_bound=coeffs_bound), Tree_Non_Terminal()], p=[0.5, 0.5])
            self.operator = Operator(operator_functions)
            self.second_subtree = np.random.choice([Tree_Terminal(coeffs_bound=coeffs_bound), Tree_Non_Terminal()], p=[0.5, 0.5])
        else:
            self.first_subtree = Tree_Chrom(operator_functions=operator_functions, depth=depth + 1, depth_limit=self.depth_limit, coeffs_bound=coeffs_bound)
            self.operator = Operator(operator_functions)
            self.second_subtree = Tree_Chrom(operator_functions=operator_functions, depth=depth + 1, depth_limit=self.depth_limit, coeffs_bound=coeffs_bound)

    def discrete_crossover(self, tree_chrom, mutation=0.01):
        bottom_up_subtrees = np.random.choice(
            [Tree_Chrom(coeffs_bound=self.coeffs_bound, depth_limit=self.depth_limit), np.random.choice(copy.deepcopy(tree_chrom).bottom_up_list_subtrees())],
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

    def __init__(self, x=None, coeffs_bound=(-50, 50)):
        if x is None:
            self.value = np.random.choice([i for i in range(coeffs_bound[0], coeffs_bound[1]) if x != 0])
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
