The GP and the GA can be run via the find_prime_polynomial()-function in Callables.py.
The function has the following parameters:
Constructor    :param constructor: 
list of Fitness enums    :param fitness_functions: Selectively choose between 3 functions (MSE has hard-coded labels, do not use for prime GP)
int    :param num_populations: Number of populations
float    :param merge_point: percentage for population size to shrink for population merges
int    :param population_size: number of trees/functions/polynomials in a population
(int, int)    :param test_interval: interval to test fitness on
float    :param birth_rate: replenishing rate
float    :param mutation: mutation rate
(int, int)    :param coeffs_bound: terminal/coefficient bounds
int    :param constraint: order/depth limit
list of functions    :param operator_functions: list of functions the GP can use to build trees
Tree_Population    :return: Object that contains fitness history and final polynomial/function/tree

The final tree can be accessed via population.polynomials[0]. It can be evaluated with its function called eval_polynomial(x), where x can be a vector or scalar. It is best to look at the examples in the jupyter notebook provided.
