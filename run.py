from Callables import *
from Chromosomes import Fitness
from Population import Poly_Population, Tree_Population

'''gp_find_prime_polynomial(Poly_Population, [Fitness.number_of_consecutive_primes_in_interval,
                                           Fitness.number_of_primes_in_interval,
                                           Fitness.number_of_consecutive_primes],
                         num_populations=1, population_size=1000, test_interval=(0, 300), death_rate=0.95)'''

gp_find_prime_polynomial(Tree_Population, [Fitness.number_of_primes_in_interval],
                         num_populations=1, population_size=3, test_interval=(0, 100), death_rate=0.95)
