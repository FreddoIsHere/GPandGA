from Callables import *
from Chromosomes import Fitness


gp_find_prime_polynomial([Fitness.number_of_primes_in_interval,
                          Fitness.number_of_consecutive_primes_in_interval],
                         population_size=1000, test_interval=(0, 100), death_rate=0.95)
