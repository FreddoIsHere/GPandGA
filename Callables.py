from math import log
import numpy as np


def find_prime_polynomial(constructor, fitness_functions, num_populations=1, merge_point=0.5, population_size=500,
                          target_population_size=1,
                          test_interval=(0, 100),
                          birth_rate=0.9, mutation=0.01,
                          coeffs_bound=(-50, 50), constraint=3):
    # print used parameters
    print("Chromosome: {0}, Amount of populations: {1}, Population-size: {2}, Target-population-size: {3}, \n"
          "Test-interval: {4}, Birth-rate: {5}, Mutation-rate: {6}, Coeffs-bound: {7}, Constraint: {8}"
          .format(constructor.__name__, num_populations, population_size, target_population_size, test_interval,
                  birth_rate, mutation, coeffs_bound, constraint))
    # init populations
    populations = np.array(
        [constructor(fitness_functions, population_size=population_size, test_interval=test_interval,
                     birth_rate=birth_rate,
                     mutation=mutation, coeffs_bound=coeffs_bound, constraint=constraint)])
    for _ in range(1, num_populations):
        np.append(populations,
                  [constructor(fitness_functions, population_size=population_size, test_interval=test_interval,
                               birth_rate=birth_rate,
                               mutation=mutation, coeffs_bound=coeffs_bound, constraint=constraint)])
    # start evolution
    i = True
    while i:
        i = populations[0].introduce_new_generation()
    # add last average fitness
    fitnesses = [p.primes_fitness(populations[0].test_interval, populations[0].fitness_functions) for p in
                 populations[0].polynomials]
    populations[0].average_fitness_over_time.append(np.mean(fitnesses))
    return populations[0]
