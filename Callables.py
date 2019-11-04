from math import log
import numpy as np
import time,sys

def find_prime_polynomial(constructor, fitness_functions, num_populations=1, merge_point=0.5, population_size=500,
                          test_interval=(0, 100),
                          birth_rate=0.9, mutation=0.01,
                          coeffs_bound=(-50, 50), constraint=3):
    # print used parameters
    print("Chromosome: {0}, Amount of populations: {1}, Population-size: {2}, \n"
          "Test-interval: {3}, Birth-rate: {4}, Mutation-rate: {5}, Coeffs-bound: {6}, Constraint: {7}"
          .format(constructor.__name__, num_populations, population_size, test_interval,
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
    not_converged = True
    while not_converged:
        not_converged = False
        for p in populations:
            not_converged = not_converged or p.introduce_new_generation()
        i = populations.size
        if i > 1 and populations[0].polynomials.size < merge_point * population_size:
            for k in range(1, i):
                populations[0].merge_populations(populations[k])
                populations = np.delete(populations, k)
    # add last average fitness
    fitnesses = [p.primes_fitness(populations[0].test_interval, populations[0].fitness_functions) for p in
                 populations[0].polynomials]
    populations[0].average_fitness_over_time.append(np.mean(fitnesses))
    return populations[0]
