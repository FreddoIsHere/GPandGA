from math import log
import numpy as np


def find_prime_polynomial(constructor, fitness_functions, num_populations=1, merge_point=0.5, population_size=500,
                          target_population_size=1,
                          test_interval=(0, 100), death_rate=0.9, mutation=0.01):
    # print used parameters
    print("Chromosome: {0}, Amount of populations: {1}, Population-size: {2}, Target-population-size: {3}, \n"
          "Test-interval: {4}, Death-rate: {5}, Mutation-rate: {6}"
          .format(constructor.__name__, num_populations, population_size, target_population_size, test_interval,
                  death_rate, mutation))
    # init populations
    populations = np.array(
        [constructor(fitness_functions, population_size=population_size, test_interval=test_interval,
                     death_rate=death_rate,
                     mutation=mutation)])
    for _ in range(1, num_populations):
        np.append(populations,
                  [constructor(fitness_functions, population_size=population_size, test_interval=test_interval,
                               death_rate=death_rate,
                               mutation=mutation)])
    # start evolution
    num_iterations = int(log(target_population_size / population_size) / log(death_rate))
    i = 1
    while populations[0].polynomials.size > target_population_size:
        print_progress_bar(i, num_iterations)
        for p in populations:
            p.introduce_new_generation()
        # merge populations upon reaching critical merge-point
        if populations[0].polynomials.size < merge_point * num_iterations and populations.size > 1:
            for k in range(num_populations - 1):
                populations[k].merge_populations(populations[k + 1])
                np.delete(populations, k + 1)
        i += 1
    print_progress_bar(num_iterations, num_iterations)
    # add last average fitness
    fitnesses = [p.primes_fitness(populations[0].test_interval, populations[0].fitness_functions) for p in
                 populations[0].polynomials]
    populations[0].average_fitness.append(np.mean(fitnesses))
    return populations[0]


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=80, fill='█', printEnd="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
