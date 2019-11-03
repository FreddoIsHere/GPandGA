from Population import Poly_Population
from math import log
import numpy as np


def gp_find_prime_polynomial(constructor, fitness_functions, num_populations=1, population_size=1000, target_population_size=1,
                             test_interval=(0, 50), death_rate=0.95, mutation=0.01):
    print_hyperparameters(constructor, num_populations, population_size, target_population_size,
                             test_interval, death_rate, mutation)

    populations = np.array(
        [constructor(fitness_functions, population_size=population_size, test_interval=test_interval, mutation=mutation)])
    for _ in range(1, num_populations):
        np.append(populations,
                  [constructor(fitness_functions, population_size=population_size, test_interval=test_interval, mutation=mutation)])

    num_iterations = int(log(target_population_size/population_size) / log(death_rate))
    i = 1
    while populations[0].polynomials.size > target_population_size:
        printProgressBar(i, num_iterations)
        for p in populations:
            p.introduce_new_generation(death_rate=death_rate)
        if populations[0].polynomials.size < num_iterations / 2 and populations.size > 1:
            for k in range(num_populations - 1):
                populations[k].merge_populations(populations[k + 1])
                np.delete(populations, k + 1)
        i += 1
    printProgressBar(num_iterations, num_iterations)

    print("Surviving polynomials: ")
    for p in populations[0].polynomials:
        print("It produced {0} distinct primes."
              .format(int(p.num_primes_fitness_in_interval(test_interval))))
        print(p.print_gp_polynomial())


def print_hyperparameters(constructor, num_populations, population_size, target_population_size,
                             test_interval, death_rate, mutation):
    print("Chromosome: {0}, Amount of populations: {1}, Population-size: {2}, Target-population-size: {3}, \n"
          "Test-interval: {4}, Death-rate: {5}, Mutation-rate: {6}"
          .format(constructor.__name__, num_populations, population_size, target_population_size, test_interval, death_rate, mutation))


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=60, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
