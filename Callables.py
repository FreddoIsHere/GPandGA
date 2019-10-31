from Population import Poly_Population
from math import log
import numpy as np


def gp_find_prime_polynomial(constructor, fitness_functions, num_populations=1, population_size=1000, test_interval=(0, 50), death_rate=0.95):
    print_hyperparameters(population_size, test_interval, death_rate)

    populations = np.array([constructor(fitness_functions, population_size=population_size, test_interval=test_interval)])
    for _ in range(1, num_populations):
        np.append(populations, [constructor(fitness_functions, population_size=population_size, test_interval=test_interval)])

    num_iterations = int(-log(population_size) / log(death_rate))
    i = 1
    while populations[0].polynomials.size > 1:
        printProgressBar(i, num_iterations)
        for p in populations:
            p.introduce_new_generation(death_rate=death_rate)
        if populations[0].polynomials.size < num_iterations/2 and populations.size > 1:
            for k in range(num_populations-1):
                populations[k].merge_populations(populations[k+1])
                np.delete(populations, k+1)
        i += 1
    printProgressBar(num_iterations, num_iterations)

    print("Surviving polynomials: ")
    for p in populations[0].polynomials:
        print("It produced {0} primes."
              .format(int(p.num_primes_fitness_in_interval(test_interval))))
        print(p.print_gp_polynomial())


def print_hyperparameters(population_size, test_interval, death_rate):
    print("Population-size: {0}, Test-interval: {1}, Death-rate: {2}"
          .format(population_size, test_interval, death_rate))

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 60, fill = '█', printEnd = "\r"):
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
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
