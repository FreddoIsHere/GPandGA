from Population import Poly_Population
from math import log


def gp_find_prime_polynomial(fitness_functions, population_size=1000, test_interval=(0, 50), death_rate=0.95):
    print_hyperparameters(population_size, test_interval, death_rate)
    population = Poly_Population(fitness_functions, population_size=population_size, test_interval=test_interval)

    num_iterations = int(-log(population_size) / log(death_rate))
    i = 1
    while population.polynomials.size > 1:
        printProgressBar(i, num_iterations)
        population.introduce_new_generation(death_rate=death_rate)
        i += 1
    printProgressBar(num_iterations, num_iterations)


    print("Surviving polynomials: ")
    for p in population.polynomials:
        print_gp_find_prime_polynomials(p, test_interval)


def print_gp_find_prime_polynomials(polynomial, test_interval):
    poly_string = "(" + str(polynomial.value[0]) + ")"
    for i in range(1, polynomial.value.size):
        if polynomial.value[i] != 0:
            poly_string = "(" + str(polynomial.value[i]) + ")" + "x^" + str(i) + " + " + poly_string
    print(poly_string)
    print("On given interval it produced {0} primes."
          .format(int(polynomial.num_primes_fitness(test_interval))))


def print_hyperparameters(population_size, test_interval, death_rate):
    print("Population-size: {0}, Test-interval: {1}, Death-rate: {2}"
          .format(population_size, test_interval, death_rate))

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
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
