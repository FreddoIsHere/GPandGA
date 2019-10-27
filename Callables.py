from Population import Poly_Population

seed = 1234
population_size = 100
test_interval = (0, 50)
birth_rate = 0.95


def gp_find_prime_polynomial():
    print_hyperparameters()
    population = Poly_Population(population_size=population_size, test_interval=test_interval, seed=seed)

    while population.polynomials.size != 1:
        population.introduce_new_generation(birth_rate=birth_rate)
    poly_string = str(population.polynomials[0])
    for i in range(1, population.polynomials.size):
        if population.polynomials[i] != 0:
            poly_string = str(population.polynomials[i]) + "x^" + str(i) + poly_string
    print("The final polynomial is: ", poly_string)


def print_hyperparameters():
    print("Population-size: {0}, Test-interval: {1}, Birth-rate: {2}"
          .format(population_size, test_interval, birth_rate))