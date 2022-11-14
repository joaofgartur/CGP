import copy
import random
import numpy as np
import utils

MIN_INPUT = -1.0
MAX_INPUT = 1.0
MIN_OUTPUT = 0.0
MAX_OUTPUT = 255.0
FITNESS_MAX_VALUE = 100
OFFSET = 1  # offset due to difference in node representation -> [f, c1, c2]
LAMBDA = 50
ONE = 1
ZERO = 0
NODE_SIZE = 3
ARITY = 2

GENERATIONS_LOG_HEADER = ["generation", "worst_fitness", "best_fitness", "mean", "std", "best_overall", "genotype",
                          "active_nodes"]
GENERATIONS_LOG_FILE = "generations.csv"
LOG_HEADER = ["generation", "individual", "fitness", "genotype", "active_nodes"]
LOG_FILE = "log.csv"


class Individual:

    def __init__(self, configs):
        self.num_rows = configs['num_rows']
        self.num_columns = configs['num_columns']
        self.graph_length = self.num_rows * self.num_columns
        self.num_input = configs['num_input']
        self.num_output = configs['num_output']
        self.level_back = configs['level_back']
        self.num_functions = configs['num_functions']
        self.mutation_rate = configs['mutation_rate']
        self.function_genes_indexes = None
        self.genotype = None
        self.active_nodes = None
        self.genotype, self.function_genes_indexes = self.generate_genes()
        self.data = None
        self.fitness = 0.0

    def get_connection_range(self, j):
        if j >= self.level_back:
            min_range = self.num_input + (j - self.level_back) * self.num_rows
            max_range = self.num_input + j * self.num_rows - 1
        else:
            min_range = 0
            max_range = self.num_input + j * self.num_rows - 1

        return min_range, max_range

    def generate_genes(self):
        genes = np.zeros(ZERO, dtype=np.int)
        function_genes_indexes = np.zeros(ZERO, dtype=np.int)

        # generate connection nodes
        gene_index = 0
        for j in range(self.num_columns):
            for i in range(self.num_rows):
                node = np.zeros(ZERO, dtype=np.int)
                min_range, max_range = self.get_connection_range(j)

                # function gene
                function_gene = random.randint(0, self.num_functions)
                node = np.append(node, function_gene)
                function_genes_indexes = np.append(function_genes_indexes, gene_index)
                gene_index += 1

                # connection genes
                con_gene_1 = random.randint(min_range, max_range)
                node = np.append(node, con_gene_1)
                gene_index += 1

                con_gene_2 = random.randint(min_range, max_range)
                node = np.append(node, con_gene_2)
                gene_index += 1

                genes = np.append(genes, node)

        # generate output nodes
        for index in range(self.num_output):
            max_range = self.num_input + self.graph_length - 1
            output_gene = random.randint(0, max_range)
            genes = np.append(genes, output_gene)

        return genes, function_genes_indexes

    def count_genes_in_node(self, index):
        if index < self.num_input or index >= self.num_input + self.graph_length:
            return ONE
        else:
            return NODE_SIZE

    # mutate genotype
    def mutate(self):
        for j in range(self.num_columns):
            for i in range(self.num_rows):

                if random.random() < self.mutation_rate:
                    index = i * self.num_columns + j
                    if index in self.function_genes_indexes:
                        self.genotype[index] = random.randint(0, self.num_functions)
                    else:
                        min_range, max_range = self.get_connection_range(j)
                        self.genotype[index] = random.randint(min_range, max_range)

    # discover active nodes
    def nodes_to_process(self):
        M = self.num_columns * self.num_rows + self.num_input
        NU = [False for _ in range(M)]
        NP = []

        lg = self.genotype.size
        for i in range(lg - self.num_output, lg):
            NU[self.genotype[i]] = True

        for i in reversed(range(self.num_input, M)):
            if NU[i]:
                n_n = self.count_genes_in_node(i)
                index = n_n * (i - self.num_input)
                NG = []

                for j in range(0, n_n):
                    NG.append(self.genotype[index + j])

                for j in range(0, ARITY):
                    NU[NG[j + OFFSET]] = True

        n_u = 0
        for j in range(self.num_input, M):
            if NU[j]:
                NP.append(j)
                n_u += 1

        return n_u, NP

    # decode active nodes
    def decode(self, input_data, n_u, NP):
        num_rows, num_columns = input_data[0, :, :].shape
        num_values = self.num_input + self.num_rows * self.num_columns
        o = np.zeros((num_values, num_rows, num_columns))

        # record input values
        for i in range(0, self.num_input):
            o[i, :, :] = input_data[i, :, :]

        for j in range(0, n_u):

            # get node location in genotype
            n = NP[j] - self.num_input
            n_n = self.count_genes_in_node(NP[j])
            g = n_n * n

            # get connection genes
            in_array = np.zeros((num_values, num_rows, num_columns))
            for i in range(0, n_n - 1):
                in_array[i, :, :] = o[self.genotype[g + OFFSET + i], :, :]

            # get function gene
            function_gene = self.genotype[g]

            # calculate node output
            calculated_output = compute_function(in_array, function_gene)
            o[n + self.num_input, :, :] = calculated_output

        lg = self.genotype.size
        output = np.zeros((self.num_output, num_rows, num_columns))
        for j in range(0, self.num_output):
            output[j, :, :] = o[self.genotype[lg - self.num_output + j], :, :]

        return output


# compute function result
def compute_function(input_array, function):
    x = input_array[0]
    y = input_array[1]

    result = 0.0
    if function == 0:
        result = x
    elif function == 1:
        result = y
    elif function == 2:
        result = x * y
    elif function == 3:
        result = (np.sin(2 * np.pi * x) + np.cos(2 * np.pi * y)) * 0.5
    elif function == 4:
        result = (np.cos(2 * np.pi * x) + np.sin(2 * np.pi * y)) * 0.5
    elif function == 5:
        result = (np.cos(3 * np.pi * x) + np.sin(2 * np.pi * y)) * 0.5
    elif function == 6:
        result = np.tanh(x + y)
    elif function == 7:
        result = np.sin(np.pi * (x + y))
    elif function == 8:
        result = np.cos(np.pi * (x + y))
    elif function == 9:
        result = np.sqrt((np.power(x, 2) + np.power(y, 2)) * 0.5) - 0.5

    result = np.clip(result, -1.0, 1.0)

    return np.interp(result, [-1.0, 1.0], [0.0, 1.0])


def print_stats(statistics):
    print("\tWorst fitness: {:.4f}".format(statistics['worst_fitness']))
    print("\tBest fitness: {:.4f}".format(statistics['best_fitness']))
    print("\tMean: {:.4f}".format(statistics['mean']))
    print("\tStd: {:.4f}".format(statistics['std']))
    print("\tBest overall: {:.4f}".format(statistics['best_overall']))


def population_statistics(output_folder, generation_folder, generation, population, best, parentStats=False,
                          parent=None):
    fitness_array = np.zeros(0, dtype=np.float32)

    log_file = generation_folder + "/" + LOG_FILE
    utils.write_to_file(log_file, LOG_HEADER)

    # write log for parent
    if parentStats:
        fitness_array = np.append(fitness_array, parent.fitness)
        parent_stats = [generation, "parent", parent.fitness, parent.genotype, parent.active_nodes]
        utils.write_to_file(log_file, parent_stats)

    # write log for individuals
    for i in range(len(population)):
        individual = population[i]
        fitness_array = np.append(fitness_array, individual.fitness)
        individual_stats = [generation, i, individual.fitness, individual.genotype, individual.active_nodes]
        utils.write_to_file(log_file, individual_stats)

    # write log for generation
    generations_file = output_folder + "/" + GENERATIONS_LOG_FILE
    mean_fitness = np.mean(fitness_array)
    std_fitness = np.std(fitness_array)
    worst_fitness = np.min(fitness_array)
    best_fitness = np.max(fitness_array)
    generation_stats = [generation, worst_fitness, best_fitness, mean_fitness, std_fitness, best.fitness,
                        best.genotype, best.active_nodes]
    utils.write_to_file(generations_file, generation_stats)

    generation_dict = {
        "worst_fitness": worst_fitness,
        "best_fitness": best_fitness,
        "mean": mean_fitness,
        "std": std_fitness,
        "best_overall": best.fitness
    }

    return generation_dict


def express_phenotype(individual, data):
    # discover active nodes of individual
    n_u, NP = individual.nodes_to_process()

    num_rows, num_columns = np.shape(data[:, :, 0])

    # prepare inputs
    x_values = get_input_matrix(MIN_INPUT, MAX_INPUT, num_rows, num_columns, 0)
    y_values = get_input_matrix(MIN_INPUT, MAX_INPUT, num_rows, num_columns, 3)
    input_data = np.array((x_values, y_values))

    # obtain output
    output = individual.decode(input_data, n_u, NP)
    output_data = np.zeros((num_rows, num_columns, individual.num_output))
    for k in range(individual.num_output):
        output_data[:, :, k] = np.interp(output[k, :, :], [0.0, 1.0], [0.0, 255.0])

    individual.data = output_data

    return NP


def get_input_matrix(min_value, max_value, num_rows, num_columns, num_rotations):
    matrix = np.linspace(min_value, max_value, num=num_rows)
    matrix = np.resize(matrix, num_rows * num_columns)
    matrix = np.reshape(matrix, (num_rows, num_columns))
    matrix = np.rot90(matrix, k=num_rotations)

    return matrix


def copy_individual(individual_to_copy):
    copied_individual = copy.deepcopy(individual_to_copy)
    copied_individual.fitness = 0.0
    copied_individual.active_nodes = np.array(individual_to_copy.active_nodes)
    copied_individual.genotype = np.array(individual_to_copy.genotype)

    return copied_individual


def select_parent(population, parent):
    if parent is None:
        max_fitness = 0
    else:
        max_fitness = parent.fitness

    individual_index = 0
    parent_index = 0
    new_parent = False
    for individual in population:
        # print("\t[INDIVIDUAL " + str(individual_index) + "] Fitness: " + "{:.4f}".format(individual.fitness))
        if individual.fitness >= max_fitness:
            parent = copy.deepcopy(individual)
            max_fitness = parent.fitness
            new_parent = True
            parent_index = individual_index
        individual_index += 1

    if new_parent:
        print("\t[PARENT]: Individual " + str(parent_index) + " selected as parent")
    else:
        print("\t[PARENT]: Parent remains the same")

    return parent


def generate(configs, save_folder, fitness_function, input_data):
    max_generation = configs.get('max_generation')

    output_folder = save_folder + "/" + utils.get_current_timestamp()
    utils.create_directory(output_folder)
    generations_file = output_folder + "/" + GENERATIONS_LOG_FILE
    utils.write_to_file(generations_file, GENERATIONS_LOG_HEADER)
    best_folder = output_folder + "/best_individuals"
    utils.create_directory(best_folder)
    generation_folder = output_folder + "/generation_0"
    utils.create_directory(generation_folder)

    print("[GENERATION " + str(ZERO) + "] Lifecycle")

    # create first generation and express phenotype of individuals
    population = []
    for i in range(ONE + LAMBDA):
        individual = Individual(configs)
        active_nodes = express_phenotype(individual, input_data)  # genotype -> genotype
        individual.active_nodes = np.array(active_nodes)
        population.append(individual)

    if configs['export_individuals']:
        utils.export_images(population, generation_folder)

    # evaluate first generation
    population = fitness_function(population)

    # select parent
    parent = select_parent(population, None)

    # get statistics
    statistics = population_statistics(output_folder, generation_folder, ZERO, population, best=parent)
    print_stats(statistics)

    if configs['export_individuals']:
        utils.save_img(best_folder, ZERO, parent.data)

    # evolve
    for generation in range(1, max_generation):
        print("[GENERATION " + str(generation) + "] Lifecycle")

        generation_folder = output_folder + "/generation_" + str(generation)
        utils.create_directory(generation_folder)

        # export image of parent
        if configs['export_individuals']:
            utils.save_img(generation_folder, "parent", parent.data)

        # create new generation and express phenotype of individuals
        population = []
        for i in range(LAMBDA):
            offspring = copy_individual(parent)
            offspring.mutate()
            active_nodes = express_phenotype(offspring, input_data)  # genotype -> genotype
            offspring.active_nodes = np.array(active_nodes)
            population.append(offspring)

        # export images of individuals
        if configs['export_individuals']:
            utils.export_images(population, generation_folder)

        # evaluate generation
        population = fitness_function(population)

        # select parent
        # print("\t[PARENT]: Fitness " + "{:.4f}".format(parent.fitness))
        parent = select_parent(population, parent)
        if configs['export_individuals']:
            utils.save_img(best_folder, generation, parent.data)

        statistics = population_statistics(output_folder, generation_folder, generation, population,
                                           best=parent, parentStats=True, parent=parent)
        print_stats(statistics)

    return population, statistics
