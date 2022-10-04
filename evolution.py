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

CSV_HEADER = ["generation", "max_fitness", "min_fitness", "mean", "std", "genotype", "active_nodes"]
INDIVIDUAL_CSV_HEADER = ["generation", "individual", "fitness", "genotype", "active_nodes"]


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
        self.genotype, self.function_genes_indexes = self.generate_genes()
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
        genes = []
        function_genes_indexes = []

        # generate connection nodes
        node_index = self.num_input
        gene_index = 0
        for j in range(self.num_columns):
            for i in range(self.num_rows):
                min_range, max_range = self.get_connection_range(j)

                function_gene = random.randint(0, self.num_functions)
                genes.append(function_gene)
                function_genes_indexes.append(gene_index)  # store function gene position; used in mutation
                gene_index += 1

                con_gene_1 = random.randint(min_range, max_range)
                genes.append(con_gene_1)
                gene_index += 1

                con_gene_2 = random.randint(min_range, max_range)
                genes.append(con_gene_2)
                gene_index += 1

                node_index += 1

        # generate output nodes
        for index in range(self.num_output):
            max_range = self.num_input + self.graph_length - 1
            output_gene = random.randint(0, max_range)
            genes.append(output_gene)

        return genes, function_genes_indexes

    def count_genes_in_node(self, index):
        if index < self.num_input or index >= self.num_input + self.graph_length:
            return 1
        else:
            return 3

    # mutate genotype
    def mutate(self):
        for j in range(self.num_columns):
            for i in range(self.num_rows):

                if random.random() < self.mutation_rate:
                    index = i * self.num_rows + j
                    if index in self.function_genes_indexes:
                        self.genotype[index] = random.randint(0, self.num_functions)
                    else:
                        min_range, max_range = self.get_connection_range(j)
                        self.genotype[index] = random.randint(min_range, max_range)

    # determine coding nodes
    def nodes_to_process(self):
        M = self.num_columns * self.num_rows + self.num_input
        NU = [False for _ in range(M)]
        NP = []

        lg = len(self.genotype)
        for i in range(lg - self.num_output, lg):
            NU[self.genotype[i]] = True

        for i in reversed(range(self.num_input, M)):
            if NU[i]:
                n_n = self.count_genes_in_node(i)
                index = n_n * (i - self.num_input)
                NG = []

                for j in range(0, n_n):
                    NG.append(self.genotype[index + j])

                for j in range(0, arity(True)):
                    NU[NG[j + 1]] = True

        n_u = 0
        for j in range(self.num_input, M):
            if NU[j]:
                NP.append(j)
                n_u += 1

        return n_u, NP

    # decode active nodes
    def decode(self, input_data, n_u, NP):
        o = [0 for _ in range(self.num_input + self.num_rows * self.num_columns)]

        # record pixel value on both inputs
        for i in range(0, self.num_input):
            o[i] = input_data[i]

        for j in range(0, n_u):

            # get node location in genotype
            n = NP[j] - self.num_input
            n_n = self.count_genes_in_node(NP[j])
            g = n_n * n

            # get connection genes

            in_array = []
            for i in range(0, n_n - 1):
                in_array.append(o[self.genotype[g + OFFSET + i]])

            # get function gene
            function_gene = self.genotype[g]

            # calculate node output
            calculated_output = compute_function(in_array, function_gene)
            if calculated_output < 0:
                print(str(function_gene) + " -> negative")
            o[n + self.num_input] = calculated_output

        lg = len(self.genotype)
        output = [0 for _ in range(self.num_output)]
        for j in range(0, self.num_output):
            output[j] = o[self.genotype[lg - self.num_output + j]]

        return output

    def evaluate(self, generation, individual):

        fitness = 0
        needs_input = True
        while needs_input:
            print("[FITNESS] Evaluate individual " + str(individual) + " from generation " + str(generation)
                  + " from 0 to " + str(FITNESS_MAX_VALUE) + ". The image is " + str(generation)
                  + "_" + str(individual) + ".png")

            while True:
                try:
                    fitness = float(input('[FITNESS]: '))
                except ValueError:
                    continue
                else:
                    break

            if 0 <= fitness <= 100:
                needs_input = False

        self.fitness = fitness / FITNESS_MAX_VALUE

        return self.fitness

    def evaluate_fitness(self, img_data, output_path, generation, index):
        n_u, NP = self.nodes_to_process()

        num_rows, num_columns = np.shape(img_data[:, :, 0])
        output_img = np.zeros(np.shape(img_data))

        x_values = np.linspace(-1, 1.0, num=num_columns)
        y_values = np.linspace(-1, 1.0, num=num_rows)
        for i in range(len(x_values)):
            for j in range(len(x_values)):
                x = x_values[i]
                y = y_values[i]
                input_data = np.array([x, y])
                output = self.decode(input_data, n_u, NP)
                for k in range(self.num_output):
                    output_img[i, j, k] = output[k]

        utils.save_img(output_path, generation, index, output_img)

        return self.evaluate(generation, index), NP

    # return shallow copy of the individual
    def new(self):
        new = copy.copy(self)
        new.genotype = list(self.genotype)
        new.function_genes_indexes = list(self.function_genes_indexes)

        return new


# compute function result
def compute_function(input_array, function):
    x = input_array[0]
    y = input_array[1]

    result = 0
    if function == 0:
        result = np.interp(x, [MIN_INPUT, MAX_INPUT], [MIN_OUTPUT, MAX_OUTPUT])
    elif function == 1:
        result = np.interp(y, [MIN_INPUT, MAX_INPUT], [MIN_OUTPUT, MAX_OUTPUT])
    elif function == 2:
        result = np.sqrt(abs(x + y))
    elif function == 3:
        result = np.sqrt(abs(x - y))
    elif function == 4:
        result = 255 * (abs(np.sin(2 * np.pi * x / 255) + np.cos(2 * np.pi * x / 255))) / 2
    elif function == 5:
        result = 255 * (abs(np.cos(2 * np.pi * x / 255) + np.sin(2 * np.pi * x / 255))) / 2
    elif function == 6:
        result = 255 * (abs(np.cos(3 * np.pi * x / 255) + np.sin(2 * np.pi * x / 255))) / 2
    elif function == 7:
        result = np.exp(x + y) % 256
    elif function == 8:
        result = abs(np.sinh(x + y)) % 256
    elif function == 9:
        result = np.cosh(x + y) % 256
    elif function == 10:
        result = 255 * abs(np.tanh(x + y))
    elif function == 11:
        result = 255 * abs(np.sin(np.pi * (x + y) / 255))
    elif function == 12:
        result = 255 * abs(np.cos(np.pi * (x + y) / 255))
    elif function == 13:
        result = 255 * abs(np.tan(np.pi * (x + y) / (255 * 8)))
    elif function == 14:
        result = np.sqrt((pow(x, 2) + pow(y, 2)) / 2)
    elif function == 15:
        result = x * y / 255
    elif function == 16:
        result = abs(x + y) % 256
    elif function == 17:
        result = abs(x - y) % 256

    # print(result)
    return result


def arity(is_node):
    if is_node:
        return 2


def population_statistics(generation, parent, fitness_array, active_nodes_array):
    best_fitness_index = np.argmax(fitness_array)
    worst_fitness_index = np.argmin(fitness_array)
    fitness_mean = np.mean(fitness_array)
    fitness_std = np.std(fitness_array)

    parent_active_nodes = active_nodes_array[best_fitness_index]

    return [generation, fitness_array[best_fitness_index], fitness_array[worst_fitness_index], fitness_mean,
            fitness_std, utils.list_to_str(parent.genotype), utils.list_to_str(parent_active_nodes)]


def select_fittest(output_path, csv_path, individual_csv_path, img_data, generation, population):
    max_fitness = 0
    parent = None
    individual_index = 0
    parent_index = 0
    population_active_nodes = []
    population_fitness = []

    print("[GENERATION " + str(generation) + "] Choosing first parent")
    for individual in population:

        # evaluate individual
        fitness, individual_active_nodes = individual.evaluate_fitness(img_data, output_path, generation,
                                                                       individual_index)
        population_fitness.append(fitness)
        population_active_nodes.append(individual_active_nodes)

        individual_data = [generation, individual_index, fitness, individual.genotype, individual_active_nodes]
        utils.write_to_csv(individual_csv_path, individual_data)

        print("\t[INDIVIDUAL " + str(individual_index) + "] Fitness: " + "{:.4f}".format(fitness))

        if fitness > max_fitness:
            max_fitness = fitness
            parent = individual
            parent_index = individual_index

        individual_index += 1

    print("[GENERATION " + str(generation) + "] Individual " + str(parent_index) + " is the parent")

    data = population_statistics(generation, parent, population_fitness, population_active_nodes)
    utils.write_to_csv(csv_path, data)

    return parent


def generate(configs, input_img):
    max_generation = configs.get('max_generation')
    lambda_arg = configs['lambda_arg']
    generation = 0
    population = []

    output_path = "outputs/" + utils.get_current_timestamp()
    csv_path = output_path + "/log.csv"
    individuals_csv_path = output_path + "/individuals_log.csv"
    utils.create_output_folder(output_path)
    utils.write_to_csv(csv_path, CSV_HEADER)
    utils.write_to_csv(individuals_csv_path, INDIVIDUAL_CSV_HEADER)

    print("[OUTPUT] Generated images' path: " + output_path)

    # create first generation
    for i in range(1 + lambda_arg):
        individual = Individual(configs, )
        population.append(individual)

    # select first parent
    parent = select_fittest(output_path, csv_path, individuals_csv_path, input_img, generation, population)
    generation += 1

    # evolve
    while generation < max_generation:
        print("[GENERATION " + str(generation) + "] Evolving")

        population = []
        population_active_nodes = []
        population_fitness = []
        for i in range(lambda_arg):
            offspring = parent.new()
            offspring.mutate()
            population.append(offspring)

        print("\t[PARENT] Fitness: " + "{:.4f}".format(parent.fitness))

        index = 0
        parent_index = 0
        new_parent = False
        for individual in population:
            fitness, individual_active_nodes = individual.evaluate_fitness(input_img, output_path, generation, index)

            print("\t[INDIVIDUAL " + str(index) + "] Fitness: " + "{:.4f}".format(fitness))

            population_fitness.append(fitness)
            population_active_nodes.append(individual_active_nodes)

            individual_data = [generation, index, fitness, individual.genotype, individual_active_nodes]
            utils.write_to_csv(individuals_csv_path, individual_data)

            if fitness >= parent.fitness:
                new_parent = True
                parent_index = index
                parent = individual

            index += 1

        if new_parent:
            print("\t[PARENT] Individual " + str(parent_index) + " is the new parent.")
        else:
            print("\t[PARENT] Parent remains the same")

        data = population_statistics(generation, parent, population_fitness, population_active_nodes)
        utils.write_to_csv(csv_path, data)

        generation += 1
