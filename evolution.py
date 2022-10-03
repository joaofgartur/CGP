import copy
import random
import numpy as np
import main


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

    def generate_genes(self):
        genes = []
        function_genes_indexes = []

        # generate connection nodes
        node_index = self.num_input
        gene_index = 0
        for j in range(self.num_columns):
            for i in range(self.num_rows):
                if j >= self.level_back:
                    min_range = self.num_input + (j - self.level_back) * self.num_rows
                    max_range = self.num_input + j * self.num_rows - 1
                else:
                    min_range = 0
                    max_range = self.num_input + j * self.num_rows - 1

                function_gene = random.randint(0, self.num_functions)
                genes.append(function_gene)
                function_genes_indexes.append(gene_index)
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
                        if j >= self.level_back:
                            min_range = self.num_input + (j - self.level_back) * self.num_rows
                            max_range = self.num_input + j * self.num_rows - 1
                        else:
                            min_range = 0
                            max_range = self.num_input + j * self.num_rows - 1

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
    def decode(self, input_data, n_u, NP, x, y):
        o = [0 for _ in range(self.num_input + self.num_rows * self.num_columns)]

        # record pixel value on both inputs
        # for i in range(0, self.num_input):
        #    o[i] = input_data[x, y, i]

        o[0] = x
        o[1] = y

        for j in range(0, n_u):

            # get node location in genotype
            n = NP[j] - self.num_input
            n_n = self.count_genes_in_node(NP[j])
            g = n_n * n

            # get connection genes
            # offset due to difference in node representation -> [f, c1, c2]
            in_array = []
            offset = 1
            for i in range(0, n_n - 1):
                in_array.append(o[self.genotype[g + offset + i]])

            # get function gene
            function_gene = self.genotype[g]

            # calculate node output
            calculated_output = compute_function(in_array, function_gene)
            o[n + self.num_input] = calculated_output

        lg = len(self.genotype)
        output = [0 for _ in range(self.num_output)]
        for j in range(0, self.num_output):
            output[j] = o[self.genotype[lg - self.num_output + j]]

        return output

    def evaluate(self):
        self.fitness = random.uniform(0.0, 1.0)
        return self.fitness

    def evaluate_fitness(self, img_data):
        n_u, NP = self.nodes_to_process()

        num_rows, num_columns = np.shape(img_data[:, :, 0])
        output_img = np.zeros(np.shape(img_data))

        for x in range(num_rows):
            for y in range(num_columns):
                output = self.decode(img_data, n_u, NP, x, y)
                for i in range(self.num_output):
                    output_img[x, y, i] = output[i]

        return self.evaluate(), output_img

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

    if x is None:
        x = 255
    if y is None:
        y = 255

    if function == 0:
        return x
    elif function == 1:
        return y
    elif function == 2:
        return np.sqrt(x + y)
    elif function == 3:
        return np.sqrt(abs(x - y))
    elif function == 4:
        return 255 * (abs(np.sin(2 * np.pi * x / 255) + np.cos(2 * np.pi * x / 255))) / 2
    elif function == 5:
        return 255 * (abs(np.cos(2 * np.pi * x / 255) + np.sin(2 * np.pi * x / 255))) / 2
    elif function == 6:
        return 255 * (abs(np.cos(3 * np.pi * x / 255) + np.sin(2 * np.pi * x / 255))) / 2
    elif function == 7:
        return np.exp(x + y) % 256
    elif function == 8:
        return abs(np.sinh(x + y)) % 256
    elif function == 9:
        return np.cosh(x + y) % 256
    elif function == 10:
        return 255 * abs(np.tanh(x + y))
    elif function == 11:
        return 255 * abs(np.sin(np.pi * (x + y) / 255))
    elif function == 12:
        return 255 * abs(np.cos(np.pi * (x + y) / 255))
    elif function == 13:
        return 255 * abs(np.tan(np.pi * (x + y) / (255 * 8)))
    elif function == 14:
        return np.sqrt((pow(x, 2) + pow(y, 2)) / 2)
    elif function == 15:
        return x * y / 255
    elif function == 16:
        return abs(x + y) % 256
    elif function == 17:
        return abs(x - y) % 256


def arity(is_node):
    if is_node:
        return 2


def select_fittest(output_path, img_data, generation, population):
    max_fitness = 0
    parent = None
    individual_index = 0
    parent_index = 0

    print("[GENERATION " + str(generation) + "] Choosing first parent")
    for individual in population:
        # evaluate individual
        fitness, evolved_img = individual.evaluate_fitness(img_data)
        main.save_img(output_path, generation, individual_index, evolved_img)

        print("\t[INDIVIDUAL " + str(individual_index) + "] Fitness: " + "{:.4f}".format(fitness))

        if fitness > max_fitness:
            max_fitness = fitness
            parent = individual
            parent_index = individual_index

        individual_index += 1

    print("[GENERATION " + str(generation) + "] Individual " + str(parent_index) + " is the parent")

    return parent


def generate(configs, input_img):
    max_generation = configs.get('max_generation')
    lambda_arg = configs['lambda_arg']
    generation = 0
    population = []
    output_path = "outputs/" + main.get_current_timestamp()
    main.create_output_folder(output_path)

    # create first generation
    for i in range(1 + lambda_arg):
        individual = Individual(configs, )
        population.append(individual)

    # select first parent
    parent = select_fittest(output_path, input_img, generation, population)
    generation += 1

    # evolve
    while generation < max_generation:
        print("[GENERATION " + str(generation) + "] Evolving")

        population = []
        for i in range(lambda_arg):
            offspring = parent.new()
            offspring.mutate()
            population.append(offspring)

        print("\t[PARENT] Fitness: " + "{:.4f}".format(parent.fitness))

        index = 0
        parent_index = 0
        new_parent = False
        for individual in population:
            fitness, evolved_img = individual.evaluate_fitness(input_img)
            main.save_img(output_path, generation, index, evolved_img)

            print("\t[INDIVIDUAL " + str(index) + "] Fitness: " + "{:.4f}".format(fitness))

            if fitness >= parent.fitness:
                new_parent = True
                parent_index = index
                parent = individual

            index += 1

        if new_parent:
            print("\t[PARENT] Individual " + str(parent_index) + " is the new parent.")
        else:
            print("\t[PARENT] Parent remains the same")
        generation += 1
