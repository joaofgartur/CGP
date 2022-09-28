import random

import numpy as np

import main


class Individual:

    def __init__(self, configs):
        self.num_rows = configs['num_rows']
        self.num_columns = configs['num_columns']
        self.arity = configs['arity']
        self.graph_length = self.num_rows * self.num_columns
        self.num_input = configs['num_input']
        self.num_output = configs['num_output']
        self.level_back = configs['level_back']
        self.num_functions = configs['num_functions']
        self.genotype = None
        self.genotype = self.generate_genes()
        self.genes_per_node = []
        self.fitness = 0.0

    def generate_genes(self):
        genes = []

        # generate connection nodes
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

                con_gene_1 = random.randint(min_range, max_range)
                genes.append(con_gene_1)

                con_gene_2 = random.randint(min_range, max_range)
                genes.append(con_gene_2)

        # generate output nodes
        for index in range(self.num_output):
            max_range = self.num_input + self.graph_length - 1
            output_gene = random.randint(0, max_range)
            genes.append(output_gene)

        return genes

    def print_genotype(self):
        if self.genotype is None:
            print('Individual\'s genotype not mapped\n')
        else:
            print(self.genotype)

    def set_fitness(self, fitness):
        self.fitness = fitness

    def count_genes_in_node(self, index):
        if index < self.num_input or index >= self.num_input + self.graph_length:
            return 1
        else:
            return 3

    def mutate(self):
        return ""

    # appears to be correct
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

                for j in range(0, arity(NG[n_n - 1], True)):
                    NU[NG[j + 1]] = True

        n_u = 0
        for j in range(self.num_input, M):
            if NU[j]:
                NP.append(j)
                n_u += 1

        return n_u, NP

    def decode(self, input_data, n_u, NP, x, y):
        o = []

        """
        # record pixel value on both inputs
        for i in range(0, self.num_input):
            o.append(input_data[x][y])
        """
        o.append(x)
        o.append(y)

        for j in range(0, n_u):
            n = NP[j] - self.num_input
            n_n = self.count_genes_in_node(NP[j])
            g = n_n * n

            in_array = []
            for i in range(0, n_n - 1):
                in_array.append(o[self.genotype[g + i]])

            function_gene = self.genotype[g + n_n - 1]
            o.append(compute_function(x, y, function_gene))

        lg = len(self.genotype)
        output = []
        for j in range(0, self.num_output):
            output.append(o[self.genotype[lg - self.num_output + j]])

        return output

    def evaluate(self):
        return random.uniform(0.0, 1.0)

    def evaluate_fitness(self, img_data):
        n_u, NP = self.nodes_to_process()

        num_rows, num_columns = np.shape(img_data[:, :])
        output_img = np.zeros(np.shape(img_data))

        for x in range(num_rows):
            for y in range(num_columns):
                output = self.decode(img_data, n_u, NP, x, y)
                for value in output:
                    output_img[x][y] = value

        return self.evaluate(), output_img


def compute_function(x, y, function):
    return x * y


def arity(function, is_node):
    if is_node:
        return 2


def select_fittest(img_data, generation, population):
    max_fitness = 0
    parent = None

    individual_index = 0
    for individual in population:
        print("Evaluating individual " + str(individual_index) + " from generation " + str(generation))
        fitness, evolved_img = individual.evaluate_fitness(img_data)
        main.save_img(generation, individual_index, evolved_img)
        individual_index += 1
        if fitness > max_fitness:
            max_fitness = fitness
            parent = individual

    return parent


def generate(configs, input_img):
    generation = 0
    population = []
    for i in range(1 + configs['lambda_arg']):
        individual = Individual(configs, )
        population.append(individual)

    parent = select_fittest(input_img, generation, population)
    print("1st generation parent:\n\t")
    print(parent.genotype)

    return ""
