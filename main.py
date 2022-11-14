import argparse
import sys

import utils
import numpy as np
import random
from evolution import ZERO, LAMBDA, generate


def fitness_function(population):
    """
    It takes a population of individuals, asks the user to evaluate them, and returns the population with the fitness
    values set

    :param population: set of Individual objects to be evaluated
    :return: The population with the fitness values of each individual.
    """
    while True:
        try:
            evaluations = input("[FITNESS] Evaluate generation. Check README.txt for evaluation format.\nEvaluation: ")

            individuals_evaluated = np.zeros(ZERO, dtype=np.int8)
            fitness_array = np.zeros(1 + LAMBDA, dtype=np.float32)

            if len(evaluations) > 0:
                for evaluation in evaluations.split(","):
                    evaluation_data = evaluation.split("=")

                    if len(evaluation_data) != 2:  # a = b
                        raise ValueError

                    index = int(evaluation_data[0])
                    if index < 0 or index >= 1 + LAMBDA:  # evaluation of non-existing individual
                        raise ValueError

                    individuals_evaluated = np.append(individuals_evaluated, index)
                    score = float(evaluation_data[1])
                    if score < 0 or score > 100:
                        raise ValueError
                    fitness_array[index] = score / 100

            for i in individuals_evaluated:
                population[i].fitness = fitness_array[i]

            return population

        except ValueError or IndexError:
            print("[ERROR] Invalid evaluation")


def main():
    description = "Cartesian Genetic Programming"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--configs', metavar='configs', type=str, help='Configs file')
    parser.add_argument('--save_folder', dest='save_folder', type=str, help='Folder in which the exported images '
                                                                            'are saved')

    args = parser.parse_args()
    configs = utils.load_configs(args.configs)
    if not utils.validate_configs(configs):
        sys.exit()

    if utils.validate_folder(args.save_folder):
        save_folder = args.save_folder
    else:
        print("[ERROR] Save folder does not exist!")
        sys.exit()

    random.seed(configs['seed'])

    input_img = utils.create_white_img(configs['image_width'], configs['image_height'])
    generate(configs, save_folder, fitness_function, input_img)


if __name__ == '__main__':
    main()
