import argparse
import utils
import numpy as np
import random
from evolution import ZERO, LAMBDA, generate


def fitness_function(population):
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
    parser.add_argument('configs', metavar='configs', type=str, help='Configs file')
    args = parser.parse_args()
    configs = utils.load_configs(args.configs)

    random.seed(configs['seed'])
    input_img = utils.create_white_img(configs['image_width'], configs['image_height'])
    # input_img = utils.create_random_img(configs['image_width'], configs['image_height'])
    generate(configs, fitness_function, input_img)


if __name__ == '__main__':
    main()
