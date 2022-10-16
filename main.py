import argparse
import evolution
import utils


def main():

    # add random seed and its value to the parameters
    description = "Cartesian Genetic Programming"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('configs', metavar='configs', type=str, help='Configs file')
    args = parser.parse_args()
    configs = utils.load_configs(args.configs)

    input_img = utils.create_white_img(configs['image_width'], configs['image_height'])
    evolution.generate(configs, input_img)


if __name__ == '__main__':
    main()
