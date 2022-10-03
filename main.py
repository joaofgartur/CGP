import evolution
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import datetime


def load_input(img_path):
    img = plt.imread(img_path)

    return img


def show_img(img):
    plt.imshow(img)
    plt.show()


def save_img(path, generation, individual, img):
    filename = path + "/" +str(generation) + "_" + str(individual) + ".png"
    img = img.astype(np.uint8)
    imageio.imwrite(filename, img)


def get_current_timestamp():
    ct = datetime.datetime.now()

    timestamp = str(ct)
    timestamp = timestamp[:timestamp.find('.')]
    timestamp = timestamp.replace('-', '_').replace(':', '_').replace(' ', '_')

    return timestamp


def create_output_folder(path):
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)

    if not isExist:
        os.makedirs(path)


def main():
    img_path = "input/img.png"
    input_img = load_input(img_path)

    # show_img(input_img)

    configs = {
        'num_rows': 10,
        'num_columns': 10,
        'level_back': 8,
        'num_input': 2,
        'num_output': 3,
        'num_functions': 6,
        'lambda_arg': 4,
        'arity': 2,
        'mutation_rate': 0.5,
        'max_generation': 3,
    }

    evolution.generate(configs, input_img)


if __name__ == '__main__':
    main()
