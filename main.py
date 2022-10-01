import evolution
import numpy as np
import matplotlib.pyplot as plt
import imageio


def load_input(img_path):
    img = plt.imread(img_path)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    img = (0.2989 * r + 0.5870 * g + 0.1140 * b) * 255

    return img


def show_img(img):
    plt.imshow(img)
    plt.show()


def save_img(generation, individual, img):
    filename = "outputs/" + str(generation) + "_" + str(individual) + ".png"
    img = img.astype(np.uint8)
    imageio.imwrite(filename, img)


def main():
    img_path = "input/img.png"
    input_img = load_input(img_path)

    # show_img(input_img)

    configs = {
        'num_rows': 10,
        'num_columns': 10,
        'level_back': 8,
        'num_input': 2,
        'num_output': 1,
        'num_functions': 6,
        'lambda_arg': 4,
        'arity': 2,
        'mutation_rate': 0.5,
        'max_generation': 10,
    }

    evolution.generate(configs, input_img)


if __name__ == '__main__':
    main()
