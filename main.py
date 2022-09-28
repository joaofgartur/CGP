import evolution
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio


def load_input(img_path):
    img = plt.imread(img_path)
    num_rows, num_columns = np.shape(img[:, :, 0])

    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    img = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return img, num_rows, num_columns


def show_img(img):
    plt.imshow(img)
    plt.show()


def save_img(generation, individual, img):
    filename = "outputs/" + str(generation) + "_" + str(individual) + ".png"
    img = img.astype(np.uint8)
    imageio.imwrite(filename, img)


def main():
    img_path = "input/img.png"
    input_img, img_rows, img_columns = load_input(img_path)

    show_img(input_img)

    configs = {
        'num_rows': 1,
        'num_columns': 1,
        'level_back': 1,
        'num_input': 2,
        'num_output': 1,
        'num_functions': 1,
        'lambda_arg': 4,
        'arity': 2,
    }

    evolution.generate(configs, input_img)

if __name__ == '__main__':
    main()
