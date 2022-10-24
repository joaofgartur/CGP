import csv
import datetime
import json
import os
import imageio
import numpy as np
from matplotlib import pyplot as plt


def load_img(img_path):
    img = plt.imread(img_path)
    return img


def show_img(img):
    img = img.astype(np.uint8)
    plt.imshow(img)
    plt.show()


def save_img(path, individual, image):
    filename = path + "/" + str(individual) + ".png"
    image = image.astype(np.uint8)
    imageio.imwrite(filename, image)


def export_images(population, destination):
    index = 0
    for individual in population:
        save_img(destination, index, individual.data)
        index += 1


def create_white_img(width, height):
    img = np.zeros([width, height, 3], dtype=np.uint8)
    return img


def create_random_img(width, height):
    img = np.random.rand(width, height, 3) * 255
    img = img.astype(np.uint8)
    return img


def get_current_timestamp():
    ct = datetime.datetime.now()

    timestamp = str(ct)
    timestamp = timestamp[:timestamp.find('.')]
    timestamp = timestamp.replace('-', '_').replace(':', '_').replace(' ', '_')

    return timestamp


def create_directory(path):
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)


def write_to_file(filename, data):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)


def list_to_str(list_array):
    return "[" + ';'.join(map(str, list_array)) + "]"


def load_configs(filename):
    result = {}
    with open(filename, 'r') as f:
        result.update(json.load(f))
    return result

