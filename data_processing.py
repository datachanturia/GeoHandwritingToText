
import os
import cv2
import numpy as np
import pandas as pd

char_dict = \
    {"g": "გ",
     "v": "ვ",
     "z": "ზ",
     "k": "კ",
     "n": "ნ",
     "t": "ტ",
     "f": "ფ",
     "R": "ღ",
     "y": "ყ",
     "S": "ს",
     }


def get_all_symbols(path):
    # reads images and sets training set
    symbols_array = []
    expected_symbols = "gvzkntfRyS"
    for symbol_name in expected_symbols:
        for file in os.listdir(path + symbol_name + "/"):
            filename = os.fsdecode(file)
            if filename.endswith(".jpg"):
                im_gray = cv2.imread(path + symbol_name + "/" + filename, cv2.IMREAD_GRAYSCALE)

                thresh = 100
                im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
                symbols_array.append([symbol_name, im_bw])
    return symbols_array


def get_training_set(symbols_array, chars_map):
    training_matrix = []

    for element in symbols_array:
        new_row = []
        new_row.append(chars_map[element[0]])
        for line in element[1]:
            for pixel in line:
                new_row.append(pixel)
        training_matrix.append(new_row.copy())
    data_frame = pd.DataFrame(np.array(training_matrix))
    columns = ["label"]
    for i in range(625):
        columns.append("pixel" + str(i + 1))
    data_frame.columns = columns
    return data_frame


path = "./images/chars/"
all_symbols = get_all_symbols(path)

training_set = get_training_set(all_symbols, char_dict)
training_set.to_csv('train.csv', index=False)
