
import numpy as np
import cv2
import math
from PIL import Image
import os


class Coordinate:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def equals(self, x, y):
        return x == self.x and y == self.y


class Symbol:
    def __init__(self):
        self.coordinates = []
        self.max_x = 0
        self.max_y = 0

        self.min_x = 100000
        self.min_y = 100000

    def add_coordinate(self, x, y):
        self.coordinates.append(Coordinate(x, y))
        if x > self.max_x:
            self.max_x = x
        if y > self.max_y:
            self.max_y = y
        if x < self.min_x:
            self.min_x = x
        if y < self.min_y:
            self.min_y = y

    def contains_coordinate(self, x, y):
        for element in self.coordinates:
            if element.equals(x, y):
                return True
        return False


def find_neighbours(image, ref_x, ref_y, symbol):
    before_coordinates = len(symbol.coordinates)

    min_y = max(ref_y - 1, 0)
    max_y = min(ref_y + 2, len(image))

    min_x = max(ref_x - 1, 0)
    max_x = min(ref_x + 2, len(image[0]))
    for y_coord in range(min_y, max_y):
        for x_coord in range(min_x, max_x):
            if image[y_coord][x_coord] < 255 and not symbol.contains_coordinate(x_coord, y_coord):
                symbol.add_coordinate(x_coord, y_coord)

    for coord_index in range(before_coordinates, len(symbol.coordinates)):
        find_neighbours(image, symbol.coordinates[coord_index].x, symbol.coordinates[coord_index].y, symbol)


def refactor_images():
    file_name = "x6"
    max_height = 0
    max_width = 0
    # for file in os.listdir("./images"):
    #     filename = os.fsdecode(file)
    #     if filename.endswith(".jpg"):
    im_gray = cv2.imread("./images/" + file_name + ".jpg", cv2.IMREAD_GRAYSCALE)

    thresh = 100
    im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]

    symbol_array = []
    prev_symbol = Symbol()

    n_lines = 17
    line_len = math.floor(len(im_bw) / n_lines)
    for line_y in range(0, n_lines):
        bw_y = line_y * line_len + math.floor(line_len / 2)
        max_x = 0
        bw_x = 0
        while bw_x < len(im_bw[0]):
            if im_bw[bw_y][bw_x] == 0 and not prev_symbol.contains_coordinate(bw_x, bw_y):
                prev_symbol = Symbol()
                find_neighbours(im_bw, bw_x, bw_y, prev_symbol)
                symbol_array.append(prev_symbol)
                if prev_symbol.max_x > max_x:
                    max_x = prev_symbol.max_x
                    bw_x = max_x
                    continue
            bw_x += 1

    i = 0
    for symbol in symbol_array:
        sm_array = np.zeros([symbol.max_y + 1 - symbol.min_y, symbol.max_x + 1 - symbol.min_x])
        if len(sm_array[0]) > max_width:
            max_width = len(sm_array[0])
        if len(sm_array) > max_height:
            max_height = len(sm_array)

        sm_array += 255
        for elem in symbol.coordinates:
            sm_array[elem.y - symbol.min_y, elem.x - symbol.min_x] = 0

        cv2.imwrite('./images/rz/' + file_name + str(i) + '.jpg', sm_array)
        i += 1
    #     else:
    #         continue
    # return max_height, max_width


# max_height, max_width = refactor_images()
refactor_images()
