# Run with python lsr.py train/adv_3.csv --plot

import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv


def load_points_from_file(filename):
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values

########################################################################################################################
# Least Squares Methods


def least_squares(x_1, y_1):
    y = np.matrix(y_1).reshape(20, 1)
    x = np.matrix(x_1).reshape(20, 1)

    # Creating X and EXP matrix
    ones = np.ones((20, 1))
    squared = np.power(x, 2)
    cubed = np.power(x, 3)
    x = np.concatenate((ones, x, squared, cubed), axis=1)

    matrix = x.transpose() * x
    als = inv(matrix) * x.transpose() * y

    a_1 = als.item(0)
    b_1 = als.item(1)
    c_1 = als.item(2)
    d_1 = als.item(3)

    error_temp = 0
    for i in range(0, 20):
        error_temp += ((d_1 * (x_1[i])**3) + (c_1 * (x_1[i]**2) + b_1 * x_1[i] + a_1) - y_1[i]) ** 2

    return a_1, b_1, c_1, d_1, error_temp

def least_squares_linear(x_1, y_1):
    y = np.matrix(y_1).reshape(20, 1)
    x = np.matrix(x_1).reshape(20, 1)

    # Creating X and EXP matrix
    ones = np.ones((20, 1))

    x = np.concatenate((ones, x), axis=1)

    matrix = x.transpose() * x
    als = inv(matrix) * x.transpose() * y

    a_1 = als.item(0)
    b_1 = als.item(1)

    error_temp = 0
    for i in range(0, 20):
        error_temp += ((b_1 * x_1[i] + a_1) - y_1[i]) ** 2

    return a_1, b_1, error_temp


def least_squares_sine(x_1, y_1):
    y = np.matrix(y_1).reshape(20, 1)
    x = np.matrix(x_1).reshape(20, 1)

    # Creating X and EXP matrix
    ones = np.ones((20, 1))
    sine = np.sin(x)
    x = np.concatenate((ones, sine), axis=1)

    matrix = x.transpose() * x
    als = inv(matrix) * x.transpose() * y

    a_1 = als.item(0)
    b_1 = als.item(1)

    error_temp = 0
    for i in range(0, 20):
        error_temp += ((np.sin(x_1[i]) * b_1 + a_1) - y_1[i]) ** 2

    return a_1, b_1, error_temp


########################################################################################################################
# MAIN FUNCTION


x, y = load_points_from_file(sys.argv[1])
assert len(x) == len(y)
assert len(x) % 20 == 0
num_segments = len(x) // 20

error = 0

if len(sys.argv) > 2:
    if sys.argv[2] == '--plot':

        colour = 'b'
        plt.set_cmap('Dark2')
        plt.scatter(x, y, c=colour)

        for i in range(num_segments):
            m = x[i * 20:(i * 20) + 20]
            n = y[i * 20:(i * 20) + 20]
            a, b, c, d, error_1 = least_squares(m, n)
            a_2, b_2, error_2 = least_squares_sine(m, n)
            a_3, b_3, error_3 = least_squares_linear(m, n)

            if error_1 > error_2:  # Sine function
                error += error_2
                plt.plot(m, (b_2 * np.sin(m)) + a_2, 'r')
            else:

                if ((error_3 - error_1) / error_3) < 0.2:   # If relative error is less than 0.1%, use linear, the difference is noise!
                    error += error_3
                    plt.plot(m, (b_3 * m) + a_3, 'r')
                else:
                    error += error_1
                    plt.plot(m, d * (m ** 3) + c * (m ** 2) + b * m + a, 'r')

        plt.show()
        print(error)

else:   #   separate function for no-plotting, reduces time for large inputs
    for i in range(num_segments):
        m = x[i * 20:(i * 20) + 20]
        n = y[i * 20:(i * 20) + 20]
        a, b, c, d, error_1 = least_squares(m, n)
        a_2, b_2, error_2 = least_squares_sine(m, n)
        a_3, b_3, error_3 = least_squares_linear(m, n)

        if error_1 > error_2:  # Sine function
            error += error_2
        else:
            if (error_3 - error_1) / (error_1 + error_3) < 0.2:  # If relative error is less than 0.1%, use linear, the difference is noise!
                error += error_3

            else:
                error += error_1

    print(error)
