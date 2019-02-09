import warnings
from math import log
from statistics import pstdev, mean
from typing import List

import numpy as np
from scipy.linalg import lstsq


def gen_y(k, v, current_y):
    if k == 0:
        result = v
    else:
        result = v + current_y[k - 1]

    # result = float(format(result, '.4f'))
    current_y[k] = result

    return result


def hurst(series: List[float]):
    x_data = []
    y_data = []
    for i in range(len(series) + 1):
        if i <= 1:
            continue

        data = series[0:i]
        avg = mean(data)
        y_temp = [i - avg for i in data]
        y = [gen_y(k, v, y_temp) for k, v in enumerate(y_temp)]
        # print(y)
        r = max(y) - min(y)
        # print(r)
        s = pstdev(data)
        # print(s)
        # print(r / s)
        log_n = log(len(data), 2)
        log_rs = log(r / s, 2)
        # print(log_n)
        # print(log_rs)
        # if log_n in [1, 2, 3, 4]:
        x_data.append(log_n)
        y_data.append(log_rs)

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    matrix = x_data[:, np.newaxis] ** [1, 0]
    # print(M)
    p, res, rnk, s = lstsq(matrix, y_data, lapack_driver='gelss')
    return p[0]


file = open('tests/data/sample_data.csv')
line = file.readline()
n_series: List[float] = list(map(lambda x: float(x), line.split(',')))
file.close()

print(hurst(n_series))
#
# print(hurst([12, 10, 8, 6, 4, 2, 4, 6, 8, 6, 4, 2, 1, 2, 4, 6, 8, 6, 8, 10, 8, 6, 2, 4, 6, 8, 10, 10, 12, 10, 8, 4]))
#
# print(hurst([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]))
#
# print(hurst([1.1, 1.5, 1.9, 2.2, 2.5, 2.7, 2.9, 3.3, 3.5, 3.7, 3.9, 4.0, 4.2, 4.3, 4.5, 4.7]))

print(hurst(np.random.rand(1000)))
