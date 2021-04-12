from multiprocessing import cpu_count, Pool
from datetime import datetime

import numpy as np
from numpy.random import choice
import cv2 as cv
import math
import random
import logging
import matplotlib.pyplot as plt
import statistics


def add_noise(img, i) -> np.ndarray:
    row, col = img.shape

    number_of_pixels = int((row * col) / i)  # not a random number anymore
    for i in range(number_of_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)

        img[y_coord][x_coord] = 255

    for i in range(number_of_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)

        img[y_coord][x_coord] = 0

    return img


def moravec(image, threshold=100):
    """Moravec's corner detection for each pixel of the image."""

    corners = []
    xy_shifts = [(1, 0), (1, 1), (0, 1), (-1, 1)]
    width, height = image.shape[:2]
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Look for local maxima in min(E) above threshold:
            E = 100000
            for shift in xy_shifts:
                diff = int(image[x + shift[0], y + shift[1]])
                diff = int(diff - image[x, y])
                diff = int(diff * diff)

                if (diff < E):
                    E = diff
            if E > threshold:
                corners.append((x, y))

    return corners


def harris(image, threshold=100000000, sigma=1.5, k=0.04) -> list:
    """Harris' corner detection for each pixel of the image."""

    harris_corners = []
    width, height = image.shape[:2]

    # Calculate gradients:
    X2 = [[0] * width for y in range(height)]
    Y2 = [[0] * width for y in range(height)]
    XY = [[0] * width for y in range(height)]

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            X = int(image[x + 1, y]) - int(image[x - 1, y])
            Y = int(image[x, y + 1]) - int(image[x, y - 1])

            X2[y][x] = int(X * X)
            Y2[y][x] = int(Y * Y)
            XY[y][x] = int(X * Y)

    # Gaussian 3x3:
    G = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for y in range(3):
        for x in range(3):
            u, v = x - 1, y - 1
            G[y][x] = (math.exp(-(u * u + v * v) / (2 * sigma * sigma)))

    # Convolve with Gaussian 3x3:
    A = [[0] * width for y in range(height)]
    B = [[0] * width for y in range(height)]
    C = [[0] * width for y in range(height)]

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            for i in range(3):
                for j in range(3):
                    u, v = j - 1, i - 1
                    A[y][x] = A[y][x] + X2[y + v][x + u] * G[i][j]
                    B[y][x] = B[y][x] + Y2[y + v][x + u] * G[i][j]
                    C[y][x] = C[y][x] + XY[y + v][x + u] * G[i][j]
    del X2, Y2, XY

    # Harris Response Function:
    R = [[0] * width for y in range(height)]
    for y in range(height):
        for x in range(width):
            a, b, c = A[y][x], B[y][x], C[y][x]
            Tr = a + b
            Det = a * b - c * c
            R[y][x] = Det - k * Tr * Tr
    del A, B, C

    # Suppress Non-Maximum Points:
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            maximum = True
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if R[y][x] < R[y + dy][x + dx]:
                        maximum = False
            if maximum and R[y][x] > threshold:
                harris_corners.append((x, y))

    return harris_corners


def loop(params: list) -> dict:
    results = {'params': params}
    img = cv.imread(params[0], cv.IMREAD_GRAYSCALE)
    img = add_noise(img, params[1])

    moravec_corners = moravec(img, 1000)
    harris_corners = harris(img, 100000000)

    results['moravec'] = len(moravec_corners)
    results['harris'] = len(harris_corners)
    results['moravec_data'] = moravec_corners
    results['harris_data'] = harris_corners

    return results


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level='INFO')
    logging.getLogger(__name__)

    start_ = datetime.now()
    logging.info(f'Start: {start_}')
    start, stop, step = 10, 80, 10
    # images = ['parking1.png', 'parking2.png', 'PST_1920x1088_cam4.png', 'PC_c1_1 (1).png']
    images = ['parking1.png', 'parking2.png']
    # Get your todos:
    todo = [[image, percentage] for image in images for percentage in range(start, stop, step)]

    # Get the pool size
    pool_size = cpu_count() if cpu_count() <= len(todo) else len(todo)

    # Prepare worker pool
    logging.info(f'Creating pool of size {pool_size}')
    worker_pool = Pool(pool_size)

    # Let it happen
    logging.info(f'Starting to process your data...')
    results = worker_pool.map(loop, todo)
    logging.info(f'Processing completed.')
    worker_pool.close()

    # Show the results
    # print('\n------------------------\n')
    # for result in results:
    #     for item in result.items():
    #         print(f'{item[0]}: {item[1]}')
    #     print('\n------------------------\n')

    stop_ = datetime.now()

    moravecs = [res['moravec'] for res in results]
    harriss = [res['harris'] for res in results]
    levels = [i for i, j in enumerate(moravecs)]

    print(statistics.mean(moravecs))
    print(statistics.mean(harriss))

    plt.plot(levels, moravecs, 'ro')
    plt.plot(levels, harriss, 'o')
    plt.show()

    logging.info(f'Stop: {stop_}')
    logging.info(f'Time: {stop_ - start_}')
