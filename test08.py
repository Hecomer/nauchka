import numpy as np


def compute_polygon_area(x1, y1, x2, y2, x3, y3, x4, y4):
    points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

