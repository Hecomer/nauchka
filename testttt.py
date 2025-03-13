import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev


def lopatki_points(n_points=100):
    x_left, x_right = -2, 3  # Границы по x
    y_lower, y_shift = -3, 6  # Смещение верхней границы

    # Нижняя граница - обрезанная парабола
    x_lower = np.linspace(x_left + 0.5, x_right, n_points)  # Обрезка слева сильнее
    y_lower = -0.2 * x_lower ** 2 + y_lower

    # Верхняя граница - такая же парабола, сдвинутая вверх
    x_upper = x_lower.copy()
    y_upper = y_lower + y_shift

    # Левые и правые границы (вертикальные линии)
    y_left = np.linspace(y_lower[0], y_upper[0], n_points)
    y_right = np.linspace(y_lower[-1], y_upper[-1], n_points)
    x_left = np.linspace(x_lower[0], x_upper[0], n_points)
    x_right = np.linspace(x_lower[-1], x_upper[-1], n_points)
    for i in range(n_points):
        x_left[i] -= 3
        x_right[i] += 3

    # Отображение точек границ
    plt.plot(x_lower, y_lower, 'r', label='Нижняя граница')
    plt.plot(x_upper, y_upper, 'b', label='Верхняя граница')
    plt.plot(x_left, y_left, 'g', label='Левая граница')
    plt.plot(x_right, y_right, 'm', label='Правая граница')

    plt.axis('equal')
    plt.legend()
    plt.show()


lopatki_points(100)
