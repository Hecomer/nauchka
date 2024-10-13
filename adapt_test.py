import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# Функция для вычисления якобиана
def compute_jacobian(grid, i, j, nx, ny):
    if i == 0:
        dx_dxi = grid[i + 1, j, 0] - grid[i, j, 0]
        dy_dxi = grid[i + 1, j, 1] - grid[i, j, 1]
    elif i == nx - 1:
        dx_dxi = grid[i, j, 0] - grid[i - 1, j, 0]
        dy_dxi = grid[i, j, 1] - grid[i - 1, j, 1]
    else:
        dx_dxi = (grid[i + 1, j, 0] - grid[i - 1, j, 0]) / 2
        dy_dxi = (grid[i + 1, j, 1] - grid[i - 1, j, 1]) / 2

    if j == 0:
        dx_deta = grid[i, j + 1, 0] - grid[i, j, 0]
        dy_deta = grid[i, j + 1, 1] - grid[i, j, 1]
    elif j == ny - 1:
        dx_deta = grid[i, j, 0] - grid[i, j - 1, 0]
        dy_deta = grid[i, j, 1] - grid[i, j - 1, 1]
    else:
        dx_deta = (grid[i, j + 1, 0] - grid[i, j - 1, 0]) / 2
        dy_deta = (grid[i, j + 1, 1] - grid[i, j - 1, 1]) / 2

    J = np.array([[dx_dxi, dx_deta], [dy_dxi, dy_deta]])
    return J


# Функция для вычисления метрического тензора
def compute_metric_tensor(J):
    return np.dot(J.T, J)


# Функция для вычисления контрвариантного метрического тензора
def compute_contravariant_metric_tensor(g):
    return np.linalg.inv(g)


# Пример управляющей метрики G как функции от параметрических координат
def control_metric(xi, eta):
    # Здесь задается функция управляющей метрики G
    # Пример: G = [[1 + xi, 0], [0, 1 + eta]]
    G = np.array([[1 + xi, 0], [0, 1 + eta]])
    return G


# Функция для вычисления значения функционала
def functional(x, grid_shape):
    nx, ny = grid_shape
    grid = x.reshape((nx, ny, 2))

    total_integral = 0.0
    for i in range(nx):
        for j in range(ny):
            J_g = compute_jacobian(grid, i, j, nx, ny)
            g = compute_metric_tensor(J_g)

            # Вычисление параметрических координат (xi, eta)
            xi = i / (nx - 1)
            eta = j / (ny - 1)

            # Вычисление управляющей метрики G в точке (xi, eta)
            G = control_metric(xi, eta)
            G_inv = compute_contravariant_metric_tensor(G)

            tr_Ginv_g = np.trace(np.dot(G_inv, g))
            det_G = np.linalg.det(G)
            det_g = np.linalg.det(g)

            integrand = (tr_Ginv_g ** (1 / 2)) * (det_G ** (1 / 4)) / (det_g ** (1 / 2))
            total_integral += integrand

    return total_integral / (nx * ny)


# Создание начальной сетки 10x10
nx, ny = 10, 10
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
xx, yy = np.meshgrid(x, y)
initial_grid = np.vstack([xx.ravel(), yy.ravel()]).T

# Оптимизация положения узлов сетки
result = minimize(lambda x: functional(x, (nx, ny)), initial_grid.flatten(), method='L-BFGS-B')
optimized_grid = result.x.reshape(-1, 2)


# Функция для отрисовки сетки
def plot_grid(grid, nx, ny, title):
    fig, ax = plt.subplots()
    for i in range(nx):
        for j in range(ny):
            if i < nx - 1:
                ax.plot([grid[i * ny + j, 0], grid[(i + 1) * ny + j, 0]],
                        [grid[i * ny + j, 1], grid[(i + 1) * ny + j, 1]], 'k-', lw=0.5)
            if j < ny - 1:
                ax.plot([grid[i * ny + j, 0], grid[i * ny + j + 1, 0]], [grid[i * ny + j, 1], grid[i * ny + j + 1, 1]],
                        'k-', lw=0.5)
    ax.scatter(grid[:, 0], grid[:, 1], color='red', s=10)
    ax.set_aspect('equal', 'box')
    plt.title(title)
    plt.show()


# Визуализация начальной и оптимизированной сеток
plot_grid(initial_grid, nx, ny, 'Initial Grid')
plot_grid(optimized_grid, nx, ny, 'Optimized Grid')