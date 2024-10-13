import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# Функция для генерации начальной сетки
def generate_initial_grid(nx, ny):
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xv, yv = np.meshgrid(x, y)
    return xv, yv


# Размеры сетки
nx, ny = 10, 10
xv, yv = generate_initial_grid(nx, ny)

plt.figure(figsize=(6, 6))
plt.plot(xv, yv, 'k-')
plt.plot(xv.T, yv.T, 'k-')
plt.title('Initial Grid')
plt.show()


def quality_function(x, y, nx, ny):
    quality = 0
    for i in range(nx - 1):
        for j in range(ny - 1):
            dx1 = x[i+1, j] - x[i, j]
            dy1 = y[i+1, j] - y[i, j]
            dx2 = x[i, j+1] - x[i, j]
            dy2 = y[i, j+1] - y[i, j]
            area = abs(dx1 * dy2 - dx2 * dy1)
            quality += (area - 1/nx/ny)**2
    return quality


def objective(params, nx, ny):
    x = params[:nx*ny].reshape((nx, ny))
    y = params[nx*ny:].reshape((nx, ny))
    return quality_function(x, y, nx, ny)

# Начальные координаты узлов сетки
initial_params = np.hstack((xv.flatten(), yv.flatten()))

# Запуск оптимизации
result = minimize(objective, initial_params, args=(nx, ny), method='L-BFGS-B')

# Получение оптимизированных координат
optimized_params = result.x
x_opt = optimized_params[:nx*ny].reshape((nx, ny))
y_opt = optimized_params[nx*ny:].reshape((nx, ny))

plt.figure(figsize=(6, 6))
plt.plot(x_opt, y_opt, 'k-')
plt.plot(x_opt.T, y_opt.T, 'k-')
plt.title('Optimized Grid')
plt.show()


def plot_grid(x, y, title):
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, 'k-')
    plt.plot(x.T, y.T, 'k-')
    plt.title(title)
    plt.show()


# Оценка качества начальной и оптимизированной сеток
initial_quality = quality_function(xv, yv, nx, ny)
optimized_quality = quality_function(x_opt, y_opt, nx, ny)

print(f'Initial Quality: {initial_quality}')
print(f'Optimized Quality: {optimized_quality}')


