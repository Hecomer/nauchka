import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# Функция для создания регулярной сетки
def generate_grid(nx, ny, x_range=(0, 1), y_range=(0, 1)):
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    return np.meshgrid(x, y)


# Функция, описывающая целевую форму ячеек (например, квадратная ячейка)
def target_shape(x, y):
    # Для примера целевая форма - квадрат
    return np.array([x + 1, y + 1])


# Функция для вычисления отклонения формы ячейки от целевой
def cell_deviation(cell, target_cell):
    # Разница по длинам сторон
    side_diff = np.linalg.norm(cell[1] - cell[0]) - np.linalg.norm(target_cell[1] - target_cell[0])
    return side_diff ** 2


# Функция для минимизации отклонений сетки
def minimize_grid(grid_x, grid_y):
    nx, ny = grid_x.shape
    # Целевая сетка для коррекции (для примера: сдвиг всех ячеек на (1, 1))
    target_grid_x, target_grid_y = target_shape(grid_x, grid_y)

    def objective(params):
        # Формируем сетку на основе параметров
        new_grid_x = params[:nx * ny].reshape((nx, ny))
        new_grid_y = params[nx * ny:].reshape((nx, ny))

        # Вычисляем сумму отклонений от целевой формы
        deviation = 0
        for i in range(nx - 1):
            for j in range(ny - 1):
                cell = [(new_grid_x[i, j], new_grid_y[i, j]),
                        (new_grid_x[i + 1, j], new_grid_y[i + 1, j])]
                target_cell = [(target_grid_x[i, j], target_grid_y[i, j]),
                               (target_grid_x[i + 1, j], target_grid_y[i + 1, j])]
                deviation += cell_deviation(cell, target_cell)

        return deviation

    # Начальное приближение - текущая сетка
    initial_guess = np.hstack([grid_x.ravel(), grid_y.ravel()])

    # Минимизируем отклонение
    result = minimize(objective, initial_guess)

    # Восстанавливаем сетку из результата оптимизации
    optimized_grid_x = result.x[:nx * ny].reshape((nx, ny))
    optimized_grid_y = result.x[nx * ny:].reshape((nx, ny))

    return optimized_grid_x, optimized_grid_y


# Основная функция
def main():
    # Параметры сетки
    nx, ny = 10, 10
    grid_x, grid_y = generate_grid(nx, ny)

    # Отображение начальной сетки
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(grid_x, grid_y, color='b')
    plt.plot(grid_x.T, grid_y.T, color='b')
    plt.title('Начальная сетка')

    # Минимизация и корректировка сетки
    optimized_grid_x, optimized_grid_y = minimize_grid(grid_x, grid_y)

    # Отображение оптимизированной сетки
    plt.subplot(1, 2, 2)
    plt.plot(optimized_grid_x, optimized_grid_y, color='r')
    plt.plot(optimized_grid_x.T, optimized_grid_y.T, color='r')
    plt.title('Оптимизированная сетка')

    plt.show()


if __name__ == "__main__":
    main()