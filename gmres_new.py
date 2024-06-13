import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from time import perf_counter_ns
import random

np.random.seed(179)

n = 2
N = n
shape = (n, n)


# Создаем случайную разреженную (n, n) матрицу с 2*N (2, потому что на диагонали N единиц) ненулевыми элементами
coords = np.random.choice(n * n, size=N, replace=False)
coords = np.unravel_index(coords, shape)
values = np.random.normal(size=N)
A_sparse = scipy.sparse.coo_matrix((values, coords), shape=shape)
A_sparse = A_sparse.tocsr()         # в разреж. матр. хранятся только массивы строк и столбцов ненулевых значений,
A_sparse += scipy.sparse.eye(n)     # а также сами ненулевые значения
A_dense = A_sparse.toarray()    # плотная матрица (нужна для сравнения в конце)
print(A_dense)


b = np.random.normal(size=n)
print(b)
b = A_sparse.dot(b)    # Матричное умножение
print(b)


def givens_rotation(a, b):  # реализация вращений Гивенса
    if b == 0:
        c = 1
        s = 0
    else:
        if abs(b) > abs(a):
            r = a / b
            s = 1 / np.sqrt(1 + r**2)
            c = s * r
        else:
            r = b / a
            c = 1 / np.sqrt(1 + r**2)
            s = c * r
    return c, s


def least_squares(A, b):    # Метод наименьших квадратов, используя вращения Гивенса
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()

    for j in range(min(m, n)):
        for i in range(j + 1, m):
            c, s = givens_rotation(R[j, j], R[i, j])
            G = np.array([[c, s], [-s, c]])
            R[[j, i], j:] = G @ R[[j, i], j:]
            Q[:, [j, i]] = Q[:, [j, i]] @ G.T

    Q1 = Q[:, :n]
    R1 = R[:n, :]
    x = np.linalg.solve(R1, Q1.T @ b)

    return x


def gmres(linear_map, b, x0, n_iter):
    # Инициализация
    n = x0.shape[0]
    H = np.zeros((n_iter + 1, n_iter))
    r0 = b - linear_map(x0)
    beta = np.linalg.norm(r0)
    V = np.zeros((n_iter + 1, n))
    V[0] = r0 / beta

    for j in range(n_iter):
        # Вычисляем следующий вектор из подпространства Крылова
        w = linear_map(V[j])

        # Ортогонализация Грама-Шмидта
        for i in range(j + 1):
            H[i, j] = np.dot(w, V[i])
            w -= H[i, j] * V[i]
        H[j + 1, j] = np.linalg.norm(w)

        # Добавить новый вектор к базису
        V[j + 1] = w / H[j + 1, j]

    # Найти наилучшее приближение в базисе V
    e1 = np.zeros(n_iter + 1)
    e1[0] = beta
    y = least_squares(H, e1)   # Вычисляем вектор, который приближенно решает A@x=b

    # Преобразовываем результат обратно в полный базис и возвращаем
    x_new = x0 + V[:-1].T @ y   # Конечная оценка
    return x_new


time_before = perf_counter_ns()
x0 = np.zeros(n)
linear_map = lambda x: A_sparse.dot(x)
x = gmres(linear_map, b, x0, 2)
time_taken = (perf_counter_ns() - time_before) * 1e-6
error = np.linalg.norm(A_sparse.dot(x) - b) ** 2
print(f"Используя GMRES: ошибка: {error:.4e}, за время {time_taken:.1f}мс, решение: {x}")

# Для сравнения решим слау методом наименьших квадратов
time_before = perf_counter_ns()
x = np.linalg.lstsq(A_dense, b, rcond=None)[0]
time_taken = (perf_counter_ns() - time_before) * 1e-6
error = np.linalg.norm(A_dense.dot(x) - b)**2
print(f"Используя lstsq: ошибка: {error:.4e}, за время {time_taken:.1f}мс, решение {x}")
