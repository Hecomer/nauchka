import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from time import perf_counter_ns

np.random.seed(179)

n = 2500
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

b = np.random.normal(size=n)
b = A_sparse @ b    # Матричное умножение


def givens_rotation(a, b):
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


def gmres(A, b, x0, max_iter=50, tol=1e-6):
    n = len(b)
    Q = np.zeros((n, max_iter+1))
    H = np.zeros((max_iter+1, max_iter))
    x = x0.copy()
    r = b - A(x)
    beta = np.linalg.norm(r)
    Q[:, 0] = r / beta
    for k in range(max_iter):
        y = np.zeros(k+2)
        for i in range(k+1):
            c, s = givens_rotation(H[i, i], H[i+1, i])
            temp = c*H[i, i] + s*H[i+1, i]
            H[i+1, i] = -s*H[i, i] + c*H[i+1, i]
            H[i, i] = temp
            y[i] = c*y[i] + s*y[i+1]
            y[i+1] = -s*y[i] + c*y[i+1]
        c, s = givens_rotation(H[k+1, k], H[k+1, k])
        H[k+1, k] = c*H[k+1, k] + s*H[k+1, k]
        H[k+1, k] = 0
        print(H)
        y[k+1] = -s*y[k]
        y = y[:-1]
        x = x + Q[:, :k+1].dot(y)
        r = b - A(x)
        beta = np.linalg.norm(r)
        if beta < tol:
            break
        Q[:, k+1] = r / beta
    return x


'''
time_before = perf_counter_ns()
x0 = np.zeros(n)
linear_map = lambda x: A_sparse @ x
x = gmres(linear_map, b, x0, 50)
time_taken = (perf_counter_ns() - time_before) * 1e-6
error = np.linalg.norm(A_sparse @ x - b) ** 2
print(f"Используя GMRES: ошибка: {error:.4e}, за время {time_taken:.1f}мс, решение: {x}")
'''
