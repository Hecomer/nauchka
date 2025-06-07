import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize
import sys
import threading
import time
from sklearn.decomposition import PCA
sys.setrecursionlimit(10000)


def find_intersection(f1, f2):
    intersection = fsolve(lambda x: f1(x) - f2(x), 0)
    xy = [intersection[0], f1(intersection[0])]
    return xy


def Xnewt(s):
    x = s
    y = s**2 + 3
    return np.array([x, y])


def Xnewb(s):
    x = s
    y = s**2 - 3
    return np.array([x, y])


def Xnewl(s):
    x = 0
    y = s
    return np.array([x, y])


def Xnewr(s):
    x = 1
    y = s
    return np.array([x, y])


def Xyb(s):
    x = s
    y = 0
    return np.array([x, y])


def Xyl(s):
    x = 0
    y = s
    return np.array([x, y])


def Xyr(s):
    x = 1 + 2 * s - 2 * s**1.5
    y = s
    return np.array([x, y])


def Xyt(s):
    x = s
    y = 1 - 3 * s + 3 * s**2
    return np.array([x, y])


def sq_t2(s):
    y = 4
    return y


def sq_b2(s):
    y = 0
    return y


def sq_l2(s):
    y = 1000000*s
    return y


def sq_r2(s):
    y = 1000000 - 1000000*s
    return y


def sq_t1(s):
    y = 1
    return y


def sq_b1(s):
    y = 0
    return y


def sq_l1(s):
    y = 1000000*s
    return y


def sq_r1(s):
    y = 1000000 - 1000000*s
    return y


def sq_t(s):
    y = 2
    return y


def sq_b(s):
    y = 0
    return y


def sq_l(s):
    y = 1000000*s
    return y


def sq_r(s):
    y = 2000000 - 1000000*s
    return y


def x_top_chevron(s):
    x = s
    if s <= 1 / 2:
        y = 1 - s
    else:
        y = s
    xy = np.array([x, y])
    return xy


def x_bottom_chevron(s):
    x = s
    if s <= 1 / 2:
        y = -s
    else:
        y = s - 1
    xy = np.array([x, y])
    return xy


def x_right_chevron(s):
    x = 1
    y = s
    xy = np.array([x, y])
    return xy


def x_left_chevron(s):
    x = 0
    y = s
    xy = np.array([x, y])
    return xy


def chevl(x):
    return 10000000 * x


def chevr(x):
    return 100000 - 100000 * x


def chevt(x):
    if x <= 1 / 2:
        y = 1 - x
    else:
        y = x
    return y


def chevb(x):
    if x <= 1 / 2:
        y = -x
    else:
        y = x - 1
    return y

def myl(x):
    return 10000000 * x + 10000000


def myr(x):
    return 100000 - 100000 * x

def myt(x):
    return (2*x)**2 + 4


def myb(x):
    return (2*x)**2


def Myl(s):
    x = -1
    y = s
    xy = np.array([x, y])
    return xy


def Myr(s):
    x = 1
    y = s
    xy = np.array([x, y])
    return xy


def Myt(s):
    x = s
    y = x**2 + 4
    xy = np.array([x, y])
    return xy


def Myb(s):
    x = s
    y = x**2
    xy = np.array([x, y])
    return xy


def horsl(s):
    x = -s
    y = 0
    xy = np.array([x, y])
    return xy


def horsr(s):
    x = s
    y = 0
    xy = np.array([x, y])
    return xy


def horst(s):
    x = math.sin(math.pi/2*(1-2*s))
    y = math.cos(math.pi/2*(1-2*s))
    xy = np.array([x, y])
    return xy


def horsb(s):
    x = math.sin(math.pi/2*(1-2*s))
    y = math.cos(math.pi/2*(1-2*s))
    xy = np.array([x, y])
    return xy


def implicit_transfinite_interpol(discr, Xt, Xb, Xr, Xl, nod=None):
    # nod = [x, y, xn, yn] - [положение в сетке по индексам, коорд в прве], по умолчанию не используется

    m = discr
    n = discr

    intersection_tr = find_intersection(Xt, Xr)
    intersection_tl = find_intersection(Xt, Xl)
    intersection_br = find_intersection(Xb, Xr)
    intersection_bl = find_intersection(Xb, Xl)
    #   print(intersection_bl)
    #   print(intersection_tl)

    x_discr_t = np.linspace(intersection_tl[0], intersection_tr[0], discr)    # дискретизация границ по x
    x_discr_b = np.linspace(intersection_bl[0], intersection_br[0], discr)
    x_discr_l = np.linspace(intersection_bl[0], intersection_tl[0], discr)
    x_discr_r = np.linspace(intersection_br[0], intersection_tr[0], discr)

    t_x_val = []
    b_x_val = []
    r_x_val = []
    l_x_val = []
    t_y_val = []
    b_y_val = []
    r_y_val = []
    l_y_val = []

    for i in range(discr):
        t_x_val.append(x_discr_t[i])
        b_x_val.append(x_discr_b[i])
        l_x_val.append(x_discr_l[i])
        r_x_val.append(x_discr_r[i])

        t_y_val.append(Xt(x_discr_t[i]))
        b_y_val.append(Xb(x_discr_b[i]))
        l_y_val.append(Xl(x_discr_l[i]))
        r_y_val.append(Xr(x_discr_r[i]))
    #   print(r_y_val)
    """print(t_x_val,
    b_x_val,
    r_x_val,
    l_x_val,
    t_y_val,
    b_y_val,
    r_y_val,
    l_y_val, sep="\n"+"\n")"""

    X = np.zeros((m*n, m*n))  # матрица (позиция иксов исходя из дискретизации; иксы на этой позиции)
    Y = np.zeros((m*m, m*n))
    bx = np.zeros(m*n)
    by = np.zeros(m*n)
    for i in range(discr):
        for j in range(discr):

            gidx = i*discr + j

            if j == 0:
                X[gidx, i*discr + j] = 1.0
                bx[gidx] = b_x_val[i]
            elif j == discr - 1:
                X[gidx, i*discr + j] = 1.0
                bx[gidx] = t_x_val[i]
            elif i == 0:
                X[gidx, i*discr + j] = 1.0
                bx[gidx] = l_x_val[j]
            elif i == discr - 1:
                X[gidx, i*discr + j] = 1.0
                bx[gidx] = r_x_val[j]
            elif nod != None and i == nod[0]-1 and j == nod[1]-1:
                X[gidx, i * discr + j] = 1.0
                bx[gidx] = nod[2]
            else:
                X[gidx, i * discr + j] = 1.0
                X[gidx, (i-1) * discr + j] = -1/2
                X[gidx, (i+1) * discr + j] = -1/2
                X[gidx, i * discr + (j-1)] = -1/2
                X[gidx, i * discr + (j+1)] = -1/2
                X[gidx, (i+1) * discr + (j+1)] = 1/4
                X[gidx, (i+1) * discr + (j-1)] = 1/4
                X[gidx, (i-1) * discr + (j+1)] = 1/4
                X[gidx, (i-1) * discr + (j-1)] = 1/4
                bx[gidx] = 0.0
    for i in range(discr):
        for j in range(discr):
            gidx = i*discr + j
            if j == 0:
                Y[gidx, i*discr + j] = 1.0
                by[gidx] = b_y_val[i]
            elif j == discr - 1:
                Y[gidx, i*discr + j] = 1.0
                by[gidx] = t_y_val[i]
            elif i == 0:
                Y[gidx, i*discr + j] = 1.0
                by[gidx] = l_y_val[j]
            elif i == discr - 1:
                Y[gidx, i*discr + j] = 1.0
                by[gidx] = r_y_val[j]
            elif nod != None and i == nod[1]-1 and j == nod[0]-1:
                Y[gidx, i * discr + j] = 1.0
                by[gidx] = nod[3]
            else:
                Y[gidx, i * discr + j] = 1.0
                Y[gidx, (i-1) * discr + j] = -1/2
                Y[gidx, (i+1) * discr + j] = -1/2
                Y[gidx, i * discr + (j-1)] = -1/2
                Y[gidx, i * discr + (j+1)] = -1/2
                Y[gidx, (i+1) * discr + (j+1)] = 1/4
                Y[gidx, (i+1) * discr + (j-1)] = 1/4
                Y[gidx, (i-1) * discr + (j+1)] = 1/4
                Y[gidx, (i-1) * discr + (j-1)] = 1/4
                by[gidx] = 0.0
    X_res = (np.linalg.solve(X, bx)).reshape((n, m))
    Y_res = (np.linalg.solve(Y, by)).reshape((n, m))
    return [X_res, Y_res]


def transfinite_interpol(discr, Xt, Xb, Xr, Xl):
    m = discr
    n = discr

    xi = np.linspace(0., 1., m)  # дискретизация границ
    eta = np.linspace(0., 1., n)

    X = np.zeros((m, n))  # матрица (позиция иксов исходя из дискретизации; иксы на этой позиции)
    Y = np.zeros((m, n))  # то же самое, но с игреками

    for i in range(0, m):
        Xi = xi[i]
        for j in range(0, n):
            Eta = eta[j]

            #  мейн формула TFI
            XY = (1 - Eta) * Xb(Xi) + Eta * Xt(Xi) + (1 - Xi) * Xl(Eta) + Xi * Xr(Eta) \
                 - (Xi * Eta * Xt(1) + Xi * (1 - Eta) * Xb(1) + Eta * (1 - Xi) * Xt(0) + (1 - Xi) * (1 - Eta) * Xb(0))

            X[i][j] = XY[0]  # добавляем новый x
            Y[i][j] = XY[1]  # добавляем новый y

    result = [X, Y]
    return result


def thomaxsolver(a, b, c, d):
    n = len(d)
    x = np.zeros(n)
    g_mod = [c[0] / b[0]]
    for j in range(1, n-1):
        gg = c[j] / (b[j] - g_mod[j-1]*a[j])
        g_mod.append(gg)
    d_mod = [d[0] / b[0]]
    for j in range(1, n):
        dd = (d[j] - d_mod[j-1]*a[j]) / (b[j] - g_mod[j-1]*a[j])
        d_mod.append(dd)
    x[n-1] = d_mod[-1]
    for j in range(n-2, -1, -1):
        x[j] = d_mod[j] - g_mod[j]*x[j+1]
    return x


def assembleCoeff(b, a, deTerm, dTerm, eTerm, x_new, y_new, deltaZi, deltaEta):
    length, height = x_new.shape

    for i in range(1, length - 1):
        for j in range(1, height - 1):
            xiNext = x_new[j, i + 1]
            xiPrev = x_new[j, i - 1]
            xjNext = x_new[j + 1, i]
            xjPrev = x_new[j - 1, i]

            yiNext = y_new[j, i + 1]
            yiPrev = y_new[j, i - 1]
            yjNext = y_new[j + 1, i]
            yjPrev = y_new[j - 1, i]

            xijNext = x_new[j + 1, i + 1]
            xijPrev = x_new[j - 1, i - 1]
            xiPrevjNext = x_new[j + 1, i - 1]
            xiNextjPrev = x_new[j - 1, i + 1]

            yijNext = y_new[j + 1, i + 1]
            yijPrev = y_new[j - 1, i - 1]
            yiPrevjNext = y_new[j + 1, i - 1]
            yiNextjPrev = y_new[j - 1, i + 1]

            x1 = 0.5 * (xiNext - xiPrev) / deltaZi
            x2 = 0.5 * (xjNext - xjPrev) / deltaEta
            y1 = 0.5 * (yiNext - yiPrev) / deltaZi
            y2 = 0.5 * (yjNext - yjPrev) / deltaEta

            g11 = x1 * x1 + y1 * y1
            g22 = x2 * x2 + y2 * y2
            g12 = x1 * x2 + y1 * y2

            b[j, i] = 2.0 * (g11 / (deltaEta ** 2) + g22 / (deltaZi ** 2))
            a[j, i] = g11 / (deltaEta ** 2)
            deTerm[j, i] = g22 / (deltaZi ** 2)
            dTerm[j, i] = -0.5 * g12 * (xijNext + xijPrev - xiNextjPrev - xiPrevjNext) / (deltaZi * deltaEta)
            eTerm[j, i] = -0.5 * g12 * (yijNext + yijPrev - yiNextjPrev - yiPrevjNext) / (deltaZi * deltaEta)


def solveTDMAN(phi, a, b, deTerm, dTerm):
    length, height = phi.shape
    imax, jmax = length - 1, height - 1
    imin, jmin = 0, 0
    P = np.zeros(length)
    Q = np.zeros(length)
    bArr = np.zeros(length)

    # Set P(1) to 0 since a(1) = c(1) = 0
    P[jmin] = 0.0

    # Start West-East sweep
    for i in range(imin + 1, imax):
        # Set Q(1) to x(1) since x(i) = P(i)x(i+1) + Q(i) and P(1) = 0
        Q[jmin] = phi[jmin, i]

        # Start South-North traverse
        for j in range(jmin + 1, jmax):
            # Assemble TDMA coefficients, rename North, South, East and West as follows
            # Store a's = c's in P
            P[j] = a[j, i]
            # Store d's/e's in Qx/Qy
            Q[j] = deTerm[j, i] * (phi[j, i + 1] + phi[j, i - 1]) + dTerm[j, i]
            # Store b's in bArr
            bArr[j] = b[j, i]

            # Calculate coefficients of recursive formula
            term = 1.0 / (bArr[j] - P[j] * P[j - 1])
            Q[j] = (Q[j] + P[j] * Q[j - 1]) * term
            P[j] = P[j] * term

        # Obtain new values of phi (either x or y)
        for j in range(jmax - 1, jmin, -1):
            phi[j, i] = P[j] * phi[j + 1, i] + Q[j]


def compute_max_diff(phi_new, phi_old):
    height, length = phi_new.shape

    # Инициализируем список разниц
    maxlist = []
    # Проходим по всем элементам массива, исключая границы
    for j in range(1, height-1):
        for i in range(1, length-1):
            phidiff = np.abs(phi_new[j, i] - phi_old[j, i])
            maxlist.append(phidiff)
    maxnum = max(maxlist)
    return maxnum


def winslow_with_implicit(discr, Xt, Xb, Xr, Xl, treshhold):
    x_old = implicit_transfinite_interpol(discr, Xt, Xb, Xr, Xl)[0]
    y_old = implicit_transfinite_interpol(discr, Xt, Xb, Xr, Xl)[1]
    height, length = x_old.shape
    deltaZi = 1.0
    deltaEta = 1.0
    xdiffmax = 100
    ydiffmax = 100
    x_new = x_old.copy()
    y_new = y_old.copy()
    count = 1
    while xdiffmax > treshhold or ydiffmax > treshhold:
        b = np.zeros((height, length))
        a = np.zeros((height, length))
        deTerm = np.zeros((height, length))
        dTerm = np.zeros((height, length))
        eTerm = np.zeros((height, length))
        x_old = x_new.copy()
        y_old = y_new.copy()
        assembleCoeff(b, a, deTerm, dTerm, eTerm, x_new, y_new, deltaZi, deltaEta)
        solveTDMAN(x_new, a, b, deTerm, dTerm)

        for i in range(1, length - 1):
            for j in range(1, height - 1):
                dTerm[j, i] = eTerm[j, i]
        solveTDMAN(y_new, a, b, deTerm, dTerm)

        xdiffmax = compute_max_diff(x_new, x_old)
        ydiffmax = compute_max_diff(y_new, y_old)
        print(max(xdiffmax, ydiffmax))
        count += 1

    return np.array([x_new, y_new]), count


def winslow_without_implicit(discr, Xt, Xb, Xr, Xl, treshhold, iter_count):
    x_old = transfinite_interpol(discr, Xt, Xb, Xr, Xl)[0]
    y_old = transfinite_interpol(discr, Xt, Xb, Xr, Xl)[1]
    height, length = x_old.shape
    deltaZi = 1.0
    deltaEta = 1.0
    xdiffmax = 100
    ydiffmax = 100
    x_new = x_old.copy()
    y_new = y_old.copy()
    count = 1
    while (xdiffmax > treshhold or ydiffmax > treshhold) and count < iter_count:
        b = np.zeros((height, length))
        a = np.zeros((height, length))
        deTerm = np.zeros((height, length))
        dTerm = np.zeros((height, length))
        eTerm = np.zeros((height, length))
        x_old = x_new.copy()
        y_old = y_new.copy()
        assembleCoeff(b, a, deTerm, dTerm, eTerm, x_new, y_new, deltaZi, deltaEta)
        solveTDMAN(x_new, a, b, deTerm, dTerm)

        for i in range(1, length - 1):
            for j in range(1, height - 1):
                dTerm[j, i] = eTerm[j, i]
        solveTDMAN(y_new, a, b, deTerm, dTerm)

        xdiffmax = compute_max_diff(x_new, x_old)
        ydiffmax = compute_max_diff(y_new, y_old)
        print(max(xdiffmax, ydiffmax))
        count += 1

    return np.array([x_new, y_new]), count


def winslow(discr, x_old, y_old, treshhold, maxcount):
    height, length = x_old.shape
    deltaZi = 1.0
    deltaEta = 1.0
    xdiffmax = 100
    ydiffmax = 100
    x_new = x_old.copy()
    y_new = y_old.copy()
    count = 1
    while (xdiffmax > treshhold or ydiffmax > treshhold) and count < maxcount:
        b = np.zeros((height, length))
        a = np.zeros((height, length))
        deTerm = np.zeros((height, length))
        dTerm = np.zeros((height, length))
        eTerm = np.zeros((height, length))
        x_old = x_new.copy()
        y_old = y_new.copy()
        assembleCoeff(b, a, deTerm, dTerm, eTerm, x_new, y_new, deltaZi, deltaEta)
        solveTDMAN(x_new, a, b, deTerm, dTerm)

        for i in range(1, length - 1):
            for j in range(1, height - 1):
                dTerm[j, i] = eTerm[j, i]
        solveTDMAN(y_new, a, b, deTerm, dTerm)

        xdiffmax = compute_max_diff(x_new, x_old)
        ydiffmax = compute_max_diff(y_new, y_old)
        count += 1

    return np.array([x_new, y_new]), count


def pca_bounding_box(points):
    # Центрирование
    mean = points.mean(axis=0)
    centered = points - mean

    # PCA
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(centered)

    # Прямоугольник в PCA-пространстве
    min_vals = np.min(transformed, axis=0)
    max_vals = np.max(transformed, axis=0)

    rect_pca = np.array([
        [min_vals[0], min_vals[1]],
        [max_vals[0], min_vals[1]],
        [max_vals[0], max_vals[1]],
        [min_vals[0], max_vals[1]],
    ])

    # Обратное преобразование
    rect_global = pca.inverse_transform(rect_pca) + mean
    return rect_global


def functional2(grid_omega, cells, grid_shape):
    cells_x = cells[0]
    cells_y = cells[1]
    nx, ny = grid_shape
    aproximation = 0
    x = grid_omega[0]
    y = grid_omega[1]
    x = np.reshape(x, (nx, ny))
    y = np.reshape(y, (nx, ny))

    flag = 0

    # Производные функционала в точках сетки
    Rx = np.zeros((nx, ny))
    Ry = np.zeros((nx, ny))
    Rxx = np.zeros((nx, ny))
    Ryy = np.zeros((nx, ny))
    Rxy = np.zeros((nx, ny))
    weight = 1

    for i in range(nx - 1):
        for j in range(ny - 1):
            # вот здесь X Y определять
            X1 = cells_x[i, j][0, 0]
            X2 = cells_x[i, j][0, 1]
            X3 = cells_x[i, j][1, 1]
            X4 = cells_x[i, j][1, 0]
            Y1 = cells_y[i, j][0, 0]
            Y2 = cells_y[i, j][0, 1]
            Y3 = cells_y[i, j][1, 1]
            Y4 = cells_y[i, j][1, 0]
            for k in range(4):
                if k == 0:  # lb corner perf; k = i,j; k+1 = i,j+1; k-1 = i+1,j
                    G11 = ((X2 - X1) ** 2) + ((Y2 - Y1) ** 2)
                    G12 = (X2 - X1) * (X4 - X1) + (Y2 - Y1) * (
                                Y4 - Y1)
                    G22 = (X4 - X1) ** 2 + (Y4 - Y1) ** 2
                    Jk = (x[i, j + 1] - x[i, j]) * (y[i + 1, j] - y[i, j]) - (x[i + 1, j] - x[i, j]) * (
                                y[i, j + 1] - y[i, j])
                    if Jk <= 0:
                        print("1: flag")
                        flag = 1
                    Dk = (X2 - X1) * (Y4 - Y1) - (X4 - X1) * (
                                Y2 - Y1)
                    alpha = ((x[i, j + 1] - x[i, j]) ** 2) * G22 - 2 * (x[i, j + 1] - x[i, j]) * (
                            x[i + 1, j] - x[i, j]) * G12 + ((
                                                                    x[i + 1, j] - x[i, j]) ** 2) * G11
                    gamma = ((y[i, j + 1] - y[i, j]) ** 2) * G22 - 2 * (y[i, j + 1] - y[i, j]) * (
                            y[i + 1, j] - y[i, j]) * G12 + ((y[i + 1, j] - y[i, j]) ** 2) * G11

                    numerator = alpha + gamma
                    denominator = Jk * Dk

                    Fk = (numerator / denominator)
                    aproximation += Fk

                    # DERIVS:
                    x1 = x[i, j]
                    x2 = x[i, j + 1]
                    x3 = x[i + 1, j]
                    y1 = y[i, j]
                    y2 = y[i, j + 1]
                    y3 = y[i + 1, j]

                    # xk : x[i, j]  done
                    Fx = (-(G22 * ((x1 - x2)**2 + (y1 - y2)**2) -
                            2 * G12 * ((x1 - x2) * (x1 - x3) + (y1 - y2) * (y1 - y3)) +
                                G11 * ((x1 - x3)**2 + (y1 - y3)**2)) * (y2 - y3) +
                            2 * (G22 * (x1 - x2) + G11 * (x1 - x3) +
                                    G12 * (-2 * x1 + x2 + x3)) *
                            (x3 * (y1 - y2) + x1 * (y2 - y3) + x2 * (-y1 + y3))) / \
                                (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3))**2)
                    Fy = -(
                        ((-x2 + x3) * (G22 * ((x1 - x2)**2 + (y1 - y2)**2) -
                                       2 * G12 * ((x1 - x2) * (x1 - x3) + (y1 - y2) * (y1 - y3)) +
                                       G11 * ((x1 - x3)**2 + (y1 - y3)**2)) +
                         2 * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) *
                           (G22 * (y1 - y2) + G11 * (y1 - y3) + G12 * (-2 * y1 + y2 + y3)))
                        / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3))**2)
                    )
                    Fxx = -(
                        2 * ((x2 - x3)**2 + (y2 - y3)**2) *
                        (G11 * (y1 - y3)**2 + (y1 - y2) * (G22 * (y1 - y2) + 2 * G12 * (-y1 + y3)))
                        / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3))**3)
                    )
                    Fyy = -(
                        2 * (G11 * (x1 - x3)**2 + (x1 - x2) * (G22 * (x1 - x2) + 2 * G12 * (-x1 + x3))) *
                        ((x2 - x3)**2 + (y2 - y3)**2) /
                        (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3))**3)
                    )
                    Fxy = (
                        2 * ((x2 - x3)**2 + (y2 - y3)**2) *
                        (G22 * (x1 - x2) * (y1 - y2) +
                         G11 * (x1 - x3) * (y1 - y3) +
                         G12 * (x2 * y1 + x3 * y1 - x3 * y2 - x2 * y3 + x1 * (-2 * y1 + y2 + y3)))
                        / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3))**3)
                    )
                    Rx[i, j] += Fx / weight
                    Rxy[i, j] += Fxy / weight
                    Ry[i, j] += Fy / weight
                    Rxx[i, j] += Fxx / weight
                    Ryy[i, j] += Fyy / weight

                    # xk+1 : x[i, j + 1] done
                    Fx = (
                        (-2 * G12 * (y1 - y2) * ((x1 - x3)**2 + (y1 - y3)**2) +
                         G11 * ((x1 - x3)**2 + (y1 - y3)**2) * (y1 - y3) +
                         G22 * (2 * x2 * x3 * (y1 - y2) +
                                2 * x1 * (x3 * (-y1 + y2) + x2 * (y2 - y3)) +
                                (y1 - y2)**2 * (y1 - y3) +
                                x2**2 * (-y1 + y3) +
                                x1**2 * (y1 - 2 * y2 + y3))
                        ) / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3))**2)
                    )
                    Fy = result = -(
                        (G22 * (x1**3 - x2**2 * x3 - x1**2 * (2 * x2 + x3) + x3 * (y1 - y2)**2 +
                                x1 * (x2**2 + 2 * x2 * x3 + (y1 - y2) * (y1 + y2 - 2 * y3)) -
                                2 * x2 * (y1 - y2) * (y1 - y3)) -
                         2 * G12 * (x1 - x2) * ((x1 - x3)**2 + (y1 - y3)**2) +
                         G11 * (x1 - x3) * ((x1 - x3)**2 + (y1 - y3)**2))
                        / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3))**2)
                    )
                    Fxx = -(
                        2 * ((x1 - x3)**2 + (y1 - y3)**2) *
                        (G11 * (y1 - y3)**2 + (y1 - y2) * (G22 * (y1 - y2) + 2 * G12 * (-y1 + y3)))
                        / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3))**3)
                    )
                    Fyy = -(
                        2 * (G11 * (x1 - x3)**2 + (x1 - x2) * (G22 * (x1 - x2) + 2 * G12 * (-x1 + x3))) *
                        ((x1 - x3)**2 + (y1 - y3)**2) /
                        (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3))**3)
                    )
                    Fxy = (
                        2 * ((x1 - x3)**2 + (y1 - y3)**2) *
                        (G22 * (x1 - x2) * (y1 - y2) +
                         G11 * (x1 - x3) * (y1 - y3) +
                         G12 * (x2 * y1 + x3 * y1 - x3 * y2 - x2 * y3 + x1 * (-2 * y1 + y2 + y3)))
                        / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3))**3)
                    )
                    Rx[i, j + 1] += Fx / weight
                    Rxy[i, j + 1] += Fxy / weight
                    Ry[i, j + 1] += Fy / weight
                    Rxx[i, j + 1] += Fxx / weight
                    Ryy[i, j + 1] += Fyy / weight

                    # xk-1 : x[i + 1, j] done
                    Fx = (
                        ((x1 - x2)**2 + (y1 - y2)**2) * (G22 * (-y1 + y2) + 2 * G12 * (y1 - y3)) -
                        G11 * (-2 * x1 * (x2 * y1 + x3 * y2) +
                               (y1 - y2) * (-x3**2 + (y1 - y3)**2) +
                               x1**2 * (y1 + y2 - 2 * y3) +
                               2 * x2 * x3 * (y1 - y3) +
                               2 * x1 * (x2 + x3) * y3)
                    ) / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3))**2)
                    Fy = (
                        (G22 * (x1 - x2) + 2 * G12 * (-x1 + x3)) * ((x1 - x2)**2 + (y1 - y2)**2) +
                        G11 * (x1**3 - x1**2 * (x2 + 2 * x3) + x2 * (-x3**2 + (y1 - y3)**2) -
                               2 * x3 * (y1 - y2) * (y1 - y3) +
                               x1 * (2 * x2 * x3 + x3**2 + (y1 - y3) * (y1 - 2 * y2 + y3)))
                    ) / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3))**2)
                    Fxx = -(
                        2 * ((x1 - x2)**2 + (y1 - y2)**2) *
                        (G11 * (y1 - y3)**2 + (y1 - y2) * (G22 * (y1 - y2) + 2 * G12 * (-y1 + y3)))
                        / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3))**3)
                    )
                    Fyy = -(
                        2 * (G11 * (x1 - x3)**2 + (x1 - x2) * (G22 * (x1 - x2) + 2 * G12 * (-x1 + x3))) *
                        ((x1 - x2)**2 + (y1 - y2)**2) /
                        (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3))**3)
                    )
                    Fxy = (
                        2 * ((x1 - x2)**2 + (y1 - y2)**2) *
                        (G22 * (x1 - x2) * (y1 - y2) +
                         G11 * (x1 - x3) * (y1 - y3) +
                         G12 * (x2 * y1 + x3 * y1 - x3 * y2 - x2 * y3 + x1 * (-2 * y1 + y2 + y3)))
                        / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3))**3)
                    )
                    Rx[i + 1, j] += Fx / weight
                    Rxy[i + 1, j] += Fxy / weight
                    Ry[i + 1, j] += Fy / weight
                    Rxx[i + 1, j] += Fxx / weight
                    Ryy[i + 1, j] += Fyy / weight

                elif k == 1:    # rt corner perf xk=x[i+1, j+1], xk+1=x[i+1, j], xk-1=x[i, j+1]
                    G11 = ((X4 - X3) ** 2) + (Y4 - Y3) ** 2
                    G12 = (X4 - X3) * (X2 - X3) + (
                                Y4 - Y3) * (
                                  Y2 - Y3)
                    G22 = ((X2 - X3) ** 2) + (Y2 - Y3) ** 2
                    Jk = (x[i + 1, j] - x[i + 1, j + 1]) * (y[i, j + 1] - y[i + 1, j + 1]) - (
                                x[i, j + 1] - x[i + 1, j + 1]) * (
                                 y[i + 1, j] - y[i + 1, j + 1])
                    if Jk <= 0:
                        print("2: flag")
                        flag = 1
                    Dk = (X4 - X3) * (Y2 - Y3) - (
                                X2 - X3) * (
                                 Y4 - Y3)
                    alpha = ((x[i + 1, j] - x[i + 1, j + 1]) ** 2) * G22 - 2 * (x[i + 1, j] - x[i + 1, j + 1]) * (
                            x[i, j + 1] - x[i + 1, j + 1]) * G12 + ((
                                                                            x[i, j + 1] - x[i + 1, j + 1]) ** 2) * G11
                    gamma = ((y[i + 1, j] - y[i + 1, j + 1]) ** 2) * G22 - 2 * (y[i + 1, j] - y[i + 1, j + 1]) * (
                            y[i, j + 1] - y[i + 1, j + 1]) * G12 + ((
                                                                            y[i, j + 1] - y[i + 1, j + 1]) ** 2) * G11

                    numerator = alpha + gamma
                    denominator = Jk * Dk

                    Fk = (numerator / denominator)
                    aproximation += Fk

                    # DERIVS:

                    x1 = x[i + 1, j + 1]
                    x2 = x[i + 1, j]
                    x3 = x[i, j + 1]
                    y1 = y[i + 1, j + 1]
                    y2 = y[i + 1, j]
                    y3 = y[i, j + 1]

                    # xk : x[i + 1, j + 1]
                    Fx = (-(G22 * ((x1 - x2) ** 2 + (y1 - y2) ** 2) -
                            2 * G12 * ((x1 - x2) * (x1 - x3) + (y1 - y2) * (y1 - y3)) +
                            G11 * ((x1 - x3) ** 2 + (y1 - y3) ** 2)) * (y2 - y3) +
                          2 * (G22 * (x1 - x2) + G11 * (x1 - x3) +
                               G12 * (-2 * x1 + x2 + x3)) *
                          (x3 * (y1 - y2) + x1 * (y2 - y3) + x2 * (-y1 + y3))) / \
                         (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 2)
                    Fy = -(
                            ((-x2 + x3) * (G22 * ((x1 - x2) ** 2 + (y1 - y2) ** 2) -
                                           2 * G12 * ((x1 - x2) * (x1 - x3) + (y1 - y2) * (y1 - y3)) +
                                           G11 * ((x1 - x3) ** 2 + (y1 - y3) ** 2)) +
                             2 * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) *
                             (G22 * (y1 - y2) + G11 * (y1 - y3) + G12 * (-2 * y1 + y2 + y3)))
                            / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 2)
                    )
                    Fxx = -(
                            2 * ((x2 - x3) ** 2 + (y2 - y3) ** 2) *
                            (G11 * (y1 - y3) ** 2 + (y1 - y2) * (G22 * (y1 - y2) + 2 * G12 * (-y1 + y3)))
                            / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 3)
                    )
                    Fyy = -(
                            2 * (G11 * (x1 - x3) ** 2 + (x1 - x2) * (G22 * (x1 - x2) + 2 * G12 * (-x1 + x3))) *
                            ((x2 - x3) ** 2 + (y2 - y3) ** 2) /
                            (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 3)
                    )
                    Fxy = (
                            2 * ((x2 - x3) ** 2 + (y2 - y3) ** 2) *
                            (G22 * (x1 - x2) * (y1 - y2) +
                             G11 * (x1 - x3) * (y1 - y3) +
                             G12 * (x2 * y1 + x3 * y1 - x3 * y2 - x2 * y3 + x1 * (-2 * y1 + y2 + y3)))
                            / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 3)
                    )
                    Rx[i + 1, j + 1] += Fx / weight
                    Rxy[i + 1, j + 1] += Fxy / weight
                    Ry[i + 1, j + 1] += Fy / weight
                    Rxx[i + 1, j + 1] += Fxx / weight
                    Ryy[i + 1, j + 1] += Fyy / weight

                    # xk+1 : x[i + 1, j]
                    Fx = (
                            (-2 * G12 * (y1 - y2) * ((x1 - x3) ** 2 + (y1 - y3) ** 2) +
                             G11 * ((x1 - x3) ** 2 + (y1 - y3) ** 2) * (y1 - y3) +
                             G22 * (2 * x2 * x3 * (y1 - y2) +
                                    2 * x1 * (x3 * (-y1 + y2) + x2 * (y2 - y3)) +
                                    (y1 - y2) ** 2 * (y1 - y3) +
                                    x2 ** 2 * (-y1 + y3) +
                                    x1 ** 2 * (y1 - 2 * y2 + y3))
                             ) / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 2)
                    )
                    Fy = result = -(
                            (G22 * (x1 ** 3 - x2 ** 2 * x3 - x1 ** 2 * (2 * x2 + x3) + x3 * (y1 - y2) ** 2 +
                                    x1 * (x2 ** 2 + 2 * x2 * x3 + (y1 - y2) * (y1 + y2 - 2 * y3)) -
                                    2 * x2 * (y1 - y2) * (y1 - y3)) -
                             2 * G12 * (x1 - x2) * ((x1 - x3) ** 2 + (y1 - y3) ** 2) +
                             G11 * (x1 - x3) * ((x1 - x3) ** 2 + (y1 - y3) ** 2))
                            / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 2)
                    )
                    Fxx = -(
                            2 * ((x1 - x3) ** 2 + (y1 - y3) ** 2) *
                            (G11 * (y1 - y3) ** 2 + (y1 - y2) * (G22 * (y1 - y2) + 2 * G12 * (-y1 + y3)))
                            / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 3)
                    )
                    Fyy = -(
                            2 * (G11 * (x1 - x3) ** 2 + (x1 - x2) * (G22 * (x1 - x2) + 2 * G12 * (-x1 + x3))) *
                            ((x1 - x3) ** 2 + (y1 - y3) ** 2) /
                            (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 3)
                    )
                    Fxy = (
                            2 * ((x1 - x3) ** 2 + (y1 - y3) ** 2) *
                            (G22 * (x1 - x2) * (y1 - y2) +
                             G11 * (x1 - x3) * (y1 - y3) +
                             G12 * (x2 * y1 + x3 * y1 - x3 * y2 - x2 * y3 + x1 * (-2 * y1 + y2 + y3)))
                            / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 3)
                    )
                    Rx[i + 1, j] += Fx / weight
                    Rxy[i + 1, j] += Fxy / weight
                    Ry[i + 1, j] += Fy / weight
                    Rxx[i + 1, j] += Fxx / weight
                    Ryy[i + 1, j] += Fyy / weight

                    # xk-1 : x[i, j + 1]
                    Fx = (
                                 ((x1 - x2) ** 2 + (y1 - y2) ** 2) * (G22 * (-y1 + y2) + 2 * G12 * (y1 - y3)) -
                                 G11 * (-2 * x1 * (x2 * y1 + x3 * y2) +
                                        (y1 - y2) * (-x3 ** 2 + (y1 - y3) ** 2) +
                                        x1 ** 2 * (y1 + y2 - 2 * y3) +
                                        2 * x2 * x3 * (y1 - y3) +
                                        2 * x1 * (x2 + x3) * y3)
                         ) / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 2)
                    Fy = (
                                 (G22 * (x1 - x2) + 2 * G12 * (-x1 + x3)) * ((x1 - x2) ** 2 + (y1 - y2) ** 2) +
                                 G11 * (x1 ** 3 - x1 ** 2 * (x2 + 2 * x3) + x2 * (-x3 ** 2 + (y1 - y3) ** 2) -
                                        2 * x3 * (y1 - y2) * (y1 - y3) +
                                        x1 * (2 * x2 * x3 + x3 ** 2 + (y1 - y3) * (y1 - 2 * y2 + y3)))
                         ) / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 2)
                    Fxx = -(
                            2 * ((x1 - x2) ** 2 + (y1 - y2) ** 2) *
                            (G11 * (y1 - y3) ** 2 + (y1 - y2) * (G22 * (y1 - y2) + 2 * G12 * (-y1 + y3)))
                            / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 3)
                    )
                    Fyy = -(
                            2 * (G11 * (x1 - x3) ** 2 + (x1 - x2) * (G22 * (x1 - x2) + 2 * G12 * (-x1 + x3))) *
                            ((x1 - x2) ** 2 + (y1 - y2) ** 2) /
                            (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 3)
                    )
                    Fxy = (
                            2 * ((x1 - x2) ** 2 + (y1 - y2) ** 2) *
                            (G22 * (x1 - x2) * (y1 - y2) +
                             G11 * (x1 - x3) * (y1 - y3) +
                             G12 * (x2 * y1 + x3 * y1 - x3 * y2 - x2 * y3 + x1 * (-2 * y1 + y2 + y3)))
                            / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 3)
                    )
                    Rx[i, j + 1] += Fx / weight
                    Rxy[i, j + 1] += Fxy / weight
                    Ry[i, j + 1] += Fy / weight
                    Rxx[i, j + 1] += Fxx / weight
                    Ryy[i, j + 1] += Fyy / weight

                elif k == 2:    # rb corner perf xk = x[i, j + 1], xk+1=x[i+1, j+1], xk-1=x[i, j]
                    G11 = ((X3 - X2) ** 2) + (Y3 - Y2) ** 2
                    G12 = (X3 - X2) * (X1 - X2) + (
                            Y3 - Y2) * (
                                  Y1 - Y2)
                    G22 = (X1 - X2) ** 2 + (Y1 - Y2) ** 2
                    Jk = (x[i + 1, j + 1] - x[i, j + 1]) * (y[i, j] - y[i, j + 1]) - (x[i, j] - x[i, j + 1]) * (
                            y[i + 1, j + 1] - y[i, j + 1])
                    if Jk <= 0:
                        print("3: flag")
                        flag = 1
                    Dk = (X3 - X2) * (Y1 - Y2) - (X1 - X2) * (
                            Y3 - Y2)
                    alpha = ((x[i + 1, j + 1] - x[i, j + 1]) ** 2) * G22 - 2 * (x[i + 1, j + 1] - x[i, j + 1]) * (
                            x[i, j] - x[i, j + 1]) * G12 + ((
                                                                    x[i, j] - x[i, j + 1]) ** 2) * G11
                    gamma = ((y[i + 1, j + 1] - y[i, j + 1]) ** 2) * G22 - 2 * (y[i + 1, j + 1] - y[i, j + 1]) * (
                            y[i, j] - y[i, j + 1]) * G12 + ((
                                                                    y[i, j] - y[i, j + 1]) ** 2) * G11

                    numerator = alpha + gamma
                    denominator = Jk * Dk

                    Fk = (numerator / denominator)
                    aproximation += Fk

                    # DERIVS:

                    x1 = x[i, j + 1]
                    x2 = x[i + 1, j + 1]
                    x3 = x[i, j]
                    y1 = y[i, j + 1]
                    y2 = y[i + 1, j + 1]
                    y3 = y[i, j]

                    # xk : x[i , j + 1]
                    Fx = (-(G22 * ((x1 - x2) ** 2 + (y1 - y2) ** 2) -
                            2 * G12 * ((x1 - x2) * (x1 - x3) + (y1 - y2) * (y1 - y3)) +
                            G11 * ((x1 - x3) ** 2 + (y1 - y3) ** 2)) * (y2 - y3) +
                          2 * (G22 * (x1 - x2) + G11 * (x1 - x3) +
                               G12 * (-2 * x1 + x2 + x3)) *
                          (x3 * (y1 - y2) + x1 * (y2 - y3) + x2 * (-y1 + y3))) / \
                         (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 2)
                    Fy = -(
                            ((-x2 + x3) * (G22 * ((x1 - x2) ** 2 + (y1 - y2) ** 2) -
                                           2 * G12 * ((x1 - x2) * (x1 - x3) + (y1 - y2) * (y1 - y3)) +
                                           G11 * ((x1 - x3) ** 2 + (y1 - y3) ** 2)) +
                             2 * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) *
                             (G22 * (y1 - y2) + G11 * (y1 - y3) + G12 * (-2 * y1 + y2 + y3)))
                            / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 2)
                    )
                    Fxx = -(
                            2 * ((x2 - x3) ** 2 + (y2 - y3) ** 2) *
                            (G11 * (y1 - y3) ** 2 + (y1 - y2) * (G22 * (y1 - y2) + 2 * G12 * (-y1 + y3)))
                            / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 3)
                    )
                    Fyy = -(
                            2 * (G11 * (x1 - x3) ** 2 + (x1 - x2) * (G22 * (x1 - x2) + 2 * G12 * (-x1 + x3))) *
                            ((x2 - x3) ** 2 + (y2 - y3) ** 2) /
                            (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 3)
                    )
                    Fxy = (
                            2 * ((x2 - x3) ** 2 + (y2 - y3) ** 2) *
                            (G22 * (x1 - x2) * (y1 - y2) +
                             G11 * (x1 - x3) * (y1 - y3) +
                             G12 * (x2 * y1 + x3 * y1 - x3 * y2 - x2 * y3 + x1 * (-2 * y1 + y2 + y3)))
                            / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 3)
                    )
                    Rx[i, j + 1] += Fx / weight
                    Rxy[i, j + 1] += Fxy / weight
                    Ry[i, j + 1] += Fy / weight
                    Rxx[i, j + 1] += Fxx / weight
                    Ryy[i, j + 1] += Fyy / weight

                    # xk+1 : x[i + 1, j + 1]
                    Fx = (
                            (-2 * G12 * (y1 - y2) * ((x1 - x3) ** 2 + (y1 - y3) ** 2) +
                             G11 * ((x1 - x3) ** 2 + (y1 - y3) ** 2) * (y1 - y3) +
                             G22 * (2 * x2 * x3 * (y1 - y2) +
                                    2 * x1 * (x3 * (-y1 + y2) + x2 * (y2 - y3)) +
                                    (y1 - y2) ** 2 * (y1 - y3) +
                                    x2 ** 2 * (-y1 + y3) +
                                    x1 ** 2 * (y1 - 2 * y2 + y3))
                             ) / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 2)
                    )
                    Fy = result = -(
                            (G22 * (x1 ** 3 - x2 ** 2 * x3 - x1 ** 2 * (2 * x2 + x3) + x3 * (y1 - y2) ** 2 +
                                    x1 * (x2 ** 2 + 2 * x2 * x3 + (y1 - y2) * (y1 + y2 - 2 * y3)) -
                                    2 * x2 * (y1 - y2) * (y1 - y3)) -
                             2 * G12 * (x1 - x2) * ((x1 - x3) ** 2 + (y1 - y3) ** 2) +
                             G11 * (x1 - x3) * ((x1 - x3) ** 2 + (y1 - y3) ** 2))
                            / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 2)
                    )
                    Fxx = -(
                            2 * ((x1 - x3) ** 2 + (y1 - y3) ** 2) *
                            (G11 * (y1 - y3) ** 2 + (y1 - y2) * (G22 * (y1 - y2) + 2 * G12 * (-y1 + y3)))
                            / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 3)
                    )
                    Fyy = -(
                            2 * (G11 * (x1 - x3) ** 2 + (x1 - x2) * (G22 * (x1 - x2) + 2 * G12 * (-x1 + x3))) *
                            ((x1 - x3) ** 2 + (y1 - y3) ** 2) /
                            (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 3)
                    )
                    Fxy = (
                            2 * ((x1 - x3) ** 2 + (y1 - y3) ** 2) *
                            (G22 * (x1 - x2) * (y1 - y2) +
                             G11 * (x1 - x3) * (y1 - y3) +
                             G12 * (x2 * y1 + x3 * y1 - x3 * y2 - x2 * y3 + x1 * (-2 * y1 + y2 + y3)))
                            / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 3)
                    )
                    Rx[i + 1, j + 1] += Fx / weight
                    Rxy[i + 1, j + 1] += Fxy / weight
                    Ry[i + 1, j + 1] += Fy / weight
                    Rxx[i + 1, j + 1] += Fxx / weight
                    Ryy[i + 1, j + 1] += Fyy / weight

                    # xk-1 : x[i, j]
                    Fx = (
                                 ((x1 - x2) ** 2 + (y1 - y2) ** 2) * (G22 * (-y1 + y2) + 2 * G12 * (y1 - y3)) -
                                 G11 * (-2 * x1 * (x2 * y1 + x3 * y2) +
                                        (y1 - y2) * (-x3 ** 2 + (y1 - y3) ** 2) +
                                        x1 ** 2 * (y1 + y2 - 2 * y3) +
                                        2 * x2 * x3 * (y1 - y3) +
                                        2 * x1 * (x2 + x3) * y3)
                         ) / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 2)
                    Fy = (
                                 (G22 * (x1 - x2) + 2 * G12 * (-x1 + x3)) * ((x1 - x2) ** 2 + (y1 - y2) ** 2) +
                                 G11 * (x1 ** 3 - x1 ** 2 * (x2 + 2 * x3) + x2 * (-x3 ** 2 + (y1 - y3) ** 2) -
                                        2 * x3 * (y1 - y2) * (y1 - y3) +
                                        x1 * (2 * x2 * x3 + x3 ** 2 + (y1 - y3) * (y1 - 2 * y2 + y3)))
                         ) / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 2)
                    Fxx = -(
                            2 * ((x1 - x2) ** 2 + (y1 - y2) ** 2) *
                            (G11 * (y1 - y3) ** 2 + (y1 - y2) * (G22 * (y1 - y2) + 2 * G12 * (-y1 + y3)))
                            / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 3)
                    )
                    Fyy = -(
                            2 * (G11 * (x1 - x3) ** 2 + (x1 - x2) * (G22 * (x1 - x2) + 2 * G12 * (-x1 + x3))) *
                            ((x1 - x2) ** 2 + (y1 - y2) ** 2) /
                            (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 3)
                    )
                    Fxy = (
                            2 * ((x1 - x2) ** 2 + (y1 - y2) ** 2) *
                            (G22 * (x1 - x2) * (y1 - y2) +
                             G11 * (x1 - x3) * (y1 - y3) +
                             G12 * (x2 * y1 + x3 * y1 - x3 * y2 - x2 * y3 + x1 * (-2 * y1 + y2 + y3)))
                            / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 3)
                    )
                    Rx[i, j] += Fx / weight
                    Rxy[i, j] += Fxy / weight
                    Ry[i, j] += Fy / weight
                    Rxx[i, j] += Fxx / weight
                    Ryy[i, j] += Fyy / weight

                elif k == 3:    # lt corner perf xk=x[i+1, j], xk+1=x[i, j], xk-1=x[i+1, j+1]
                    G11 = (X1 - X4) ** 2 + (Y1 - Y4) ** 2
                    G12 = (X1 - X4) * (X3 - X4) + (
                            Y1 - Y4) * (
                                  Y3 - Y4)
                    G22 = (X3 - X4) ** 2 + (Y3 - Y4) ** 2
                    Jk = (x[i, j] - x[i + 1, j]) * (y[i + 1, j + 1] - y[i + 1, j]) - (x[i + 1, j + 1] - x[i + 1, j]) * (
                            y[i, j] - y[i + 1, j])
                    if Jk <= 0:
                        print("4: flag")
                        flag = 1
                    Dk = (X1 - X4) * (Y3 - Y4) - (X3 - X4) * (
                            Y1 - Y4)
                    alpha = ((x[i, j] - x[i + 1, j]) ** 2) * G22 - 2 * (x[i, j] - x[i + 1, j]) * (
                            x[i + 1, j + 1] - x[i + 1, j]) * G12 + ((
                                                                            x[i + 1, j + 1] - x[i + 1, j]) ** 2) * G11
                    gamma = ((y[i, j] - y[i + 1, j]) ** 2) * G22 - 2 * (y[i, j] - y[i + 1, j]) * (
                            y[i + 1, j + 1] - y[i + 1, j]) * G12 + ((
                                                                            y[i + 1, j + 1] - y[i + 1, j]) ** 2) * G11

                    numerator = alpha + gamma
                    denominator = Jk * Dk

                    Fk = (numerator / denominator)
                    aproximation += Fk

                    # DERIVS:

                    x1 = x[i + 1, j]
                    x2 = x[i, j]
                    x3 = x[i + 1, j + 1]
                    y1 = y[i + 1, j]
                    y2 = y[i, j]
                    y3 = y[i + 1, j + 1]

                    # xk : x[i + 1, j]
                    Fx = (-(G22 * ((x1 - x2) ** 2 + (y1 - y2) ** 2) -
                            2 * G12 * ((x1 - x2) * (x1 - x3) + (y1 - y2) * (y1 - y3)) +
                            G11 * ((x1 - x3) ** 2 + (y1 - y3) ** 2)) * (y2 - y3) +
                          2 * (G22 * (x1 - x2) + G11 * (x1 - x3) +
                               G12 * (-2 * x1 + x2 + x3)) *
                          (x3 * (y1 - y2) + x1 * (y2 - y3) + x2 * (-y1 + y3))) / \
                         (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 2)
                    Fy = -(
                            ((-x2 + x3) * (G22 * ((x1 - x2) ** 2 + (y1 - y2) ** 2) -
                                           2 * G12 * ((x1 - x2) * (x1 - x3) + (y1 - y2) * (y1 - y3)) +
                                           G11 * ((x1 - x3) ** 2 + (y1 - y3) ** 2)) +
                             2 * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) *
                             (G22 * (y1 - y2) + G11 * (y1 - y3) + G12 * (-2 * y1 + y2 + y3)))
                            / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 2)
                    )
                    Fxx = -(
                            2 * ((x2 - x3) ** 2 + (y2 - y3) ** 2) *
                            (G11 * (y1 - y3) ** 2 + (y1 - y2) * (G22 * (y1 - y2) + 2 * G12 * (-y1 + y3)))
                            / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 3)
                    )
                    Fyy = -(
                            2 * (G11 * (x1 - x3) ** 2 + (x1 - x2) * (G22 * (x1 - x2) + 2 * G12 * (-x1 + x3))) *
                            ((x2 - x3) ** 2 + (y2 - y3) ** 2) /
                            (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 3)
                    )
                    Fxy = (
                            2 * ((x2 - x3) ** 2 + (y2 - y3) ** 2) *
                            (G22 * (x1 - x2) * (y1 - y2) +
                             G11 * (x1 - x3) * (y1 - y3) +
                             G12 * (x2 * y1 + x3 * y1 - x3 * y2 - x2 * y3 + x1 * (-2 * y1 + y2 + y3)))
                            / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 3)
                    )
                    Rx[i + 1, j] += Fx / weight
                    Rxy[i + 1, j] += Fxy / weight
                    Ry[i + 1, j] += Fy / weight
                    Rxx[i + 1, j] += Fxx / weight
                    Ryy[i + 1, j] += Fyy / weight

                    # xk+1 : x[i, j]
                    Fx = (
                            (-2 * G12 * (y1 - y2) * ((x1 - x3) ** 2 + (y1 - y3) ** 2) +
                             G11 * ((x1 - x3) ** 2 + (y1 - y3) ** 2) * (y1 - y3) +
                             G22 * (2 * x2 * x3 * (y1 - y2) +
                                    2 * x1 * (x3 * (-y1 + y2) + x2 * (y2 - y3)) +
                                    (y1 - y2) ** 2 * (y1 - y3) +
                                    x2 ** 2 * (-y1 + y3) +
                                    x1 ** 2 * (y1 - 2 * y2 + y3))
                             ) / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 2)
                    )
                    Fy = -(
                            (G22 * (x1 ** 3 - x2 ** 2 * x3 - x1 ** 2 * (2 * x2 + x3) + x3 * (y1 - y2) ** 2 +
                                    x1 * (x2 ** 2 + 2 * x2 * x3 + (y1 - y2) * (y1 + y2 - 2 * y3)) -
                                    2 * x2 * (y1 - y2) * (y1 - y3)) -
                             2 * G12 * (x1 - x2) * ((x1 - x3) ** 2 + (y1 - y3) ** 2) +
                             G11 * (x1 - x3) * ((x1 - x3) ** 2 + (y1 - y3) ** 2))
                            / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 2)
                    )
                    Fxx = -(
                            2 * ((x1 - x3) ** 2 + (y1 - y3) ** 2) *
                            (G11 * (y1 - y3) ** 2 + (y1 - y2) * (G22 * (y1 - y2) + 2 * G12 * (-y1 + y3)))
                            / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 3)
                    )
                    Fyy = -(
                            2 * (G11 * (x1 - x3) ** 2 + (x1 - x2) * (G22 * (x1 - x2) + 2 * G12 * (-x1 + x3))) *
                            ((x1 - x3) ** 2 + (y1 - y3) ** 2) /
                            (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 3)
                    )
                    Fxy = (
                            2 * ((x1 - x3) ** 2 + (y1 - y3) ** 2) *
                            (G22 * (x1 - x2) * (y1 - y2) +
                             G11 * (x1 - x3) * (y1 - y3) +
                             G12 * (x2 * y1 + x3 * y1 - x3 * y2 - x2 * y3 + x1 * (-2 * y1 + y2 + y3)))
                            / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 3)
                    )
                    Rx[i, j] += Fx / weight
                    Rxy[i, j] += Fxy / weight
                    Ry[i, j] += Fy / weight
                    Rxx[i, j] += Fxx / weight
                    Ryy[i, j] += Fyy / weight

                    # xk-1 : x[i + 1, j + 1]
                    Fx = (
                                 ((x1 - x2) ** 2 + (y1 - y2) ** 2) * (G22 * (-y1 + y2) + 2 * G12 * (y1 - y3)) -
                                 G11 * (-2 * x1 * (x2 * y1 + x3 * y2) +
                                        (y1 - y2) * (-x3 ** 2 + (y1 - y3) ** 2) +
                                        x1 ** 2 * (y1 + y2 - 2 * y3) +
                                        2 * x2 * x3 * (y1 - y3) +
                                        2 * x1 * (x2 + x3) * y3)
                         ) / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 2)
                    Fy = (
                                 (G22 * (x1 - x2) + 2 * G12 * (-x1 + x3)) * ((x1 - x2) ** 2 + (y1 - y2) ** 2) +
                                 G11 * (x1 ** 3 - x1 ** 2 * (x2 + 2 * x3) + x2 * (-x3 ** 2 + (y1 - y3) ** 2) -
                                        2 * x3 * (y1 - y2) * (y1 - y3) +
                                        x1 * (2 * x2 * x3 + x3 ** 2 + (y1 - y3) * (y1 - 2 * y2 + y3)))
                         ) / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 2)
                    Fxx = -(
                            2 * ((x1 - x2) ** 2 + (y1 - y2) ** 2) *
                            (G11 * (y1 - y3) ** 2 + (y1 - y2) * (G22 * (y1 - y2) + 2 * G12 * (-y1 + y3)))
                            / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 3)
                    )
                    Fyy = -(
                            2 * (G11 * (x1 - x3) ** 2 + (x1 - x2) * (G22 * (x1 - x2) + 2 * G12 * (-x1 + x3))) *
                            ((x1 - x2) ** 2 + (y1 - y2) ** 2) /
                            (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 3)
                    )
                    Fxy = (
                            2 * ((x1 - x2) ** 2 + (y1 - y2) ** 2) *
                            (G22 * (x1 - x2) * (y1 - y2) +
                             G11 * (x1 - x3) * (y1 - y3) +
                             G12 * (x2 * y1 + x3 * y1 - x3 * y2 - x2 * y3 + x1 * (-2 * y1 + y2 + y3)))
                            / (Dk * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 3)
                    )
                    Rx[i + 1, j + 1] += Fx / weight
                    Rxy[i + 1, j + 1] += Fxy / weight
                    Ry[i + 1, j + 1] += Fy / weight
                    Rxx[i + 1, j + 1] += Fxx / weight
                    Ryy[i + 1, j + 1] += Fyy / weight

    return aproximation/((nx-1)*(ny-1)*8), Rx, Ry, Rxx, Ryy, Rxy, flag


def mimimize(omega, canon, grid_shape, tau, eps, stop_flag=[False]):

    maxdif = 99999999
    x, y = omega
    value = 0
    iter_count = 0
    while maxdif > eps:
        if stop_flag[0]:
            print("Алгоритм прерван пользователем")
            break

        Fk, Rx, Ry, Rxx, Ryy, Rxy, _ = functional2((x, y), canon, grid_shape)
        if iter_count == 0:
            print("start value = ", Fk)
        value = Fk
        iter_count += 1
        # Вычисляем только внутренние точки
        denom = Rxx[1:-1, 1:-1] * Ryy[1:-1, 1:-1] - Rxy[1:-1, 1:-1] ** 2
        denom[denom == 0] = 1e-12

        dx = -tau * (Rx[1:-1, 1:-1] * Ryy[1:-1, 1:-1] - Ry[1:-1, 1:-1] * Rxy[1:-1, 1:-1]) / denom
        dy = -tau * (Ry[1:-1, 1:-1] * Rxx[1:-1, 1:-1] - Rx[1:-1, 1:-1] * Rxy[1:-1, 1:-1]) / denom

        # Создаём новые массивы, копируя старые
        x_new = x.copy()
        y_new = y.copy()

        # Обновляем только внутренние элементы
        x_new[1:-1, 1:-1] += dx
        y_new[1:-1, 1:-1] += dy

        maxdif = max(np.max(np.abs(dx)), np.max(np.abs(dy)))
        print("maxdif is now at: ", maxdif)

        if stop_flag[0]:
            print("Алгоритм прерван пользователем")
            break

        new_grid = (x_new, y_new)
        Fk_new, *_, flag_new = functional2(new_grid, canon, grid_shape)

        if flag_new == 1 or Fk_new > Fk:
            tau *= 0.5
            print("tau went down")
            if Fk_new > Fk:
                print("minus value")
            continue
        x, y = x_new, y_new

    print("func value = ", value)
    print("count of iterations = ", iter_count)

    return np.array([x, y]), tau


def plot_grid(grid, nx, ny, title):
    fig, ax = plt.subplots()
    for i in range(nx):
        for j in range(ny):
            if i < nx - 1:
                ax.plot([grid[i * ny + j, 0], grid[(i + 1) * ny + j, 0]], [grid[i * ny + j, 1], grid[(i + 1) * ny + j, 1]], 'b-', lw=1)
            if j < ny - 1:
                ax.plot([grid[i * ny + j, 0], grid[i * ny + j + 1, 0]], [grid[i * ny + j, 1], grid[i * ny + j + 1, 1]], 'b-', lw=1)
    ax.set_aspect('auto', 'box')
    plt.title(title)
    plt.show()


def approximate_quad_to_rectangle(x1, y1, x2, y2, x3, y3, x4, y4):
    # Точки четырёхугольника (в порядке обхода)
    quad_points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

    # 1. Вычисляем площадь четырёхугольника (формула шнурования)
    def compute_polygon_area(points):
        x = points[:, 0]
        y = points[:, 1]
        return 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

    area = compute_polygon_area(quad_points)

    # 2. Находим центроид (среднее координат)
    centroid = np.mean(quad_points, axis=0)

    # 3. Вычисляем габариты (ширину и высоту) четырёхугольника
    min_x, max_x = np.min(quad_points[:, 0]), np.max(quad_points[:, 0])
    min_y, max_y = np.min(quad_points[:, 1]), np.max(quad_points[:, 1])
    width = max_x - min_x
    height = max_y - min_y

    # 4. Корректируем стороны прямоугольника, сохраняя площадь и пропорции
    aspect_ratio = width / height if height != 0 else 1.0  # избегаем деления на 0
    new_height = np.sqrt(area / aspect_ratio)
    new_width = area / new_height

    # 5. Строим прямоугольник с центром в центроиде
    rect_half_width = new_width / 2
    rect_half_height = new_height / 2

    # Координаты вершин прямоугольника
    rect_x1 = centroid[0] - rect_half_width
    rect_y1 = centroid[1] - rect_half_height

    rect_x2 = centroid[0] + rect_half_width
    rect_y2 = centroid[1] - rect_half_height

    rect_x3 = centroid[0] + rect_half_width
    rect_y3 = centroid[1] + rect_half_height

    rect_x4 = centroid[0] - rect_half_width
    rect_y4 = centroid[1] + rect_half_height

    # Возвращаем 8 переменных + площадь + центроид
    return (
        rect_x1, rect_y1, rect_x2, rect_y2, rect_x3, rect_y3, rect_x4, rect_y4
    )


def approximate_quad_by_pca(x1, y1, x2, y2, x3, y3, x4, y4):
    # Точки четырёхугольника
    points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

    # 1. Применяем PCA для определения главных осей
    pca = PCA(n_components=2)
    pca.fit(points)

    # Главные компоненты (направления)
    pc1 = pca.components_[0]  # первая главная компонента (наибольшая дисперсия)
    pc2 = pca.components_[1]  # вторая главная компонента

    # Центроид (среднее точек)
    centroid = np.mean(points, axis=0)

    # 2. Проецируем точки на главные оси, чтобы определить "размеры"
    projected = pca.transform(points)
    min_pc1, max_pc1 = np.min(projected[:, 0]), np.max(projected[:, 0])
    min_pc2, max_pc2 = np.min(projected[:, 1]), np.max(projected[:, 1])

    # Размеры вдоль главных осей
    length = max_pc1 - min_pc1  # "длина" (по первой компоненте)
    width = max_pc2 - min_pc2  # "ширина" (по второй компоненте)

    # 3. Угол поворота прямоугольника (наклон первой компоненты)
    angle = np.arctan2(pc1[1], pc1[0])

    # 4. Строим прямоугольник в новой системе координат
    # Углы прямоугольника (до поворота)
    half_length = length / 2
    half_width = width / 2
    rect_local = np.array([
        [-half_length, -half_width],
        [half_length, -half_width],
        [half_length, half_width],
        [-half_length, half_width]
    ])

    # Поворачиваем прямоугольник обратно в исходную систему
    rotation_matrix = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ])
    rect_rotated = np.dot(rect_local, rotation_matrix)

    # Сдвигаем к центроиду
    rect_final = rect_rotated + centroid

    # Разбиваем координаты на отдельные переменные
    rect_x1, rect_y1 = rect_final[0]
    rect_x2, rect_y2 = rect_final[1]
    rect_x3, rect_y3 = rect_final[2]
    rect_x4, rect_y4 = rect_final[3]

    # Площадь исходного четырёхугольника (для сравнения)
    def compute_area(points):
        x = points[:, 0]
        y = points[:, 1]
        return 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

    original_area = compute_area(points)
    new_area = length * width  # площадь прямоугольника

    return (
        rect_x1, rect_y1, rect_x2, rect_y2, rect_x3, rect_y3, rect_x4, rect_y4, original_area, new_area
    )


# ТЕСТЫ #
treshhold = 1e-8

"""test, _ = winslow_without_implicit(31, Xyt, Xyb, Xyr, Xyl, treshhold, 999999)   # Уинслоу + Томпсон + явная TFI
plt.plot(test[0], test[1], c="b", linewidth=1)
plt.plot(np.transpose(test[0]), np.transpose(test[1]), c="b", linewidth=1)
plt.show()"""
"""
test = implicit_transfinite_interpol(41, myt, myb, myr, myl)        # неявное задание сетки без сдвига узла
plt.plot(test[0], test[1], c="b", linewidth=1)
plt.plot(np.transpose(test[0]), np.transpose(test[1]), c="b", linewidth=1)
plt.show()

test = implicit_transfinite_interpol(41, chevt, chevb, chevr, chevl)        # неявное задание сетки без сдвига узла
plt.plot(test[0], test[1], c="b", linewidth=1)
plt.plot(np.transpose(test[0]), np.transpose(test[1]), c="b", linewidth=1)
plt.show()

node = [21, 21, 0.3, 0.2]    # некий известный узел , [положение в сетке по индексам, коорд в прве]

test = implicit_transfinite_interpol(41, chevt, chevb, chevr, chevl, node)  # неявное задание сетки со сдвигом узла
plt.plot(test[0], test[1], c="b", linewidth=1)
plt.plot(np.transpose(test[0]), np.transpose(test[1]), c="b", linewidth=1)
plt.show()"""

"""
test = transfinite_interpol(51, Xyt, Xyb, Xyl, Xyr)       # явное задание сетки / начальное приближение для Уинслоу
plt.plot(test[0], test[1], c="b", linewidth=1)
plt.plot(np.transpose(test[0]), np.transpose(test[1]), c="b", linewidth=1)
plt.show()

test = transfinite_interpol(51, x_top_chevron, x_bottom_chevron, x_left_chevron, x_right_chevron)       # явное задание сетки / начальное приближение для Уинслоу
plt.plot(test[0], test[1], c="b", linewidth=1)
plt.plot(np.transpose(test[0]), np.transpose(test[1]), c="b", linewidth=1)
plt.show()

test = transfinite_interpol(51, Myt, Myb, Myl, Myr)       # явное задание сетки / начальное приближение для Уинслоу
plt.plot(test[0], test[1], c="b", linewidth=1)
plt.plot(np.transpose(test[0]), np.transpose(test[1]), c="b", linewidth=1)
plt.show()
"""

xup = np.array([
    -1.987072766474143, -1.890210852412545, -1.7958789826267216, -1.7091372456682226,
    -1.6249255529854985, -1.5407138603027744, -1.4565021676200502, -1.3722904749373261,
    -1.288078782254602, -1.2038670895718777, -1.1326103316297087, -1.0841239721695128,
    -0.9973822352110138, -0.897990276873641, -0.7985983185362677, -0.7143866258535438,
    -0.6301749331708195, -0.5459632404880956, -0.4617515478053713, -0.3775398551226474,
    -0.2933281624399231, -0.20911646975719878, -0.1249047770744749, -0.04069308439175057,
    0.043518608290973315, 0.12773030097369764, 0.21194199365642152, 0.29615368633914585,
    0.38036537902186973, 0.46457707170459406, 0.5487887643873184, 0.6330004570700423,
    0.7172121497527666, 0.8014238424354905, 0.8856355351182148, 0.9698472278009391,
    1.0540589204836626, 1.138270613166387, 1.2299708675558763, 1.2987991523428302,
    1.3396584440227701, 1.3826691967109457, 1.4132698052142953, 1.464590922311943,
    1.5488026149946674, 1.6330143076773918, 1.7172260003601152, 1.8014376930428395,
    1.8856493857255638, 1.9825112997871632, 2.0692530367456627, 2.163584906531487,
    2.270566997696187
])

yup = np.array([
    -0.3956929864149973, -0.39573392551646036, -0.39577379528391, -0.3958104570493193,
    -0.3958460494807152, -0.39588164191211117, -0.395917234343507, -0.39595282677490296,
    -0.3959884192062988, -0.39602401163769474, -0.3980860016311343, -0.45618639357858526,
    -0.4784046917872816, -0.4575388110877494, -0.419275462611749, -0.39128514355990596,
    -0.3674712348467418, -0.34904269735976867, -0.3367688698455851, -0.32878135820530874,
    -0.3228820517343717, -0.32082943899642824, -0.323942386414219, -0.33046240542408967,
    -0.339949873885127, -0.35251469733255913, -0.36793706469592924, -0.38676650365137943,
    -0.4083435809875392, -0.432998013310094, -0.46072980061904356, -0.49142903737915955,
    -0.5248759125199853, -0.5618397647881195, -0.603199838465389, -0.6486264169461092,
    -0.6961412005961682, -0.7451946617394252, -0.8035103472564249, -0.8546404984710649,
    -0.8791593751453624, -0.8943931075006826, -0.87749485699123, -0.8754607361214952,
    -0.8754963285528912, -0.8755319209842869, -0.8755675134156828, -0.8756031058470788,
    -0.8756386982784747, -0.8756796373799378, -0.8757162991453471, -0.8757561689127966,
    -0.8758013853503135
])

xlow = np.array([
    -1.987072766474143, -1.8978009852398694, -1.8009390711782711, -1.7116672899439973,
    -1.6274555972612732, -1.543243904578549, -1.459032211895825, -1.3722904749373261,
    -1.288078782254602, -1.2038670895718777, -1.1385542208406791, -1.1114558373384673,
    -1.0564966273771104, -0.9959694732614026, -0.9196526267676837, -0.8354409340849598,
    -0.7512292414022355, -0.6670175487195116, -0.5828058560367873, -0.49859416335406337,
    -0.41438247067133904, -0.3301707779886147, -0.24595908530589083, -0.1617473926231665,
    -0.07753569994044263, 0.0066759927422817, 0.11943840404067751, 0.2314805837569276,
    0.335932630645853, 0.42773445615590244, 0.5119461488386263, 0.5961578415213507,
    0.6803695342040745, 0.7645812268867989, 0.8487929195695227, 0.9532449664584477,
    1.0551669690715975, 1.146765652340521, 1.2228793565931197, 1.313970184259246,
    1.3593170727203039, 1.3918105975613746, 1.4190408916097659, 1.5133819951338157,
    1.610243909195415, 1.7071058232570153, 1.7963776044912891, 1.8856493857255638,
    1.9698610784082882, 2.0540727710910125, 2.1306943309464113, 2.204785846526035,
    2.270566997696187
])

ylow = np.array([
    -1.5177132488915386, -1.5176464211258, -1.5176873602272631, -1.517725091326686,
    -1.517760683758082, -1.5177962761894777, -1.5178318686208736, -1.5178685303862829,
    -1.5179041228176788, -1.5179397152490748, -1.5163516260082557, -1.4383716874759258,
    -1.3685304740934108, -1.3113375436591896, -1.2575837212137575, -1.2149759659765391,
    -1.1832488587269312, -1.1596547610842236, -1.143094617696133, -1.1343377673092578,
    -1.1322851545713144, -1.1342990466368215, -1.1411487822523771, -1.1521749282066118,
    -1.1664982402176982, -1.1834592850742658, -1.2124055445850601, -1.2426931184836019,
    -1.2775256453712738, -1.3101385182791194, -1.3440250155608586, -1.3808789622937643,
    -1.4235579023937746, -1.4705231583676917, -1.519027091834807, -1.5785204926604453,
    -1.6475649648805175, -1.7154533520411932, -1.781379346270385, -1.8688197113819887,
    -1.916352284861543, -1.9554597152587587, -1.9965180494156534, -1.9973970614880834,
    -1.9974380005895465, -1.9974789396910095, -1.9975166707904322, -1.9975544018898548,
    -1.9975899943212507, -1.9976255867526467, -1.9976579711820026, -1.9976892862773448,
    -1.997717861692112
])


def lr_points(x_upper, y_upper, x_lower, y_lower, n_points=55):
    # Левые и правые границы (линии от нижних точек к верхним)
    y_left = np.linspace(y_lower[0], y_upper[0], n_points)
    y_right = np.linspace(y_lower[-1], y_upper[-1], n_points)
    x_left = np.linspace(x_lower[0], x_upper[0], n_points)
    x_right = np.linspace(x_lower[-1], x_upper[-1], n_points)
    return x_upper, y_upper, x_lower, y_lower, x_right, y_right, x_left, y_left


elem_sdvig = 10


def implicit_transfinite_better(discr, nod=None):
    # nod = [x, y, xn, yn] - [положение в сетке по индексам, коорд в прве], по умолчанию не используется

    m = discr
    n = discr

    t_x_val, t_y_val, b_x_val, b_y_val, r_x_val, r_y_val, l_x_val, l_y_val, = lr_points(xup, yup, xlow, ylow)

    X = np.zeros((m*n, m*n))  # матрица (позиция иксов исходя из дискретизации; иксы на этой позиции)
    Y = np.zeros((m*m, m*n))
    bx = np.zeros(m*n)
    by = np.zeros(m*n)
    for i in range(discr):
        for j in range(discr):

            gidx = i*discr + j

            if j == 0:
                X[gidx, i*discr + j] = 1.0
                bx[gidx] = b_x_val[i]
            elif j == discr - 1:
                X[gidx, i*discr + j] = 1.0
                bx[gidx] = t_x_val[i]
            elif i == 0:
                X[gidx, i*discr + j] = 1.0
                bx[gidx] = l_x_val[j]
            elif i == discr - 1:
                X[gidx, i*discr + j] = 1.0
                bx[gidx] = r_x_val[j]
            elif nod != None and i == nod[0]-1 and j == nod[1]-1:
                X[gidx, i * discr + j] = 1.0
                bx[gidx] = nod[2]
            else:
                X[gidx, i * discr + j] = 1.0
                X[gidx, (i-1) * discr + j] = -1/2
                X[gidx, (i+1) * discr + j] = -1/2
                X[gidx, i * discr + (j-1)] = -1/2
                X[gidx, i * discr + (j+1)] = -1/2
                X[gidx, (i+1) * discr + (j+1)] = 1/4
                X[gidx, (i+1) * discr + (j-1)] = 1/4
                X[gidx, (i-1) * discr + (j+1)] = 1/4
                X[gidx, (i-1) * discr + (j-1)] = 1/4
                bx[gidx] = 0.0
    for i in range(discr):
        for j in range(discr):
            gidx = i*discr + j
            if j == 0:
                Y[gidx, i*discr + j] = 1.0
                by[gidx] = b_y_val[i]
            elif j == discr - 1:
                Y[gidx, i*discr + j] = 1.0
                by[gidx] = t_y_val[i]
            elif i == 0:
                Y[gidx, i*discr + j] = 1.0
                by[gidx] = l_y_val[j]
            elif i == discr - 1:
                Y[gidx, i*discr + j] = 1.0
                by[gidx] = r_y_val[j]
            elif nod != None and i == nod[1]-1 and j == nod[0]-1:
                Y[gidx, i * discr + j] = 1.0
                by[gidx] = nod[3]
            else:
                Y[gidx, i * discr + j] = 1.0
                Y[gidx, (i-1) * discr + j] = -1/2
                Y[gidx, (i+1) * discr + j] = -1/2
                Y[gidx, i * discr + (j-1)] = -1/2
                Y[gidx, i * discr + (j+1)] = -1/2
                Y[gidx, (i+1) * discr + (j+1)] = 1/4
                Y[gidx, (i+1) * discr + (j-1)] = 1/4
                Y[gidx, (i-1) * discr + (j+1)] = 1/4
                Y[gidx, (i-1) * discr + (j-1)] = 1/4
                by[gidx] = 0.0
    X_res = (np.linalg.solve(X, bx)).reshape((n, m))
    Y_res = (np.linalg.solve(Y, by)).reshape((n, m))
    """for point in range(elem_sdvig-1, -1, -1):
        for i in range(n):
            Y_res[point, i] = Y_res[point+1, i]
            Y_res[-(point+1), i] = Y_res[-(point+2), i]"""
    ind = 1
    for point in range(1, elem_sdvig):  # levo
        for i in range(n):
            x0 = X_res[0, i]
            xN = X_res[elem_sdvig, i]
            X_res[point, i] = x0 + (xN - x0)*(ind/elem_sdvig)**1
        ind += 1
    ind = 1
    for point in range(n-elem_sdvig, n-1):  # pravo
        for i in range(n):
            x0 = X_res[n-(elem_sdvig+1), i]
            xN = X_res[n-1, i]
            X_res[point, i] = xN - (xN - x0)*(1-(ind/elem_sdvig))**1
        ind += 1
    return [X_res, Y_res]


node2 = [11, 11, 0.6, 0.5]

nx = ny = 53    # ------------------------------------------------------------------------------------------------------
# canon_grid = implicit_transfinite_interpol(nx, sq_t1, sq_b1, sq_l1, sq_r1)
# canon_grid, count = winslow_without_implicit(NX, Xyt, Xyb, Xyr, Xyl, treshhold)
# canon_grid = transfinite_interpol(nx, Xyt, Xyb, Xyl, Xyr)
# canon_grid = implicit_transfinite_interpol(nx, chevt, chevb, chevr, chevl)
# canon_grid = implicit_transfinite_interpol(nx, myt, myb, myr, myl)
canon_grid = implicit_transfinite_interpol(nx, sq_t2, sq_b2, sq_l2, sq_r2)

# start
coef_temp = 1
dc = 0.06

for i in range(nx//2 + coef_temp, nx-1):    # низ
    for j in range(ny):
        if i != 0 and i != ny-1 and elem_sdvig-1 < j < ny - elem_sdvig:
            dense_coef = 1
            razn = abs(canon_grid[0][i][j] - canon_grid[0][i][j]**(dense_coef+dc*coef_temp))
            canon_grid[0][i][j] = canon_grid[0][i][j]**(dense_coef+dc*coef_temp)
            canon_grid[0][nx//2 - coef_temp][j] += razn*1.07
    coef_temp += 1

"""ind = 1
for point in range(1, nx//3):   # верх
    for i in range(nx):
        if elem_sdvig < i < ny - (elem_sdvig+1):
            x0 = canon_grid[0][0][i]
            xN = canon_grid[0][nx//3+1][i]
            canon_grid[0][point][i] = xN - (xN - x0)*(1-(ind/(nx//3)))**0.35
    ind += 1
ind = 1
for point in range(nx-(nx//3)+1, nx-1):  # низ
    for i in range(nx):
        if elem_sdvig < i < ny - (elem_sdvig+1):
            x0 = canon_grid[0][nx-(nx//3)][i]
            xN = canon_grid[0][nx-1][i]
            canon_grid[0][point][i] = x0 + (xN - x0)*(ind/(nx//3-1))**0.5
    ind += 1"""

"""ind = 1
for point in range(1, nx//2):   # верх
    for i in range(nx):
        if elem_sdvig < i < ny - (elem_sdvig+1):
            x0 = canon_grid[0][0][i]
            xN = canon_grid[0][nx//2+1][i]
            canon_grid[0][point][i] = xN - (xN - x0)*(1-(ind/(nx//2)))**0.65
    ind += 1
ind = 1
for point in range(nx//2+1, nx-1):  # низ
    for i in range(nx):
        if elem_sdvig < i < ny - (elem_sdvig+1):
            x0 = canon_grid[0][nx//2][i]
            xN = canon_grid[0][nx-1][i]
            canon_grid[0][point][i] = x0 + (xN - x0)*(ind/(nx//2))**0.7
    ind += 1"""

ind = 1
for point in range(1, elem_sdvig):  # pravo
    for i in range(nx):
        x0 = canon_grid[1][i][0]
        xN = canon_grid[1][i][elem_sdvig]
        canon_grid[1][i][point] = x0 + (xN - x0)*(ind/elem_sdvig)**1
    ind += 1
ind = 1
for point in range(nx-elem_sdvig, nx-1):    # levo
    for i in range(nx):
        x0 = canon_grid[1][i][nx-(elem_sdvig+1)]
        xN = canon_grid[1][i][nx-1]
        canon_grid[1][i][point] = xN - (xN - x0)*(1-(ind/elem_sdvig))**1
    ind += 1
# finish


canon_grid[0] = np.flip(canon_grid[0], axis=0)
canon_grid[1] = np.flip(canon_grid[1], axis=0)
canon_grid[0] = np.transpose(canon_grid[0])
canon_grid[1] = np.transpose(canon_grid[1])

canon_grid, count = winslow(nx, canon_grid[0], canon_grid[1], treshhold, 27)

xx, yy = canon_grid[0], canon_grid[1]
canon_grid = (xx, yy)

# ПЕРЕХОД ОТ СЕТКИ К ЯЧЕЙКАМ ------------------------------------------------------------------------------------------

cells_x = np.empty((nx - 1, ny - 1), dtype=object)
cells_y = np.empty((nx - 1, ny - 1), dtype=object)
for i in range(nx - 1):
    for j in range(ny - 1):
        cells_x[i, j] = np.zeros((2, 2))
        cells_y[i, j] = np.zeros((2, 2))

for i in range(nx - 1):
    for j in range(ny - 1):
        cx = cells_x[i, j]
        cy = cells_y[i, j]
        cx[0, 0] = canon_grid[0][i, j]
        cy[0, 0] = canon_grid[1][i, j]
        cx[1, 1] = canon_grid[0][i + 1, j + 1]
        cy[1, 1] = canon_grid[1][i + 1, j + 1]
        cx[1, 0] = canon_grid[0][i + 1, j]
        cy[1, 0] = canon_grid[1][i + 1, j]
        cx[0, 1] = canon_grid[0][i, j + 1]
        cy[0, 1] = canon_grid[1][i, j + 1]
        # (cx[0, 0], cy[0, 0], cx[0, 1], cy[0, 1], cx[1, 1], cy[1, 1], cx[1, 0], cy[1, 0], new_ar, orig_ar) = \
        (x1, y1, x2, y2, x3, y3, x4, y4, new_ar, orig_ar) = \
            approximate_quad_by_pca(cx[0, 0], cy[0, 0], cx[0, 1], cy[0, 1], cx[1, 1], cy[1, 1], cx[1, 0], cy[1, 0])

        cx[1, 0], cy[1, 0], cx[0, 0], cy[0, 0], cx[0, 1], cy[0, 1], cx[1, 1], cy[1, 1] = \
            (x1, y1, x2, y2, x3, y3, x4, y4)
cells = np.array([cells_x, cells_y])


# omega_grid2, count = winslow_without_implicit(nx, Xyt, Xyb, Xyr, Xyl, treshhold)
# omega_grid2 = implicit_transfinite_interpol(nx, myt, myb, myr, myl)
# omega_grid2 = transfinite_interpol(nx, Xyt, Xyb, Xyl, Xyr)
# omega_grid2 = implicit_transfinite_new(nx)
omega_grid2 = implicit_transfinite_better(nx)  # vot eto na diplom
# omega_grid2 = implicit_transfinite_interpol(nx, chevt, chevb, chevr, chevl)
# omega_grid2 = transfinite_interpol(20, horst, horsb, horsr, horsl)
# omega_grid2 = winslow_without_implicit(nx, Xyt, Xyb, Xyr, Xyl, treshhold)[0]
# omega_grid2 = implicit_transfinite_interpol(nx, sq_t1, sq_b1, sq_l1, sq_r1)
# print("\n")
# print(omega_grid2)
omega_grid2[0] = np.flip(omega_grid2[0], axis=0)    # Если имплисит, то это в комменте должно быть
omega_grid2[1] = np.flip(omega_grid2[1], axis=0)
"""omega_grid2[0] = np.transpose(omega_grid2[0])
omega_grid2[1] = np.transpose(omega_grid2[1])"""
"""for i in range(nx-1, -1, -1):
    for j in range(ny):
        print(round(omega_grid2[0][i, j], 3), end="\t")
    print("\n")
print("\n"*5)
for i in range(nx-1, -1, -1):
    for j in range(ny):
        print(round(omega_grid2[1][i, j], 3), end="\t")
    print("\n")"""
"""print("check2: ", omega_grid2[0][0, 0], omega_grid2[0][nx-1, ny-1])"""

# omega_grid2, _ = winslow(nx, omega_grid2[0], omega_grid2[1], treshhold, 25)

plt.plot(omega_grid2[0], omega_grid2[1], c="b", linewidth=1)
plt.plot(np.transpose(omega_grid2[0]), np.transpose(omega_grid2[1]), c="b", linewidth=1)
plt.show()

plt.plot(-np.transpose(canon_grid[1]), np.transpose(canon_grid[0]), c="b", linewidth=1)
plt.plot(-canon_grid[1], canon_grid[0], c="b", linewidth=1)
plt.show()

### МЕЙН ВЫЗОВ ###
"""new_grid = mimimize(omega_grid, canon_grid, (nx, ny), 1, 10**(-8))"""
"""new_grid, value = hooke_jeeves(omega_grid2, canon_grid, (nx, ny), 0.1, 0.0001)"""
global_iter_flag = 0
while_iter = 0
new_tau = 0
stop_flag = [False]


def stop_optimization():
    input("Нажмите Enter для остановки...\n")
    stop_flag[0] = True


# Запуск "слушателя кнопки" в отдельном потоке
threading.Thread(target=stop_optimization).start()

omega_grid2, flag = mimimize(omega_grid2, cells, (nx, ny), 5, 1e-6, stop_flag=stop_flag)

# omega_grid2, count = winslow(nx, omega_grid2[0], omega_grid2[1], treshhold, 8)

new_grid = omega_grid2
plt.plot(new_grid[0], new_grid[1], c="b", linewidth=1)
plt.plot(np.transpose(new_grid[0]), np.transpose(new_grid[1]), c="b", linewidth=1)
plt.show()



### ТЕСТ АПРОКСИМАЦИИ ###
"""functional(omega_grid2, canon_grid, (nx, ny))
functional2(omega_grid2, canon_grid, (nx, ny))"""

"""test = implicit_transfinite_interpol(41, chevt, chevb, chevr, chevl)        # неявное задание сетки без сдвига узла
plt.plot(test[0], test[1], c="b", linewidth=1)
plt.plot(np.transpose(test[0]), np.transpose(test[1]), c="b", linewidth=1)
#plt.show()

node = [21, 21, 0.3, 0.2]    # некий известный узел , [положение в сетке по индексам, коорд в прве]

test = implicit_transfinite_interpol(41, chevt, chevb, chevr, chevl, node)  # неявное задание сетки со сдвигом узла
plt.plot(test[0], test[1], c="b", linewidth=1)
plt.plot(np.transpose(test[0]), np.transpose(test[1]), c="b", linewidth=1)
#plt.show()

test = transfinite_interpol(51, Xyt, Xyb, Xyl, Xyr)       # явное задание сетки / начальное приближение для Уинслоу
plt.plot(test[0], test[1], c="b", linewidth=1)
plt.plot(np.transpose(test[0]), np.transpose(test[1]), c="b", linewidth=1)
#plt.show()

test = transfinite_interpol(31, Xyt, Xyb, Xyl, Xyr)       # явное задание сетки / начальное приближение для Уинслоу
plt.plot(test[0], test[1], c="b", linewidth=1)
plt.plot(np.transpose(test[0]), np.transpose(test[1]), c="b", linewidth=1)
plt.show()

test, _ = winslow_without_implicit(31, Xyt, Xyb, Xyr, Xyl, treshhold, 1500)   # Уинслоу + Томпсон + явная TFI
print("кол-во итераций: ", count)
plt.plot(test[0], test[1], c="b", linewidth=1)
plt.plot(np.transpose(test[0]), np.transpose(test[1]), c="b", linewidth=1)
plt.show()
"""
