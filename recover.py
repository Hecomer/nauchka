import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize
import sys
import threading
import time
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


def functional2(grid_omega, grid_canon, grid_shape):
    nx, ny = grid_shape
    aproximation = 0
    x = grid_omega[0]
    y = grid_omega[1]
    X = grid_canon[0]
    Y = grid_canon[1]
    X = np.reshape(X, (nx, ny))
    Y = np.reshape(Y, (nx, ny))
    x = np.reshape(x, (nx, ny))
    """print("x00: ", x[0, 0])
    print("x1010: ", x[nx-1, ny-1])
    print("x010: ", x[0, ny-1])
    print("x100: ", x[nx-1, 0])"""
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
            for k in range(0, 4):
                if k == 0:  # lb corner perf; k = i,j; k+1 = i,j+1; k-1 = i+1,j
                    G11 = ((X[i, j + 1] - X[i, j]) ** 2) + ((Y[i, j + 1] - Y[i, j]) ** 2)
                    G12 = (X[i, j + 1] - X[i, j]) * (X[i + 1, j] - X[i, j]) + (Y[i, j + 1] - Y[i, j]) * (
                                Y[i + 1, j] - Y[i, j])
                    G22 = (X[i + 1, j] - X[i, j]) ** 2 + (Y[i + 1, j] - Y[i, j]) ** 2
                    Jk = (x[i, j + 1] - x[i, j]) * (y[i + 1, j] - y[i, j]) - (x[i + 1, j] - x[i, j]) * (
                                y[i, j + 1] - y[i, j])
                    if Jk <= 0:
                        print("1: flag")
                        flag = 1
                    Dk = (X[i, j + 1] - X[i, j]) * (Y[i + 1, j] - Y[i, j]) - (X[i + 1, j] - X[i, j]) * (
                                Y[i, j + 1] - Y[i, j])
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
                    G11 = ((X[i + 1, j] - X[i + 1, j + 1]) ** 2) + (Y[i + 1, j] - Y[i + 1, j + 1]) ** 2
                    G12 = (X[i + 1, j] - X[i + 1, j + 1]) * (X[i, j + 1] - X[i + 1, j + 1]) + (
                                Y[i + 1, j] - Y[i + 1, j + 1]) * (
                                  Y[i, j + 1] - Y[i + 1, j + 1])
                    G22 = ((X[i, j + 1] - X[i + 1, j + 1]) ** 2) + (Y[i, j + 1] - Y[i + 1, j + 1]) ** 2
                    Jk = (x[i + 1, j] - x[i + 1, j + 1]) * (y[i, j + 1] - y[i + 1, j + 1]) - (
                                x[i, j + 1] - x[i + 1, j + 1]) * (
                                 y[i + 1, j] - y[i + 1, j + 1])
                    if Jk <= 0:
                        print("2: flag")
                        flag = 1
                    Dk = (X[i + 1, j] - X[i + 1, j + 1]) * (Y[i, j + 1] - Y[i + 1, j + 1]) - (
                                X[i, j + 1] - X[i + 1, j + 1]) * (
                                 Y[i + 1, j] - Y[i + 1, j + 1])
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
                    G11 = ((X[i + 1, j + 1] - X[i, j + 1]) ** 2) + (Y[i + 1, j + 1] - Y[i, j + 1]) ** 2
                    G12 = (X[i + 1, j + 1] - X[i, j + 1]) * (X[i, j] - X[i, j + 1]) + (
                            Y[i + 1, j + 1] - Y[i, j + 1]) * (
                                  Y[i, j] - Y[i, j + 1])
                    G22 = (X[i, j] - X[i, j + 1]) ** 2 + (Y[i, j] - Y[i, j + 1]) ** 2
                    Jk = (x[i + 1, j + 1] - x[i, j + 1]) * (y[i, j] - y[i, j + 1]) - (x[i, j] - x[i, j + 1]) * (
                            y[i + 1, j + 1] - y[i, j + 1])
                    if Jk <= 0:
                        print("3: flag")
                        flag = 1
                    Dk = (X[i + 1, j + 1] - X[i, j + 1]) * (Y[i, j] - Y[i, j + 1]) - (X[i, j] - X[i, j + 1]) * (
                            Y[i + 1, j + 1] - Y[i, j + 1])
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
                    G11 = (X[i, j] - X[i + 1, j]) ** 2 + (Y[i, j] - Y[i + 1, j]) ** 2
                    G12 = (X[i, j] - X[i + 1, j]) * (X[i + 1, j + 1] - X[i + 1, j]) + (
                            Y[i, j] - Y[i + 1, j]) * (
                                  Y[i + 1, j + 1] - Y[i + 1, j])
                    G22 = (X[i + 1, j + 1] - X[i + 1, j]) ** 2 + (Y[i + 1, j + 1] - Y[i + 1, j]) ** 2
                    Jk = (x[i, j] - x[i + 1, j]) * (y[i + 1, j + 1] - y[i + 1, j]) - (x[i + 1, j + 1] - x[i + 1, j]) * (
                            y[i, j] - y[i + 1, j])
                    if Jk <= 0:
                        print("4: flag")
                        flag = 1
                    Dk = (X[i, j] - X[i + 1, j]) * (Y[i + 1, j + 1] - Y[i + 1, j]) - (X[i + 1, j + 1] - X[i + 1, j]) * (
                            Y[i, j] - Y[i + 1, j])
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

    if flag != 1:
        print("aproxim: ", aproximation/((nx-1)*(ny-1)*8))
    return aproximation/((nx-1)*(ny-1)*8), Rx, Ry, Rxx, Ryy, Rxy, flag


def mimimize(omega, canon, grid_shape, tau, eps, count=0, stop_flag=[False]):
    iter_flag = 0
    nx, ny = grid_shape
    Fk, Rx, Ry, Rxx, Ryy, Rxy, flag = functional2(omega, canon, grid_shape)
    x_old = omega[0]
    y_old = omega[1]
    maxdif = 0.0

    if stop_flag[0]:
        print("Алгоритм прерван пользователем")
        iter_flag = 1
        return np.array([x_old, y_old]), iter_flag, tau

    if count >= 996:
        print("count limit")
        print("value = ", Fk)
        return np.array([x_old, y_old]), iter_flag, tau

    x_new = x_old.copy()
    y_new = y_old.copy()

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            x_new[i, j] = x_old[i, j] - tau * ((Rx[i, j] * Ryy[i, j] - Ry[i, j] * Rxy[i, j]) / (Rxx[i, j] * Ryy[i, j] - Rxy[i, j] ** 2))
            y_new[i, j] = y_old[i, j] - tau * ((Ry[i, j] * Rxx[i, j] - Rx[i, j] * Rxy[i, j]) / (Rxx[i, j] * Ryy[i, j] - Rxy[i, j] ** 2))
            if abs(x_new[i, j] - x_old[i, j]) > maxdif:
                maxdif = abs(x_new[i, j] - x_old[i, j])
            if abs(y_new[i, j] - y_old[i, j]) > maxdif:
                maxdif = abs(y_new[i, j] - y_old[i, j])

    if stop_flag[0]:
        print("Алгоритм прерван пользователем")
        iter_flag = 1
        return np.array([x_old, y_old]), iter_flag, tau

    new_grid = np.array([x_new, y_new])
    temp_func = functional2(new_grid, canon, (nx, ny))
    flag = temp_func[6]
    value = temp_func[0]

    if flag == 1 or (value > Fk):
        tau_new = tau * 0.5
        count += 1
        if value > Fk:
            print("minus value")
        return mimimize(np.array([x_old, y_old]), canon, grid_shape, tau_new, eps, count, stop_flag)
    else:
        if maxdif <= eps:
            print("value = ", value)
            iter_flag = 1
            return np.array([x_old, y_old]), iter_flag, tau
        else:
            count += 1
            return mimimize(np.array([x_new, y_new]), canon, grid_shape, tau, eps, count, stop_flag)


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
    0.014031364832386428, 0.11417310395451596, 0.20736382132788145,
    0.30816392360930034, 0.39120879443859513, 0.4818630501759429,
    0.5649079210052377, 0.6479527918345325, 0.733534124299845,
    0.8064331485850691, 0.8337464023318986, 0.8762784258948337,
    0.9517139118160758, 1.014467089557229, 1.0848296522064351,
    1.1551922148556413, 1.2255547775048479, 1.295917340154054,
    1.3662799028032606, 1.4366424654524668, 1.507005028101673,
    1.5773675907508795, 1.6477301534000857, 1.7180927160492918,
    1.7884552786984984, 1.8588178413477046, 1.9291804039969107,
    1.9995429666461173, 2.0699055292953235, 2.14026809194453,
    2.210630654593736, 2.2809932172429424, 2.351355779892149,
    2.421718342541355, 2.4920809051905617, 2.562443467839768,
    2.632806030488974, 2.7031685931381806, 2.7735311557873867,
    2.843893718436593, 2.9142562810857995, 2.984618843735006,
    3.049955509052126, 3.1026098661891557, 3.1577536590222546,
    3.197678682039245, 3.275650629596505, 3.361231962061819,
    3.449349756163152, 3.5323946269924478, 3.617975959457762,
    3.7086302151951123, 3.796748009296445, 3.887402265033795,
    3.979981645187815
])

yup = np.array([
    1.5, 1.5, 1.5,
    1.5, 1.5, 1.5,
    1.5, 1.5, 1.5,
    1.5, 1.4622233688694797, 1.4334183923977117,
    1.4238543680331293, 1.4378877108122743, 1.4641552498604171,
    1.4898230734051778, 1.511772660828968, 1.5307236707358474,
    1.5451168428170217, 1.5557917787772255, 1.562508592415107,
    1.5674262595428414, 1.568385804348253, 1.568385804348253,
    1.5652672837306651, 1.5590302424954898, 1.5496746806427266,
    1.5387598584811695, 1.5256860605074363, 1.509853571218145,
    1.492341878519383, 1.47183160830371, 1.44952219157789,
    1.4249338559392175, 1.398786259991751, 1.3694002003260208,
    1.3371355062440555, 1.3019921777458554, 1.264090157932097,
    1.224269048507515, 1.1832485080761688, 1.1398291056312935,
    1.0940108411728893, 1.0580907583490216, 1.038143028289738,
    1.05, 1.05, 1.05,
    1.05, 1.05, 1.05,
    1.05, 1.05, 1.05,
    1.05
])

xlow = np.array([
    0.014031364832386428, 0.11417310395451596, 0.20736382132788145, 0.30816392360930034,
    0.39120879443859513, 0.4818630501759429, 0.5649079210052377, 0.6479527918345325,
    0.733534124299845, 0.8064331485850691, 0.8337464023318986, 0.8762784258948337,
    0.9517139118160758, 1.014467089557229, 1.0848296522064351, 1.1551922148556413,
    1.2255547775048479, 1.295917340154054, 1.3662799028032606, 1.4366424654524668,
    1.507005028101673, 1.5773675907508795, 1.6477301534000857, 1.7180927160492918,
    1.7884552786984984, 1.8588178413477046, 1.9291804039969107, 1.9995429666461173,
    2.0699055292953235, 2.14026809194453, 2.210630654593736, 2.2809932172429424,
    2.351355779892149, 2.421718342541355, 2.4920809051905617, 2.562443467839768,
    2.632806030488974, 2.7031685931381806, 2.7735311557873867, 2.843893718436593,
    2.9142562810857995, 2.984618843735006, 3.049955509052126, 3.1026098661891557,
    3.1577536590222546, 3.197678682039245, 3.275650629596505, 3.361231962061819,
    3.449349756163152, 3.5323946269924478, 3.617975959457762, 3.6883385221069704,
    3.796748009296445, 3.887402265033795, 3.979981645187815
])

ylow = np.array([
    0.457, 0.457, 0.457, 0.457,
    0.457, 0.457, 0.457, 0.457,
    0.457, 0.457, 0.5300199790389697, 0.5876556805333033,
    0.6577526731178749, 0.7005995752190535, 0.7370366057926203, 0.764399344591618,
    0.7846540167611713, 0.8019101114138132, 0.8112185377322758, 0.8151138126755304,
    0.8150509652964626, 0.8131889714072478, 0.8056739399414732, 0.7955044484160492,
    0.7856062667815127, 0.7708946492751505, 0.7547594264054382, 0.7370649432269323,
    0.7172114842362496, 0.6945993339300089, 0.6703079802142977, 0.6447129642359133,
    0.6156238864931454, 0.5842558898375216, 0.5479388023646321, 0.5100781664277159,
    0.47103381132880107, 0.42911082181365257, 0.38442914098294434, 0.34121820104988787,
    0.2900281690931181, 0.22965961410587044, 0.16858711235933033, 0.11741279224732759,
    0.06017692659482288, 0.01, 0.01, 0.01,
    0.01, 0.01, 0.01, 0.01,
    0.01, 0.01, 0.01
])


def lr_points(x_upper, y_upper, x_lower, y_lower, n_points=55):
    # Левые и правые границы (линии от нижних точек к верхним)
    y_left = np.linspace(y_lower[0], y_upper[0], n_points)
    y_right = np.linspace(y_lower[-1], y_upper[-1], n_points)
    x_left = np.linspace(x_lower[0], x_upper[0], n_points)
    x_right = np.linspace(x_lower[-1], x_upper[-1], n_points)
    return x_upper, y_upper, x_lower, y_lower, x_right, y_right, x_left, y_left


elem_sdvig = 9


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

nx = ny = 55    # -----------------------------------------------------------
# canon_grid = implicit_transfinite_interpol(nx, sq_t1, sq_b1, sq_l1, sq_r1)
# canon_grid, count = winslow_without_implicit(NX, Xyt, Xyb, Xyr, Xyl, treshhold)
# canon_grid = transfinite_interpol(nx, Xyt, Xyb, Xyl, Xyr)
# canon_grid = implicit_transfinite_interpol(nx, chevt, chevb, chevr, chevl)
# canon_grid = implicit_transfinite_interpol(nx, myt, myb, myr, myl)
canon_grid = implicit_transfinite_interpol(nx, sq_t1, sq_b1, sq_l1, sq_r1)

# start
coef_temp = 1
dc = 0.035

for i in range(nx//2 + coef_temp, nx-1):    # низ
    for j in range(ny):
        if i != 0 and i != ny-1 and elem_sdvig < j < ny - (elem_sdvig+1):
            dense_coef = 1
            razn = abs(canon_grid[0][i][j] - canon_grid[0][i][j]**(dense_coef+dc*coef_temp))
            canon_grid[0][i][j] = canon_grid[0][i][j]**(dense_coef+dc*coef_temp)
            canon_grid[0][nx//2 - coef_temp][j] += razn
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

canon_grid, count = winslow(nx, canon_grid[0], canon_grid[1], treshhold, 1)

xx, yy = canon_grid[0], canon_grid[1]
canon_grid = (xx, yy)

"""
for i in range(nx-1, -1, -1):
    for j in range(ny):
        print(round(canon_grid[0][i, j], 3), end="\t")
    print("\n")
print("\n"*5)
for i in range(nx-1, -1, -1):
    for j in range(ny):
        print(round(canon_grid[1][i, j], 3), end="\t")
    print("\n")"""

"""print(xx)
print("x ", xx[0, 0], xx[nx-1, ny-1])
print("y ", yy[0, 0], yy[nx-1, ny-1])"""


# omega_grid2, count = winslow_without_implicit(nx, Xyt, Xyb, Xyr, Xyl, treshhold)
# omega_grid2 = implicit_transfinite_interpol(nx, myt, myb, myr, myl)
# omega_grid2 = transfinite_interpol(nx, Xyt, Xyb, Xyl, Xyr)
# omega_grid2 = implicit_transfinite_new(nx)  # vot eto ne na diplom
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

while global_iter_flag != 1:
    if while_iter == 0:
        omega_grid2, flag, iter_tau = mimimize(omega_grid2, canon_grid, (nx, ny), 10, 1e-5, stop_flag=stop_flag)
        global_iter_flag = flag
        new_tau = iter_tau
        while_iter += 1
    else:
        omega_grid2, flag, iter_tau = mimimize(omega_grid2, canon_grid, (nx, ny), new_tau, 1e-5, stop_flag=stop_flag)
        global_iter_flag = flag
        new_tau = iter_tau
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
