import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize


def find_intersection(f1, f2):
    intersection = fsolve(lambda x: f1(x) - f2(x), 0)
    xy = [intersection[0], f1(intersection[0])]
    return xy


def Xyb(s):
    x = s
    y = 0
    return np.array([x, y])


def Xyl(s):
    x = 0
    y = s
    return np.array([x, y])


def Xyr(s):
    x = 1 + 2 * s - 2 * s**2
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
    imax, jmax = length - 2, height - 2
    imin, jmin = 0, 0
    P = np.zeros(length)
    Q = np.zeros(length)
    bArr = np.zeros(length)

    # Set P(1) to 0 since a(1) = c(1) = 0
    P[jmin] = 0.0

    # Start West-East sweep
    for i in range(imin + 1, imax + 1):
        # Set Q(1) to x(1) since x(i) = P(i)x(i+1) + Q(i) and P(1) = 0
        Q[jmin] = phi[jmin, i]

        # Start South-North traverse
        for j in range(jmin + 1, jmax + 1):
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

    # Инициализируем максимальную разницу
    phidiffmax = np.abs(phi_new[1, 1] - phi_old[1, 1])

    # Проходим по всем элементам массива, исключая границы
    for j in range(1, height - 1):
        for i in range(1, length - 1):
            phidiff = np.abs(phi_new[j, i] - phi_old[j, i])
            if phidiff > phidiffmax:
                phidiffmax = phidiff

    return phidiffmax


def winslow_with_implicit(discr, Xt, Xb, Xr, Xl, treshhold):
    x_old = implicit_transfinite_interpol(discr, Xt, Xb, Xr, Xl)[0]
    y_old = implicit_transfinite_interpol(discr, Xt, Xb, Xr, Xl)[1]
    height, length = x_old.shape
    deltaZi = 1.0
    deltaEta = 1.0
    xdiffmax = 100
    ydiffmax = 100
    x_new = x_old
    y_new = y_old
    count = 1
    while xdiffmax > treshhold or ydiffmax > treshhold or count < discr // 1.3:
        b = np.zeros((height, length))
        a = np.zeros((height, length))
        deTerm = np.zeros((height, length))
        dTerm = np.zeros((height, length))
        eTerm = np.zeros((height, length))
        x_old = x_new
        y_old = y_new
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


def winslow_without_implicit(discr, Xt, Xb, Xr, Xl, treshhold):
    x_old = transfinite_interpol(discr, Xt, Xb, Xr, Xl)[0]
    y_old = transfinite_interpol(discr, Xt, Xb, Xr, Xl)[1]
    height, length = x_old.shape
    deltaZi = 1.0
    deltaEta = 1.0
    xdiffmax = 100
    ydiffmax = 100
    x_new = x_old
    y_new = y_old
    count = 1
    while xdiffmax > treshhold or ydiffmax > treshhold or count < discr // 1.3:
        b = np.zeros((height, length))
        a = np.zeros((height, length))
        deTerm = np.zeros((height, length))
        dTerm = np.zeros((height, length))
        eTerm = np.zeros((height, length))
        x_old = x_new
        y_old = y_new
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


def compute_jacobian(grid, i, j, nx, ny):
    delta_xi = 1.0 / (nx - 1)
    delta_eta = 1.0 / (ny - 1)
    if i == 0:
        dx_dxi = (grid[0][i + 1, j] - grid[0][i, j]) / delta_xi
        dy_dxi = (grid[1][i + 1, j] - grid[1][i, j]) / delta_xi
    elif i == nx - 1:
        dx_dxi = (grid[0][i, j] - grid[0][i - 1, j]) / delta_xi
        dy_dxi = (grid[1][i, j] - grid[1][i - 1, j]) / delta_xi
    else:
        dx_dxi = (grid[0][i + 1, j] - grid[0][i - 1, j]) / (2 * delta_xi)
        dy_dxi = (grid[1][i + 1, j] - grid[1][i - 1, j]) / (2 * delta_xi)

    if j == 0:
        dx_deta = (grid[0][i, j + 1] - grid[0][i, j]) / delta_eta
        dy_deta = (grid[1][i, j + 1] - grid[1][i, j]) / delta_eta
    elif j == ny - 1:
        dx_deta = (grid[0][i, j] - grid[0][i, j - 1]) / delta_eta
        dy_deta = (grid[1][i, j,] - grid[1][i, j - 1]) / delta_eta
    else:
        dx_deta = (grid[0][i, j + 1] - grid[0][i, j - 1]) / (2 * delta_eta)
        dy_deta = (grid[1][i, j + 1] - grid[1][i, j - 1]) / (2 * delta_eta)

    J = np.array([[dx_dxi, dx_deta], [dy_dxi, dy_deta]])
    # print(J, "\n, \n")
    return J


# Функция для вычисления метрического тензора
def compute_metric_tensor(J):
    return np.dot(J.T, J)


# Функция для вычисления контрвариантного метрического тензора
def compute_contravariant_metric_tensor(g):
    return np.linalg.inv(g)


# Функция для вычисления значения функционала
def functional(grid_omega, grid_canon, grid_shape):
    nx, ny = grid_shape
    total_integral = 0.0
    delta_xi = 1 / nx
    delta_eta = 1 / ny
    for i in range(nx):
        for j in range(ny):
            J_g = compute_jacobian(grid_omega, i, j, nx, ny)
            g = compute_metric_tensor(J_g)
            J_G = compute_jacobian(grid_canon, i, j, nx, ny)    # Якобиан для управляющей метрики
            G = compute_metric_tensor(J_G)
            G_inv = compute_contravariant_metric_tensor(G)

            tr_Ginv_g = np.trace(np.dot(G_inv, g))
            det_G = np.linalg.det(G)
            det_g = np.linalg.det(g)

            integrand = ((tr_Ginv_g * (det_G ** (1 / 2))) / (det_g ** (1 / 2))) * delta_eta * delta_xi
            total_integral += integrand
    print("primerno: ", total_integral/2)
    return total_integral/2


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
    weight = 12

    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(0, 4):
                if k == 0:  # lb corner perfPERF; k = i,j; k+1 = i,j+1; k-1 = i+1,j
                    G11 = ((X[i, j + 1] - X[i, j]) ** 2) + ((Y[i, j + 1] - Y[i, j]) ** 2)
                    G12 = (X[i, j + 1] - X[i, j]) * (X[i + 1, j] - X[i, j]) + (Y[i, j + 1] - Y[i, j]) * (
                                Y[i + 1, j] - Y[i, j])
                    G22 = (X[i + 1, j] - X[i, j]) ** 2 + (Y[i + 1, j] - Y[i, j]) ** 2
                    """if i == 0 and j == 0:
                        print("check: ", x[i, j], x[i, j + 1], y[i + 1, j])"""
                    Dk = (X[i, j + 1] - X[i, j]) * (Y[i + 1, j] - Y[i, j]) - (X[i + 1, j] - X[i, j]) * (
                                Y[i, j + 1] - Y[i, j])
                    Jk = (x[i, j + 1] - x[i, j]) * (y[i + 1, j] - y[i, j]) - (x[i + 1, j] - x[i, j]) * (
                            y[i, j + 1] - y[i, j])
                    if Jk <= 0:
                        print("1: ", Jk)
                        flag = 1
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
                    if i != 1 and j != 1 and i != nx - 2 and j != nx - 2:
                        # xk : x[i, j]
                        Fx = (Dk * (-(-y[i + 1, j] + y[i, j + 1]) * (
                                    G11 * ((x[i + 1, j] - x[i, j]) ** 2 + (y[i + 1, j] - y[i, j]) ** 2) -
                                    2 * G12 * ((x[i + 1, j] - x[i, j]) * (x[i, j + 1] - x[i, j]) + (
                                        y[i + 1, j] - y[i, j]) * (
                                                       y[i, j + 1] - y[i, j])) +
                                    G22 * ((x[i, j + 1] - x[i, j]) ** 2 + (y[i, j + 1] - y[i, j]) ** 2)) +
                                    2 * (-G12 * (x[i + 1, j] + x[i, j + 1] - 2 * x[i, j]) + G11 * (x[i + 1, j] - x[i, j]) +
                                         G22 * (x[i, j + 1] - x[i, j])) * (
                                                x[i, j] * (y[i + 1, j] - y[i, j + 1]) + x[i + 1, j] * (
                                                    y[i, j + 1] - y[i, j]) +
                                                x[i, j + 1] * (-y[i + 1, j] + y[i, j])))) / \
                             (x[i, j] * (-y[i + 1, j] + y[i, j + 1]) + x[i, j + 1] * (y[i + 1, j] - y[i, j]) + x[
                                 i + 1, j] * (-y[i, j + 1] + y[i, j])) ** 2
                        Fy = (Dk * (-(x[i + 1, j] - x[i, j + 1]) * (
                                    G11 * ((x[i + 1, j] - x[i, j]) ** 2 + (y[i + 1, j] - y[i, j]) ** 2) -
                                    2 * G12 * ((x[i + 1, j] - x[i, j]) * (x[i, j + 1] - x[i, j]) + (
                                        y[i + 1, j] - y[i, j]) * (
                                                       y[i, j + 1] - y[i, j])) +
                                    G22 * ((x[i, j + 1] - x[i, j]) ** 2 + (y[i, j + 1] - y[i, j]) ** 2)) -
                                    2 * (x[i, j] * (y[i + 1, j] - y[i, j + 1]) + x[i + 1, j] * (y[i, j + 1] - y[i, j]) + x[
                                    i, j + 1] * (-y[i + 1, j] + y[i, j])) *
                                    (G12 * (y[i + 1, j] + y[i, j + 1] - 2 * y[i, j]) + G11 * (
                                                -y[i + 1, j] + y[i, j]) + G22 * (-y[i, j + 1] + y[i, j])))) / \
                             (x[i, j] * (-y[i + 1, j] + y[i, j + 1]) + x[i, j + 1] * (y[i + 1, j] - y[i, j]) + x[
                                 i + 1, j] * (-y[i, j + 1] + y[i, j])) ** 2
                        Fxx = (2 * Dk * ((x[i + 1, j] - x[i, j + 1]) ** 2 + (y[i + 1, j] - y[i, j + 1]) ** 2) *
                               (G11 * (y[i + 1, j] - y[i, j]) ** 2 + (y[i, j + 1] - y[i, j]) * (
                                           G22 * (y[i, j + 1] - y[i, j]) +
                                           2 * G12 * (-y[i + 1, j] + y[i, j])))) / \
                              (x[i, j] * (-y[i + 1, j] + y[i, j + 1]) + x[i, j + 1] * (y[i + 1, j] - y[i, j]) + x[
                                  i + 1, j] * (-y[i, j + 1] + y[i, j])) ** 3
                        Fyy = (2 * Dk * (G11 * (x[i + 1, j] - x[i, j]) ** 2 + (x[i, j + 1] - x[i, j]) * (
                                    G22 * (x[i, j + 1] - x[i, j]) +
                                    2 * G12 * (-x[i + 1, j] + x[i, j]))) *
                               ((x[i + 1, j] - x[i, j + 1]) ** 2 + (y[i + 1, j] - y[i, j + 1]) ** 2)) / \
                              (x[i, j] * (-y[i + 1, j] + y[i, j + 1]) + x[i, j + 1] * (y[i + 1, j] - y[i, j]) + x[
                                  i + 1, j] * (-y[i, j + 1] + y[i, j])) ** 3
                        Fxy = -((2 * Dk * ((x[i + 1, j] - x[i, j + 1]) ** 2 + (y[i + 1, j] - y[i, j + 1]) ** 2) *
                                 (G11 * (x[i + 1, j] - x[i, j]) * (y[i + 1, j] - y[i, j]) +
                                  G22 * (x[i, j + 1] - x[i, j]) * (y[i, j + 1] - y[i, j]) +
                                  G12 * (-x[i, j + 1] * y[i + 1, j] + x[i, j] * y[i + 1, j] - x[i + 1, j] * y[i, j + 1] +
                                         x[i, j] * y[i, j + 1] + (x[i + 1, j] + x[i, j + 1] - 2 * x[i, j]) * y[i, j]))) / (
                                        x[i, j] * (-y[i + 1, j] + y[i, j + 1]) + x[i, j + 1] * (y[i + 1, j] - y[i, j]) +
                                        x[
                                            i + 1, j] * (-y[i, j + 1] + y[i, j])) ** 3)
                        Rx[i, j] += Fx / weight
                        Rxy[i, j] += Fxy/weight
                        Ry[i, j] += Fy/weight
                        Rxx[i, j] += Fxx/weight
                        Ryy[i, j] += Fyy/weight

                        # xk+1 : x[i, j + 1]
                        Fx = Dk * (-G11 * ((x[i + 1, j] - x[i, j]) ** 2 + (y[i + 1, j] - y[i, j]) ** 2) * (
                                    y[i + 1, j] - y[i, j]) +
                                   2 * G12 * ((x[i + 1, j] - x[i, j]) ** 2 + (y[i + 1, j] - y[i, j]) ** 2) * (
                                               y[i, j + 1] - y[i, j]) +
                                   G22 * (x[i, j + 1] ** 2 * (y[i + 1, j] - y[i, j]) +
                                          2 * x[i + 1, j] * x[i, j] * (y[i, j + 1] - y[i, j]) -
                                          (y[i + 1, j] - y[i, j]) * (y[i, j + 1] - y[i, j]) ** 2 +
                                          x[i, j] ** 2 * (y[i + 1, j] - 2 * y[i, j + 1] + y[i, j]) +
                                          2 * x[i, j + 1] * (x[i, j] * (-y[i + 1, j] + y[i, j + 1]) + x[i + 1, j] * (
                                                    -y[i, j + 1] + y[i, j])))) / \
                             (x[i, j] * (-y[i + 1, j] + y[i, j + 1]) + x[i, j + 1] * (y[i + 1, j] - y[i, j]) + x[
                                 i + 1, j] * (-y[i, j + 1] + y[i, j])) ** 2
                        Fy = Dk * (
                                (-(-x[i + 1, j] + x[i, j]) * (
                                            G11 * ((x[i + 1, j] - x[i, j]) ** 2 + (y[i + 1, j] - y[i, j]) ** 2)
                                            - 2 * G12 * ((x[i + 1, j] - x[i, j]) * (x[i, j + 1] - x[i, j]) + (
                                                y[i + 1, j] - y[i, j]) * (y[i, j + 1] - y[i, j]))
                                            + G22 * ((x[i, j + 1] - x[i, j]) ** 2 + (y[i, j + 1] - y[i, j]) ** 2)))
                                + 2 * (x[i, j] * (y[i + 1, j] - y[i, j + 1]) + x[i + 1, j] * (y[i, j + 1] - y[i, j]) + x[
                            i, j + 1] * (-y[i + 1, j] + y[i, j]))
                                * (G12 * (y[i + 1, j] - y[i, j]) + G22 * (-y[i, j + 1] + y[i, j]))
                        ) / (x[i, j] * (-y[i + 1, j] + y[i, j + 1]) + x[i, j + 1] * (y[i + 1, j] - y[i, j]) + x[
                            i + 1, j] * (-y[i, j + 1] + y[i, j])) ** 2
                        Fxx = 2 * Dk * ((x[i + 1, j] - x[i, j]) ** 2 + (y[i + 1, j] - y[i, j]) ** 2) * (
                                G11 * (y[i + 1, j] - y[i, j]) ** 2 +
                                (y[i, j + 1] - y[i, j]) * (
                                            G22 * (y[i, j + 1] - y[i, j]) + 2 * G12 * (-y[i + 1, j] + y[i, j]))
                        ) / (x[i, j] * (-y[i + 1, j] + y[i, j + 1]) + x[i, j + 1] * (y[i + 1, j] - y[i, j]) + x[
                            i + 1, j] * (-y[i, j + 1] + y[i, j])) ** 3
                        Fyy = (2 * Dk * (G11 * (x[i + 1, j] - x[i, j]) ** 2 +
                                         (x[i, j + 1] - x[i, j]) * (
                                                     G22 * (x[i, j + 1] - x[i, j]) + 2 * G12 * (-x[i + 1, j] + x[i, j]))) *
                               ((x[i + 1, j] - x[i, j]) ** 2 + (y[i + 1, j] - y[i, j]) ** 2)) / \
                              (x[i, j] * (-y[i + 1, j] + y[i, j + 1]) + x[i, j + 1] * (y[i + 1, j] - y[i, j]) + x[
                                  i + 1, j] * (-y[i, j + 1] + y[i, j])) ** 3
                        Fxy = (-2 * Dk * ((x[i + 1, j] - x[i, j]) ** 2 + (y[i + 1, j] - y[i, j]) ** 2) *
                               (G11 * (x[i + 1, j] - x[i, j]) * (y[i + 1, j] - y[i, j]) +
                                G22 * (x[i, j + 1] - x[i, j]) * (y[i, j + 1] - y[i, j]) +
                                G12 * (-x[i, j + 1] * y[i + 1, j] + x[i, j] * y[i + 1, j] - x[i + 1, j] * y[i, j + 1] + x[
                                           i, j] * y[i, j + 1] +
                                       (x[i + 1, j] + x[i, j + 1] - 2 * x[i, j]) * y[i, j])) *
                               (x[i, j] * (-y[i + 1, j] + y[i, j + 1]) + x[i, j + 1] * (y[i + 1, j] - y[i, j]) + x[
                                   i + 1, j] * (-y[i, j + 1] + y[i, j])) ** -3)
                        Rx[i, j + 1] += Fx / weight
                        Rxy[i, j + 1] += Fxy/weight
                        Ry[i, j + 1] += Fy/weight
                        Rxx[i, j + 1] += Fxx/weight
                        Ryy[i, j + 1] += Fyy/weight

                        # xk-1 : x[i + 1, j]
                        Fx = (Dk * (-G11 * ((x[i + 1, j] - x[i, j]) ** 2 + (y[i + 1, j] - y[i, j]) ** 2) * (
                                y[i + 1, j] - y[i, j]) +
                                    2 * G12 * ((x[i + 1, j] - x[i, j]) ** 2 + (y[i + 1, j] - y[i, j]) ** 2) * (
                                            y[i, j + 1] - y[i, j]) +
                                    G22 * (x[i, j + 1] ** 2 * (y[i + 1, j] - y[i, j]) +
                                           2 * x[i + 1, j] * x[i, j] * (y[i, j + 1] - y[i, j]) -
                                           (y[i + 1, j] - y[i, j]) * (y[i, j + 1] - y[i, j]) ** 2 +
                                           x[i, j] ** 2 * (y[i + 1, j] - 2 * y[i, j + 1] + y[i, j]) +
                                           2 * x[i, j + 1] * (x[i, j] * (-y[i + 1, j] + y[i, j + 1]) +
                                                              x[i + 1, j] * (-y[i, j + 1] + y[i, j]))))) / \
                             (x[i, j] * (-y[i + 1, j] + y[i, j + 1]) + x[i, j + 1] * (y[i + 1, j] - y[i, j]) + x[
                                 i + 1, j] * (-y[i, j + 1] + y[i, j])) ** 2
                        Fy = (Dk * (-(-x[i + 1, j] + x[i, j]) * (
                                G11 * ((x[i + 1, j] - x[i, j]) ** 2 + (y[i + 1, j] - y[i, j]) ** 2) -
                                2 * G12 * ((x[i + 1, j] - x[i, j]) * (x[i, j + 1] - x[i, j]) + (
                                y[i + 1, j] - y[i, j]) * (
                                                   y[i, j + 1] - y[i, j])) +
                                G22 * ((x[i, j + 1] - x[i, j]) ** 2 + (y[i, j + 1] - y[i, j]) ** 2)) +
                                    2 * (x[i, j] * (y[i + 1, j] - y[i, j + 1]) + x[i + 1, j] * (y[i, j + 1] - y[i, j]) + x[
                                    i, j + 1] * (-y[i + 1, j] + y[i, j])) *
                                    (G12 * (y[i + 1, j] - y[i, j]) + G22 * (-y[i, j + 1] + y[i, j])))) / \
                             (x[i, j] * (-y[i + 1, j] + y[i, j + 1]) + x[i, j + 1] * (y[i + 1, j] - y[i, j]) + x[
                                 i + 1, j] * (-y[i, j + 1] + y[i, j])) ** 2
                        Fxx = (2 * Dk * ((x[i + 1, j] - x[i, j]) ** 2 + (y[i + 1, j] - y[i, j]) ** 2) *
                               (G11 * (y[i + 1, j] - y[i, j]) ** 2 + (y[i, j + 1] - y[i, j]) * (
                                       G22 * (y[i, j + 1] - y[i, j]) +
                                       2 * G12 * (-y[i + 1, j] + y[i, j])))) / \
                              (x[i, j] * (-y[i + 1, j] + y[i, j + 1]) + x[i, j + 1] * (y[i + 1, j] - y[i, j]) + x[
                                  i + 1, j] * (-y[i, j + 1] + y[i, j])) ** 3
                        Fyy = (2 * Dk * (G11 * (x[i + 1, j] - x[i, j]) ** 2 +
                                         (x[i, j + 1] - x[i, j]) * (
                                                 G22 * (x[i, j + 1] - x[i, j]) + 2 * G12 * (-x[i + 1, j] + x[i, j]))) *
                               ((x[i + 1, j] - x[i, j]) ** 2 + (y[i + 1, j] - y[i, j]) ** 2)) / \
                              (x[i, j] * (-y[i + 1, j] + y[i, j + 1]) + x[i, j + 1] * (y[i + 1, j] - y[i, j]) + x[
                                  i + 1, j] * (-y[i, j + 1] + y[i, j])) ** 3
                        Fxy = -((2 * Dk * ((x[i + 1, j] - x[i, j]) ** 2 + (y[i + 1, j] - y[i, j]) ** 2) *
                                 (G11 * (x[i + 1, j] - x[i, j]) * (y[i + 1, j] - y[i, j]) +
                                  G22 * (x[i, j + 1] - x[i, j]) * (y[i, j + 1] - y[i, j]) +
                                  G12 * (-x[i, j + 1] * y[i + 1, j] + x[i, j] * y[i + 1, j] - x[i + 1, j] * y[i, j + 1] +
                                         x[i, j] * y[i, j + 1] + (x[i + 1, j] + x[i, j + 1] - 2 * x[i, j]) * y[i, j])))) / \
                              (x[i, j] * (-y[i + 1, j] + y[i, j + 1]) + x[i, j + 1] * (y[i + 1, j] - y[i, j]) + x[
                                  i + 1, j] * (-y[i, j + 1] + y[i, j])) ** 3
                        Rx[i + 1, j] += Fx / weight
                        Rxy[i + 1, j] += Fxy/weight
                        Ry[i + 1, j] += Fy/weight
                        Rxx[i + 1, j] += Fxx/weight
                        Ryy[i + 1, j] += Fyy/weight

                elif k == 1:    # rt corner perf xk=x[i+1, j+1], xk+1=x[i+1, j], xk-1=x[i, j+1]
                    G11 = ((X[i + 1, j] - X[i + 1, j + 1]) ** 2) + (Y[i + 1, j] - Y[i + 1, j + 1]) ** 2
                    G12 = (X[i + 1, j] - X[i + 1, j + 1]) * (X[i, j + 1] - X[i + 1, j + 1]) + (
                                Y[i + 1, j] - Y[i + 1, j + 1]) * (
                                  Y[i, j + 1] - Y[i + 1, j + 1])
                    G22 = ((X[i, j + 1] - X[i + 1, j + 1]) ** 2) + (Y[i, j + 1] - Y[i + 1, j + 1]) ** 2
                    Dk = (X[i + 1, j] - X[i + 1, j + 1]) * (Y[i, j + 1] - Y[i + 1, j + 1]) - (
                            X[i, j + 1] - X[i + 1, j + 1]) * (
                                 Y[i + 1, j] - Y[i + 1, j + 1])
                    Jk = (x[i + 1, j] - x[i + 1, j + 1]) * (y[i, j + 1] - y[i + 1, j + 1]) - (
                                x[i, j + 1] - x[i + 1, j + 1]) * (
                                 y[i + 1, j] - y[i + 1, j + 1])
                    if Jk <= 0:
                        print("2: ", Jk)
                        flag = 1
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
                    if i != 1 and j != 1 and i != nx - 2 and j != nx - 2:
                        # xk : x[i + 1, j + 1]
                        Fx = (Dk * (-(-y[i, j + 1] + y[i + 1, j]) * (
                                G11 * ((x[i, j + 1] - x[i + 1, j + 1]) ** 2 + (y[i, j + 1] - y[i + 1, j + 1]) ** 2) -
                                2 * G12 * ((x[i, j + 1] - x[i + 1, j + 1]) * (x[i + 1, j] - x[i + 1, j + 1]) + (
                                y[i, j + 1] - y[i + 1, j + 1]) * (
                                                   y[i + 1, j] - y[i + 1, j + 1])) +
                                G22 * ((x[i + 1, j] - x[i + 1, j + 1]) ** 2 + (y[i + 1, j] - y[i + 1, j + 1]) ** 2)) +
                                    2 * (-G12 * (x[i, j + 1] + x[i + 1, j] - 2 * x[i + 1, j + 1]) + G11 * (
                                            x[i, j + 1] - x[i + 1, j + 1]) +
                                         G22 * (x[i + 1, j] - x[i + 1, j + 1])) * (
                                            x[i + 1, j + 1] * (y[i, j + 1] - y[i + 1, j]) + x[i, j + 1] * (
                                            y[i + 1, j] - y[i + 1, j + 1]) +
                                            x[i + 1, j] * (-y[i, j + 1] + y[i + 1, j + 1])))) / \
                             (x[i + 1, j + 1] * (-y[i, j + 1] + y[i + 1, j]) + x[i + 1, j] * (
                                         y[i, j + 1] - y[i + 1, j + 1]) + x[
                                  i + 1, j] * (-y[i + 1, j] + y[i + 1, j + 1])) ** 2
                        Fy = (Dk * (-(x[i, j + 1] - x[i + 1, j]) * (
                                G11 * ((x[i, j + 1] - x[i + 1, j + 1]) ** 2 + (y[i, j + 1] - y[i + 1, j + 1]) ** 2) -
                                2 * G12 * ((x[i, j + 1] - x[i + 1, j + 1]) * (x[i + 1, j] - x[i + 1, j + 1]) + (
                                y[i, j + 1] - y[i + 1, j + 1]) * (
                                                   y[i + 1, j] - y[i + 1, j + 1])) +
                                G22 * ((x[i + 1, j] - x[i + 1, j + 1]) ** 2 + (y[i + 1, j] - y[i + 1, j + 1]) ** 2)) -
                                    2 * (x[i + 1, j + 1] * (y[i, j + 1] - y[i + 1, j]) + x[i, j + 1] * (
                                            y[i + 1, j] - y[i + 1, j + 1]) + x[
                                             i, j + 1] * (-y[i, j + 1] + y[i + 1, j + 1])) *
                                    (G12 * (y[i, j + 1] + y[i + 1, j] - 2 * y[i + 1, j + 1]) + G11 * (
                                            -y[i, j + 1] + y[i + 1, j + 1]) + G22 * (-y[i + 1, j] + y[i + 1, j + 1])))) / \
                             (x[i + 1, j + 1] * (-y[i, j + 1] + y[i + 1, j]) + x[i + 1, j] * (
                                         y[i, j + 1] - y[i + 1, j + 1]) + x[
                                  i + 1, j] * (-y[i + 1, j] + y[i + 1, j + 1])) ** 2
                        Fxx = (2 * Dk * ((x[i, j + 1] - x[i + 1, j]) ** 2 + (y[i, j + 1] - y[i + 1, j]) ** 2) *
                               (G11 * (y[i, j + 1] - y[i + 1, j + 1]) ** 2 + (y[i + 1, j] - y[i + 1, j + 1]) * (
                                       G22 * (y[i + 1, j] - y[i + 1, j + 1]) +
                                       2 * G12 * (-y[i, j + 1] + y[i + 1, j + 1])))) / \
                              (x[i + 1, j + 1] * (-y[i, j + 1] + y[i + 1, j]) + x[i + 1, j] * (
                                          y[i, j + 1] - y[i + 1, j + 1]) + x[
                                   i + 1, j] * (-y[i + 1, j] + y[i + 1, j + 1])) ** 3
                        Fyy = (2 * Dk * (G11 * (x[i, j + 1] - x[i + 1, j + 1]) ** 2 + (x[i + 1, j] - x[i + 1, j + 1]) * (
                                G22 * (x[i + 1, j] - x[i + 1, j + 1]) +
                                2 * G12 * (-x[i, j + 1] + x[i + 1, j + 1]))) *
                               ((x[i, j + 1] - x[i + 1, j]) ** 2 + (y[i, j + 1] - y[i + 1, j]) ** 2)) / \
                              (x[i + 1, j + 1] * (-y[i, j + 1] + y[i + 1, j]) + x[i + 1, j] * (
                                          y[i, j + 1] - y[i + 1, j + 1]) + x[
                                   i + 1, j] * (-y[i + 1, j] + y[i + 1, j + 1])) ** 3
                        Fxy = -((2 * Dk * ((x[i, j + 1] - x[i + 1, j]) ** 2 + (y[i, j + 1] - y[i + 1, j]) ** 2) *
                                 (G11 * (x[i, j + 1] - x[i + 1, j + 1]) * (y[i, j + 1] - y[i + 1, j + 1]) +
                                  G22 * (x[i + 1, j] - x[i + 1, j + 1]) * (y[i + 1, j] - y[i + 1, j + 1]) +
                                  G12 * (-x[i + 1, j] * y[i, j + 1] + x[i + 1, j + 1] * y[i, j + 1] - x[i, j + 1] * y[
                                             i + 1, j] +
                                         x[i + 1, j + 1] * y[i + 1, j] + (x[i, j + 1] + x[i + 1, j] - 2 * x[i + 1, j + 1]) *
                                         y[i + 1, j + 1]))) / (
                                        x[i + 1, j + 1] * (-y[i, j + 1] + y[i + 1, j]) + x[i + 1, j] * (
                                            y[i, j + 1] - y[i + 1, j + 1]) +
                                        x[
                                            i + 1, j] * (-y[i + 1, j] + y[i + 1, j + 1])) ** 3)
                        Rx[i + 1, j + 1] += Fx / weight
                        Rxy[i + 1, j + 1] += Fxy / weight
                        Ry[i + 1, j + 1] += Fy / weight
                        Rxx[i + 1, j + 1] += Fxx / weight
                        Ryy[i + 1, j + 1] += Fyy / weight

                        # xk+1 : x[i + 1, j]
                        Fx = Dk * (-G11 * ((x[i, j + 1] - x[i + 1, j + 1]) ** 2 + (y[i, j + 1] - y[i + 1, j + 1]) ** 2) * (
                                y[i, j + 1] - y[i + 1, j + 1]) +
                                   2 * G12 * ((x[i, j + 1] - x[i + 1, j + 1]) ** 2 + (
                                            y[i, j + 1] - y[i + 1, j + 1]) ** 2) * (
                                           y[i + 1, j] - y[i + 1, j + 1]) +
                                   G22 * (x[i + 1, j] ** 2 * (y[i, j + 1] - y[i + 1, j + 1]) +
                                          2 * x[i, j + 1] * x[i + 1, j + 1] * (y[i + 1, j] - y[i + 1, j + 1]) -
                                          (y[i, j + 1] - y[i + 1, j + 1]) * (y[i + 1, j] - y[i + 1, j + 1]) ** 2 +
                                          x[i + 1, j + 1] ** 2 * (y[i, j + 1] - 2 * y[i + 1, j] + y[i + 1, j + 1]) +
                                          2 * x[i + 1, j] * (
                                                      x[i + 1, j + 1] * (-y[i, j + 1] + y[i + 1, j]) + x[i, j + 1] * (
                                                      -y[i + 1, j] + y[i + 1, j + 1])))) / \
                             (x[i + 1, j + 1] * (-y[i, j + 1] + y[i + 1, j]) + x[i + 1, j] * (
                                         y[i, j + 1] - y[i + 1, j + 1]) + x[
                                  i + 1, j] * (-y[i + 1, j] + y[i + 1, j + 1])) ** 2
                        Fy = Dk * (
                                (-(-x[i, j + 1] + x[i + 1, j + 1]) * (
                                        G11 * ((x[i, j + 1] - x[i + 1, j + 1]) ** 2 + (y[i, j + 1] - y[i + 1, j + 1]) ** 2)
                                        - 2 * G12 * ((x[i, j + 1] - x[i + 1, j + 1]) * (x[i + 1, j] - x[i + 1, j + 1]) + (
                                        y[i, j + 1] - y[i + 1, j + 1]) * (y[i + 1, j] - y[i + 1, j + 1]))
                                        + G22 * ((x[i + 1, j] - x[i + 1, j + 1]) ** 2 + (
                                            y[i + 1, j] - y[i + 1, j + 1]) ** 2)))
                                + 2 * (x[i + 1, j + 1] * (y[i, j + 1] - y[i + 1, j]) + x[i, j + 1] * (
                                    y[i + 1, j] - y[i + 1, j + 1]) + x[
                                           i, j + 1] * (-y[i, j + 1] + y[i + 1, j + 1]))
                                * (G12 * (y[i, j + 1] - y[i + 1, j + 1]) + G22 * (-y[i + 1, j] + y[i + 1, j + 1]))
                        ) / (x[i + 1, j + 1] * (-y[i, j + 1] + y[i + 1, j]) + x[i + 1, j] * (
                                    y[i, j + 1] - y[i + 1, j + 1]) + x[
                                 i + 1, j] * (-y[i + 1, j] + y[i + 1, j + 1])) ** 2
                        Fxx = 2 * Dk * ((x[i, j + 1] - x[i + 1, j + 1]) ** 2 + (y[i, j + 1] - y[i + 1, j + 1]) ** 2) * (
                                G11 * (y[i, j + 1] - y[i + 1, j + 1]) ** 2 +
                                (y[i + 1, j] - y[i + 1, j + 1]) * (
                                        G22 * (y[i + 1, j] - y[i + 1, j + 1]) + 2 * G12 * (-y[i, j + 1] + y[i + 1, j + 1]))
                        ) / (x[i + 1, j + 1] * (-y[i, j + 1] + y[i + 1, j]) + x[i + 1, j] * (
                                    y[i, j + 1] - y[i + 1, j + 1]) + x[
                                 i + 1, j] * (-y[i + 1, j] + y[i + 1, j + 1])) ** 3
                        Fyy = (2 * Dk * (G11 * (x[i, j + 1] - x[i + 1, j + 1]) ** 2 +
                                         (x[i + 1, j] - x[i + 1, j + 1]) * (
                                                 G22 * (x[i + 1, j] - x[i + 1, j + 1]) + 2 * G12 * (
                                                     -x[i, j + 1] + x[i + 1, j + 1]))) *
                               ((x[i, j + 1] - x[i + 1, j + 1]) ** 2 + (y[i, j + 1] - y[i + 1, j + 1]) ** 2)) / \
                              (x[i + 1, j + 1] * (-y[i, j + 1] + y[i + 1, j]) + x[i + 1, j] * (
                                          y[i, j + 1] - y[i + 1, j + 1]) + x[
                                   i + 1, j] * (-y[i + 1, j] + y[i + 1, j + 1])) ** 3
                        Fxy = (-2 * Dk * ((x[i, j + 1] - x[i + 1, j + 1]) ** 2 + (y[i, j + 1] - y[i + 1, j + 1]) ** 2) *
                               (G11 * (x[i, j + 1] - x[i + 1, j + 1]) * (y[i, j + 1] - y[i + 1, j + 1]) +
                                G22 * (x[i + 1, j] - x[i + 1, j + 1]) * (y[i + 1, j] - y[i + 1, j + 1]) +
                                G12 * (-x[i + 1, j] * y[i, j + 1] + x[i + 1, j + 1] * y[i, j + 1] - x[i, j + 1] * y[
                                           i + 1, j] + x[
                                           i, j] * y[i + 1, j] +
                                       (x[i, j + 1] + x[i + 1, j] - 2 * x[i + 1, j + 1]) * y[i + 1, j + 1])) *
                               (x[i + 1, j + 1] * (-y[i, j + 1] + y[i + 1, j]) + x[i + 1, j] * (
                                           y[i, j + 1] - y[i + 1, j + 1]) + x[
                                    i + 1, j] * (-y[i + 1, j] + y[i + 1, j + 1])) ** -3)
                        Rx[i + 1, j] += Fx / weight
                        Rxy[i + 1, j] += Fxy / weight
                        Ry[i + 1, j] += Fy / weight
                        Rxx[i + 1, j] += Fxx / weight
                        Ryy[i + 1, j] += Fyy / weight

                        # xk-1 : x[i, j + 1]
                        Fx = (Dk * (-G11 * ((x[i, j + 1] - x[i + 1, j + 1]) ** 2 + (y[i, j + 1] - y[i + 1, j + 1]) ** 2) * (
                                y[i, j + 1] - y[i + 1, j + 1]) +
                                    2 * G12 * ((x[i, j + 1] - x[i + 1, j + 1]) ** 2 + (
                                            y[i, j + 1] - y[i + 1, j + 1]) ** 2) * (
                                            y[i + 1, j] - y[i + 1, j + 1]) +
                                    G22 * (x[i + 1, j] ** 2 * (y[i, j + 1] - y[i + 1, j + 1]) +
                                           2 * x[i, j + 1] * x[i + 1, j + 1] * (y[i + 1, j] - y[i + 1, j + 1]) -
                                           (y[i, j + 1] - y[i + 1, j + 1]) * (y[i + 1, j] - y[i + 1, j + 1]) ** 2 +
                                           x[i + 1, j + 1] ** 2 * (y[i, j + 1] - 2 * y[i + 1, j] + y[i + 1, j + 1]) +
                                           2 * x[i + 1, j] * (x[i + 1, j + 1] * (-y[i, j + 1] + y[i + 1, j]) +
                                                              x[i, j + 1] * (-y[i + 1, j] + y[i + 1, j + 1]))))) / \
                             (x[i + 1, j + 1] * (-y[i, j + 1] + y[i + 1, j]) + x[i + 1, j] * (
                                         y[i, j + 1] - y[i + 1, j + 1]) + x[
                                  i + 1, j] * (-y[i + 1, j] + y[i + 1, j + 1])) ** 2
                        Fy = (Dk * (-(-x[i, j + 1] + x[i + 1, j + 1]) * (
                                G11 * ((x[i, j + 1] - x[i + 1, j + 1]) ** 2 + (y[i, j + 1] - y[i + 1, j + 1]) ** 2) -
                                2 * G12 * ((x[i, j + 1] - x[i + 1, j + 1]) * (x[i + 1, j] - x[i + 1, j + 1]) + (
                                y[i, j + 1] - y[i + 1, j + 1]) * (
                                                   y[i + 1, j] - y[i + 1, j + 1])) +
                                G22 * ((x[i + 1, j] - x[i + 1, j + 1]) ** 2 + (y[i + 1, j] - y[i + 1, j + 1]) ** 2)) +
                                    2 * (x[i + 1, j + 1] * (y[i, j + 1] - y[i + 1, j]) + x[i, j + 1] * (
                                            y[i + 1, j] - y[i + 1, j + 1]) + x[
                                             i, j + 1] * (-y[i, j + 1] + y[i + 1, j + 1])) *
                                    (G12 * (y[i, j + 1] - y[i + 1, j + 1]) + G22 * (-y[i + 1, j] + y[i + 1, j + 1])))) / \
                             (x[i + 1, j + 1] * (-y[i, j + 1] + y[i + 1, j]) + x[i + 1, j] * (
                                         y[i, j + 1] - y[i + 1, j + 1]) + x[
                                  i + 1, j] * (-y[i + 1, j] + y[i + 1, j + 1])) ** 2
                        Fxx = (2 * Dk * ((x[i, j + 1] - x[i + 1, j + 1]) ** 2 + (y[i, j + 1] - y[i + 1, j + 1]) ** 2) *
                               (G11 * (y[i, j + 1] - y[i + 1, j + 1]) ** 2 + (y[i + 1, j] - y[i + 1, j + 1]) * (
                                       G22 * (y[i + 1, j] - y[i + 1, j + 1]) +
                                       2 * G12 * (-y[i, j + 1] + y[i + 1, j + 1])))) / \
                              (x[i + 1, j + 1] * (-y[i, j + 1] + y[i + 1, j]) + x[i + 1, j] * (
                                          y[i, j + 1] - y[i + 1, j + 1]) + x[
                                   i + 1, j] * (-y[i + 1, j] + y[i + 1, j + 1])) ** 3
                        Fyy = (2 * Dk * (G11 * (x[i, j + 1] - x[i + 1, j + 1]) ** 2 +
                                         (x[i + 1, j] - x[i + 1, j + 1]) * (
                                                 G22 * (x[i + 1, j] - x[i + 1, j + 1]) + 2 * G12 * (
                                                     -x[i, j + 1] + x[i + 1, j + 1]))) *
                               ((x[i, j + 1] - x[i + 1, j + 1]) ** 2 + (y[i, j + 1] - y[i + 1, j + 1]) ** 2)) / \
                              (x[i + 1, j + 1] * (-y[i, j + 1] + y[i + 1, j]) + x[i + 1, j] * (
                                          y[i, j + 1] - y[i + 1, j + 1]) + x[
                                   i + 1, j] * (-y[i + 1, j] + y[i + 1, j + 1])) ** 3
                        Fxy = -((2 * Dk * ((x[i, j + 1] - x[i + 1, j + 1]) ** 2 + (y[i, j + 1] - y[i + 1, j + 1]) ** 2) *
                                 (G11 * (x[i, j + 1] - x[i + 1, j + 1]) * (y[i, j + 1] - y[i + 1, j + 1]) +
                                  G22 * (x[i + 1, j] - x[i + 1, j + 1]) * (y[i + 1, j] - y[i + 1, j + 1]) +
                                  G12 * (-x[i + 1, j] * y[i, j + 1] + x[i + 1, j + 1] * y[i, j + 1] - x[i, j + 1] * y[
                                             i + 1, j] +
                                         x[i + 1, j + 1] * y[i + 1, j] + (x[i, j + 1] + x[i + 1, j] - 2 * x[i + 1, j + 1]) *
                                         y[i + 1, j + 1])))) / \
                              (x[i + 1, j + 1] * (-y[i, j + 1] + y[i + 1, j]) + x[i + 1, j] * (
                                          y[i, j + 1] - y[i + 1, j + 1]) + x[
                                   i + 1, j] * (-y[i + 1, j] + y[i + 1, j + 1])) ** 3
                        Rx[i, j + 1] += Fx / weight
                        Rxy[i, j + 1] += Fxy / weight
                        Ry[i, j + 1] += Fy / weight
                        Rxx[i, j + 1] += Fxx / weight
                        Ryy[i, j + 1] += Fyy / weight

                elif k == 2:    # rb corner perf xk = x[i+1, j], xk+1=x[i+1, j+1], xk-1=x[i, j]
                    G11 = ((X[i + 1, j + 1] - X[i, j + 1]) ** 2) + (Y[i + 1, j + 1] - Y[i, j + 1]) ** 2
                    G12 = (X[i + 1, j + 1] - X[i, j + 1]) * (X[i, j] - X[i, j + 1]) + (
                            Y[i + 1, j + 1] - Y[i, j + 1]) * (
                                  Y[i, j] - Y[i, j + 1])
                    G22 = (X[i, j] - X[i, j + 1]) ** 2 + (Y[i, j] - Y[i, j + 1]) ** 2
                    Jk = (x[i + 1, j + 1] - x[i, j + 1]) * (y[i, j] - y[i, j + 1]) - (x[i, j] - x[i, j + 1]) * (
                            y[i + 1, j + 1] - y[i, j + 1])
                    if Jk <= 0:
                        print("3: ", Jk)
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
                    if i != 1 and j != 1 and i != nx - 2 and j != nx - 2:
                        # xk : x[i + 1, j]
                        Fx = (Dk * (-(-y[i, j] + y[i + 1, j + 1]) * (
                                G11 * ((x[i, j] - x[i + 1, j]) ** 2 + (y[i, j] - y[i + 1, j]) ** 2) -
                                2 * G12 * ((x[i, j] - x[i + 1, j]) * (x[i + 1, j + 1] - x[i + 1, j]) + (
                                y[i, j] - y[i + 1, j]) * (
                                                   y[i + 1, j + 1] - y[i + 1, j])) +
                                G22 * ((x[i + 1, j + 1] - x[i + 1, j]) ** 2 + (y[i + 1, j + 1] - y[i + 1, j]) ** 2)) +
                                    2 * (-G12 * (x[i, j] + x[i + 1, j + 1] - 2 * x[i + 1, j]) + G11 * (
                                            x[i, j] - x[i + 1, j]) +
                                         G22 * (x[i + 1, j + 1] - x[i + 1, j])) * (
                                            x[i + 1, j] * (y[i, j] - y[i + 1, j + 1]) + x[i, j] * (
                                            y[i + 1, j + 1] - y[i + 1, j]) +
                                            x[i + 1, j + 1] * (-y[i, j] + y[i + 1, j])))) / \
                             (x[i + 1, j] * (-y[i, j] + y[i + 1, j + 1]) + x[i + 1, j + 1] * (y[i, j] - y[i + 1, j]) + x[
                                 i + 1, j] * (-y[i + 1, j + 1] + y[i + 1, j])) ** 2
                        Fy = (Dk * (-(x[i, j] - x[i + 1, j + 1]) * (
                                G11 * ((x[i, j] - x[i + 1, j]) ** 2 + (y[i, j] - y[i + 1, j]) ** 2) -
                                2 * G12 * ((x[i, j] - x[i + 1, j]) * (x[i + 1, j + 1] - x[i + 1, j]) + (
                                y[i, j] - y[i + 1, j]) * (
                                                   y[i + 1, j + 1] - y[i + 1, j])) +
                                G22 * ((x[i + 1, j + 1] - x[i + 1, j]) ** 2 + (y[i + 1, j + 1] - y[i + 1, j]) ** 2)) -
                                    2 * (x[i + 1, j] * (y[i, j] - y[i + 1, j + 1]) + x[i, j] * (
                                            y[i + 1, j + 1] - y[i + 1, j]) + x[
                                             i, j + 1] * (-y[i, j] + y[i + 1, j])) *
                                    (G12 * (y[i, j] + y[i + 1, j + 1] - 2 * y[i + 1, j]) + G11 * (
                                            -y[i, j] + y[i + 1, j]) + G22 * (-y[i + 1, j + 1] + y[i + 1, j])))) / \
                             (x[i + 1, j] * (-y[i, j] + y[i + 1, j + 1]) + x[i + 1, j + 1] * (y[i, j] - y[i + 1, j]) + x[
                                 i + 1, j] * (-y[i + 1, j + 1] + y[i + 1, j])) ** 2
                        Fxx = (2 * Dk * ((x[i, j] - x[i + 1, j + 1]) ** 2 + (y[i, j] - y[i + 1, j + 1]) ** 2) *
                               (G11 * (y[i, j] - y[i + 1, j]) ** 2 + (y[i + 1, j + 1] - y[i + 1, j]) * (
                                       G22 * (y[i + 1, j + 1] - y[i + 1, j]) +
                                       2 * G12 * (-y[i, j] + y[i + 1, j])))) / \
                              (x[i + 1, j] * (-y[i, j] + y[i + 1, j + 1]) + x[i + 1, j + 1] * (y[i, j] - y[i + 1, j]) + x[
                                  i + 1, j] * (-y[i + 1, j + 1] + y[i + 1, j])) ** 3
                        Fyy = (2 * Dk * (G11 * (x[i, j] - x[i + 1, j]) ** 2 + (x[i + 1, j + 1] - x[i + 1, j]) * (
                                G22 * (x[i + 1, j + 1] - x[i + 1, j]) +
                                2 * G12 * (-x[i, j] + x[i + 1, j]))) *
                               ((x[i, j] - x[i + 1, j + 1]) ** 2 + (y[i, j] - y[i + 1, j + 1]) ** 2)) / \
                              (x[i + 1, j] * (-y[i, j] + y[i + 1, j + 1]) + x[i + 1, j + 1] * (y[i, j] - y[i + 1, j]) + x[
                                  i + 1, j] * (-y[i + 1, j + 1] + y[i + 1, j])) ** 3
                        Fxy = -((2 * Dk * ((x[i, j] - x[i + 1, j + 1]) ** 2 + (y[i, j] - y[i + 1, j + 1]) ** 2) *
                                 (G11 * (x[i, j] - x[i + 1, j]) * (y[i, j] - y[i + 1, j]) +
                                  G22 * (x[i + 1, j + 1] - x[i + 1, j]) * (y[i + 1, j + 1] - y[i + 1, j]) +
                                  G12 * (-x[i + 1, j + 1] * y[i, j] + x[i + 1, j] * y[i, j] - x[i, j] * y[i + 1, j + 1] +
                                         x[i + 1, j] * y[i + 1, j + 1] + (x[i, j] + x[i + 1, j + 1] - 2 * x[i + 1, j]) * y[
                                             i + 1, j]))) / (
                                        x[i + 1, j] * (-y[i, j] + y[i + 1, j + 1]) + x[i + 1, j + 1] * (
                                            y[i, j] - y[i + 1, j]) +
                                        x[
                                            i + 1, j] * (-y[i + 1, j + 1] + y[i + 1, j])) ** 3)
                        Rx[i + 1, j] += Fx / weight
                        Rxy[i + 1, j] += Fxy / weight
                        Ry[i + 1, j] += Fy / weight
                        Rxx[i + 1, j] += Fxx / weight
                        Ryy[i + 1, j] += Fyy / weight

                        # xk+1 : x[i + 1, j + 1]
                        Fx = Dk * (-G11 * ((x[i, j] - x[i + 1, j]) ** 2 + (y[i, j] - y[i + 1, j]) ** 2) * (
                                y[i, j] - y[i + 1, j]) +
                                   2 * G12 * ((x[i, j] - x[i + 1, j]) ** 2 + (y[i, j] - y[i + 1, j]) ** 2) * (
                                           y[i + 1, j + 1] - y[i + 1, j]) +
                                   G22 * (x[i + 1, j + 1] ** 2 * (y[i, j] - y[i + 1, j]) +
                                          2 * x[i, j] * x[i + 1, j] * (y[i + 1, j + 1] - y[i + 1, j]) -
                                          (y[i, j] - y[i + 1, j]) * (y[i + 1, j + 1] - y[i + 1, j]) ** 2 +
                                          x[i + 1, j] ** 2 * (y[i, j] - 2 * y[i + 1, j + 1] + y[i + 1, j]) +
                                          2 * x[i + 1, j + 1] * (x[i + 1, j] * (-y[i, j] + y[i + 1, j + 1]) + x[i, j] * (
                                                -y[i + 1, j + 1] + y[i + 1, j])))) / \
                             (x[i + 1, j] * (-y[i, j] + y[i + 1, j + 1]) + x[i + 1, j + 1] * (y[i, j] - y[i + 1, j]) + x[
                                 i + 1, j] * (-y[i + 1, j + 1] + y[i + 1, j])) ** 2
                        Fy = Dk * (
                                (-(-x[i, j] + x[i + 1, j]) * (
                                        G11 * ((x[i, j] - x[i + 1, j]) ** 2 + (y[i, j] - y[i + 1, j]) ** 2)
                                        - 2 * G12 * ((x[i, j] - x[i + 1, j]) * (x[i + 1, j + 1] - x[i + 1, j]) + (
                                        y[i, j] - y[i + 1, j]) * (y[i + 1, j + 1] - y[i + 1, j]))
                                        + G22 * ((x[i + 1, j + 1] - x[i + 1, j]) ** 2 + (
                                            y[i + 1, j + 1] - y[i + 1, j]) ** 2)))
                                + 2 * (x[i + 1, j] * (y[i, j] - y[i + 1, j + 1]) + x[i, j] * (
                                    y[i + 1, j + 1] - y[i + 1, j]) + x[
                                           i, j + 1] * (-y[i, j] + y[i + 1, j]))
                                * (G12 * (y[i, j] - y[i + 1, j]) + G22 * (-y[i + 1, j + 1] + y[i + 1, j]))
                        ) / (x[i + 1, j] * (-y[i, j] + y[i + 1, j + 1]) + x[i + 1, j + 1] * (y[i, j] - y[i + 1, j]) + x[
                            i + 1, j] * (-y[i + 1, j + 1] + y[i + 1, j])) ** 2
                        Fxx = 2 * Dk * ((x[i, j] - x[i + 1, j]) ** 2 + (y[i, j] - y[i + 1, j]) ** 2) * (
                                G11 * (y[i, j] - y[i + 1, j]) ** 2 +
                                (y[i + 1, j + 1] - y[i + 1, j]) * (
                                        G22 * (y[i + 1, j + 1] - y[i + 1, j]) + 2 * G12 * (-y[i, j] + y[i + 1, j]))
                        ) / (x[i + 1, j] * (-y[i, j] + y[i + 1, j + 1]) + x[i + 1, j + 1] * (y[i, j] - y[i + 1, j]) + x[
                            i + 1, j] * (-y[i + 1, j + 1] + y[i + 1, j])) ** 3
                        Fyy = (2 * Dk * (G11 * (x[i, j] - x[i + 1, j]) ** 2 +
                                         (x[i + 1, j + 1] - x[i + 1, j]) * (
                                                 G22 * (x[i + 1, j + 1] - x[i + 1, j]) + 2 * G12 * (
                                                     -x[i, j] + x[i + 1, j]))) *
                               ((x[i, j] - x[i + 1, j]) ** 2 + (y[i, j] - y[i + 1, j]) ** 2)) / \
                              (x[i + 1, j] * (-y[i, j] + y[i + 1, j + 1]) + x[i + 1, j + 1] * (y[i, j] - y[i + 1, j]) + x[
                                  i + 1, j] * (-y[i + 1, j + 1] + y[i + 1, j])) ** 3
                        Fxy = (-2 * Dk * ((x[i, j] - x[i + 1, j]) ** 2 + (y[i, j] - y[i + 1, j]) ** 2) *
                               (G11 * (x[i, j] - x[i + 1, j]) * (y[i, j] - y[i + 1, j]) +
                                G22 * (x[i + 1, j + 1] - x[i + 1, j]) * (y[i + 1, j + 1] - y[i + 1, j]) +
                                G12 * (-x[i + 1, j + 1] * y[i, j] + x[i + 1, j] * y[i, j] - x[i, j] * y[i + 1, j + 1] + x[
                                           i, j] * y[i + 1, j + 1] +
                                       (x[i, j] + x[i + 1, j + 1] - 2 * x[i + 1, j]) * y[i + 1, j])) *
                               (x[i + 1, j] * (-y[i, j] + y[i + 1, j + 1]) + x[i + 1, j + 1] * (y[i, j] - y[i + 1, j]) + x[
                                   i + 1, j] * (-y[i + 1, j + 1] + y[i + 1, j])) ** -3)
                        Rx[i + 1, j + 1] += Fx / weight
                        Rxy[i + 1, j + 1] += Fxy / weight
                        Ry[i + 1, j + 1] += Fy / weight
                        Rxx[i + 1, j + 1] += Fxx / weight
                        Ryy[i + 1, j + 1] += Fyy / weight

                        # xk-1 : x[i, j]
                        Fx = (Dk * (-G11 * ((x[i, j] - x[i + 1, j]) ** 2 + (y[i, j] - y[i + 1, j]) ** 2) * (
                                y[i, j] - y[i + 1, j]) +
                                    2 * G12 * ((x[i, j] - x[i + 1, j]) ** 2 + (y[i, j] - y[i + 1, j]) ** 2) * (
                                            y[i + 1, j + 1] - y[i + 1, j]) +
                                    G22 * (x[i + 1, j + 1] ** 2 * (y[i, j] - y[i + 1, j]) +
                                           2 * x[i, j] * x[i + 1, j] * (y[i + 1, j + 1] - y[i + 1, j]) -
                                           (y[i, j] - y[i + 1, j]) * (y[i + 1, j + 1] - y[i + 1, j]) ** 2 +
                                           x[i + 1, j] ** 2 * (y[i, j] - 2 * y[i + 1, j + 1] + y[i + 1, j]) +
                                           2 * x[i + 1, j + 1] * (x[i + 1, j] * (-y[i, j] + y[i + 1, j + 1]) +
                                                                  x[i, j] * (-y[i + 1, j + 1] + y[i + 1, j]))))) / \
                             (x[i + 1, j] * (-y[i, j] + y[i + 1, j + 1]) + x[i + 1, j + 1] * (y[i, j] - y[i + 1, j]) + x[
                                 i + 1, j] * (-y[i + 1, j + 1] + y[i + 1, j])) ** 2
                        Fy = (Dk * (-(-x[i, j] + x[i + 1, j]) * (
                                G11 * ((x[i, j] - x[i + 1, j]) ** 2 + (y[i, j] - y[i + 1, j]) ** 2) -
                                2 * G12 * ((x[i, j] - x[i + 1, j]) * (x[i + 1, j + 1] - x[i + 1, j]) + (
                                y[i, j] - y[i + 1, j]) * (
                                                   y[i + 1, j + 1] - y[i + 1, j])) +
                                G22 * ((x[i + 1, j + 1] - x[i + 1, j]) ** 2 + (y[i + 1, j + 1] - y[i + 1, j]) ** 2)) +
                                    2 * (x[i + 1, j] * (y[i, j] - y[i + 1, j + 1]) + x[i, j] * (
                                            y[i + 1, j + 1] - y[i + 1, j]) + x[
                                             i, j + 1] * (-y[i, j] + y[i + 1, j])) *
                                    (G12 * (y[i, j] - y[i + 1, j]) + G22 * (-y[i + 1, j + 1] + y[i + 1, j])))) / \
                             (x[i + 1, j] * (-y[i, j] + y[i + 1, j + 1]) + x[i + 1, j + 1] * (y[i, j] - y[i + 1, j]) + x[
                                 i + 1, j] * (-y[i + 1, j + 1] + y[i + 1, j])) ** 2
                        Fxx = (2 * Dk * ((x[i, j] - x[i + 1, j]) ** 2 + (y[i, j] - y[i + 1, j]) ** 2) *
                               (G11 * (y[i, j] - y[i + 1, j]) ** 2 + (y[i + 1, j + 1] - y[i + 1, j]) * (
                                       G22 * (y[i + 1, j + 1] - y[i + 1, j]) +
                                       2 * G12 * (-y[i, j] + y[i + 1, j])))) / \
                              (x[i + 1, j] * (-y[i, j] + y[i + 1, j + 1]) + x[i + 1, j + 1] * (y[i, j] - y[i + 1, j]) + x[
                                  i + 1, j] * (-y[i + 1, j + 1] + y[i + 1, j])) ** 3
                        Fyy = (2 * Dk * (G11 * (x[i, j] - x[i + 1, j]) ** 2 +
                                         (x[i + 1, j + 1] - x[i + 1, j]) * (
                                                 G22 * (x[i + 1, j + 1] - x[i + 1, j]) + 2 * G12 * (
                                                     -x[i, j] + x[i + 1, j]))) *
                               ((x[i, j] - x[i + 1, j]) ** 2 + (y[i, j] - y[i + 1, j]) ** 2)) / \
                              (x[i + 1, j] * (-y[i, j] + y[i + 1, j + 1]) + x[i + 1, j + 1] * (y[i, j] - y[i + 1, j]) + x[
                                  i + 1, j] * (-y[i + 1, j + 1] + y[i + 1, j])) ** 3
                        Fxy = -((2 * Dk * ((x[i, j] - x[i + 1, j]) ** 2 + (y[i, j] - y[i + 1, j]) ** 2) *
                                 (G11 * (x[i, j] - x[i + 1, j]) * (y[i, j] - y[i + 1, j]) +
                                  G22 * (x[i + 1, j + 1] - x[i + 1, j]) * (y[i + 1, j + 1] - y[i + 1, j]) +
                                  G12 * (-x[i + 1, j + 1] * y[i, j] + x[i + 1, j] * y[i, j] - x[i, j] * y[i + 1, j + 1] +
                                         x[i + 1, j] * y[i + 1, j + 1] + (x[i, j] + x[i + 1, j + 1] - 2 * x[i + 1, j]) * y[
                                             i + 1, j])))) / \
                              (x[i + 1, j] * (-y[i, j] + y[i + 1, j + 1]) + x[i + 1, j + 1] * (y[i, j] - y[i + 1, j]) + x[
                                  i + 1, j] * (-y[i + 1, j + 1] + y[i + 1, j])) ** 3
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
                        print("4: ", Jk, "\n", x[i, j], x[i + 1, j], x[i + 1, j + 1])
                        print("4.2: ", Jk, "\n", y[i, j], y[i + 1, j], y[i + 1, j + 1])
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
                    if i != 1 and j != 1 and i != nx - 2 and j != nx - 2:
                        # xk : x[i + 1, j]
                        Fx = (Dk * (-(-y[i + 1, j + 1] + y[i, j]) * (
                                G11 * ((x[i + 1, j + 1] - x[i + 1, j]) ** 2 + (y[i + 1, j + 1] - y[i + 1, j]) ** 2) -
                                2 * G12 * ((x[i + 1, j + 1] - x[i + 1, j]) * (x[i, j] - x[i + 1, j]) + (
                                y[i + 1, j + 1] - y[i + 1, j]) * (
                                                   y[i, j] - y[i + 1, j])) +
                                G22 * ((x[i, j] - x[i + 1, j]) ** 2 + (y[i, j] - y[i + 1, j]) ** 2)) +
                                    2 * (-G12 * (x[i + 1, j + 1] + x[i, j] - 2 * x[i + 1, j]) + G11 * (
                                            x[i + 1, j + 1] - x[i + 1, j]) +
                                         G22 * (x[i, j] - x[i + 1, j])) * (
                                            x[i + 1, j] * (y[i + 1, j + 1] - y[i, j]) + x[i + 1, j + 1] * (
                                            y[i, j] - y[i + 1, j]) +
                                            x[i, j] * (-y[i + 1, j + 1] + y[i + 1, j])))) / \
                             (x[i + 1, j] * (-y[i + 1, j + 1] + y[i, j]) + x[i, j] * (y[i + 1, j + 1] - y[i + 1, j]) + x[
                                 i + 1, j] * (-y[i, j] + y[i + 1, j])) ** 2
                        Fy = (Dk * (-(x[i + 1, j + 1] - x[i, j]) * (
                                G11 * ((x[i + 1, j + 1] - x[i + 1, j]) ** 2 + (y[i + 1, j + 1] - y[i + 1, j]) ** 2) -
                                2 * G12 * ((x[i + 1, j + 1] - x[i + 1, j]) * (x[i, j] - x[i + 1, j]) + (
                                y[i + 1, j + 1] - y[i + 1, j]) * (
                                                   y[i, j] - y[i + 1, j])) +
                                G22 * ((x[i, j] - x[i + 1, j]) ** 2 + (y[i, j] - y[i + 1, j]) ** 2)) -
                                    2 * (x[i + 1, j] * (y[i + 1, j + 1] - y[i, j]) + x[i + 1, j + 1] * (
                                            y[i, j] - y[i + 1, j]) + x[
                                             i, j + 1] * (-y[i + 1, j + 1] + y[i + 1, j])) *
                                    (G12 * (y[i + 1, j + 1] + y[i, j] - 2 * y[i + 1, j]) + G11 * (
                                            -y[i + 1, j + 1] + y[i + 1, j]) + G22 * (-y[i, j] + y[i + 1, j])))) / \
                             (x[i + 1, j] * (-y[i + 1, j + 1] + y[i, j]) + x[i, j] * (y[i + 1, j + 1] - y[i + 1, j]) + x[
                                 i + 1, j] * (-y[i, j] + y[i + 1, j])) ** 2
                        Fxx = (2 * Dk * ((x[i + 1, j + 1] - x[i, j]) ** 2 + (y[i + 1, j + 1] - y[i, j]) ** 2) *
                               (G11 * (y[i + 1, j + 1] - y[i + 1, j]) ** 2 + (y[i, j] - y[i + 1, j]) * (
                                       G22 * (y[i, j] - y[i + 1, j]) +
                                       2 * G12 * (-y[i + 1, j + 1] + y[i + 1, j])))) / \
                              (x[i + 1, j] * (-y[i + 1, j + 1] + y[i, j]) + x[i, j] * (y[i + 1, j + 1] - y[i + 1, j]) + x[
                                  i + 1, j] * (-y[i, j] + y[i + 1, j])) ** 3
                        Fyy = (2 * Dk * (G11 * (x[i + 1, j + 1] - x[i + 1, j]) ** 2 + (x[i, j] - x[i + 1, j]) * (
                                G22 * (x[i, j] - x[i + 1, j]) +
                                2 * G12 * (-x[i + 1, j + 1] + x[i + 1, j]))) *
                               ((x[i + 1, j + 1] - x[i, j]) ** 2 + (y[i + 1, j + 1] - y[i, j]) ** 2)) / \
                              (x[i + 1, j] * (-y[i + 1, j + 1] + y[i, j]) + x[i, j] * (y[i + 1, j + 1] - y[i + 1, j]) + x[
                                  i + 1, j] * (-y[i, j] + y[i + 1, j])) ** 3
                        Fxy = -((2 * Dk * ((x[i + 1, j + 1] - x[i, j]) ** 2 + (y[i + 1, j + 1] - y[i, j]) ** 2) *
                                 (G11 * (x[i + 1, j + 1] - x[i + 1, j]) * (y[i + 1, j + 1] - y[i + 1, j]) +
                                  G22 * (x[i, j] - x[i + 1, j]) * (y[i, j] - y[i + 1, j]) +
                                  G12 * (-x[i, j] * y[i + 1, j + 1] + x[i + 1, j] * y[i + 1, j + 1] - x[i + 1, j + 1] * y[
                                             i, j] +
                                         x[i + 1, j] * y[i, j] + (x[i + 1, j + 1] + x[i, j] - 2 * x[i + 1, j]) * y[
                                             i + 1, j]))) / (
                                        x[i + 1, j] * (-y[i + 1, j + 1] + y[i, j]) + x[i, j] * (
                                            y[i + 1, j + 1] - y[i + 1, j]) +
                                        x[
                                            i + 1, j] * (-y[i, j] + y[i + 1, j])) ** 3)
                        Rx[i + 1, j] += Fx / weight
                        Rxy[i + 1, j] += Fxy / weight
                        Ry[i + 1, j] += Fy / weight
                        Rxx[i + 1, j] += Fxx / weight
                        Ryy[i + 1, j] += Fyy / weight

                        # xk+1 : x[i, j]
                        Fx = Dk * (-G11 * ((x[i + 1, j + 1] - x[i + 1, j]) ** 2 + (y[i + 1, j + 1] - y[i + 1, j]) ** 2) * (
                                y[i + 1, j + 1] - y[i + 1, j]) +
                                   2 * G12 * ((x[i + 1, j + 1] - x[i + 1, j]) ** 2 + (
                                            y[i + 1, j + 1] - y[i + 1, j]) ** 2) * (
                                           y[i, j] - y[i + 1, j]) +
                                   G22 * (x[i, j] ** 2 * (y[i + 1, j + 1] - y[i + 1, j]) +
                                          2 * x[i + 1, j + 1] * x[i + 1, j] * (y[i, j] - y[i + 1, j]) -
                                          (y[i + 1, j + 1] - y[i + 1, j]) * (y[i, j] - y[i + 1, j]) ** 2 +
                                          x[i + 1, j] ** 2 * (y[i + 1, j + 1] - 2 * y[i, j] + y[i + 1, j]) +
                                          2 * x[i, j] * (x[i + 1, j] * (-y[i + 1, j + 1] + y[i, j]) + x[i + 1, j + 1] * (
                                                -y[i, j] + y[i + 1, j])))) / \
                             (x[i + 1, j] * (-y[i + 1, j + 1] + y[i, j]) + x[i, j] * (y[i + 1, j + 1] - y[i + 1, j]) + x[
                                 i + 1, j] * (-y[i, j] + y[i + 1, j])) ** 2
                        Fy = Dk * (
                                (-(-x[i + 1, j + 1] + x[i + 1, j]) * (
                                        G11 * ((x[i + 1, j + 1] - x[i + 1, j]) ** 2 + (y[i + 1, j + 1] - y[i + 1, j]) ** 2)
                                        - 2 * G12 * ((x[i + 1, j + 1] - x[i + 1, j]) * (x[i, j] - x[i + 1, j]) + (
                                        y[i + 1, j + 1] - y[i + 1, j]) * (y[i, j] - y[i + 1, j]))
                                        + G22 * ((x[i, j] - x[i + 1, j]) ** 2 + (y[i, j] - y[i + 1, j]) ** 2)))
                                + 2 * (x[i + 1, j] * (y[i + 1, j + 1] - y[i, j]) + x[i + 1, j + 1] * (
                                    y[i, j] - y[i + 1, j]) + x[
                                           i, j + 1] * (-y[i + 1, j + 1] + y[i + 1, j]))
                                * (G12 * (y[i + 1, j + 1] - y[i + 1, j]) + G22 * (-y[i, j] + y[i + 1, j]))
                        ) / (x[i + 1, j] * (-y[i + 1, j + 1] + y[i, j]) + x[i, j] * (y[i + 1, j + 1] - y[i + 1, j]) + x[
                            i + 1, j] * (-y[i, j] + y[i + 1, j])) ** 2
                        Fxx = 2 * Dk * ((x[i + 1, j + 1] - x[i + 1, j]) ** 2 + (y[i + 1, j + 1] - y[i + 1, j]) ** 2) * (
                                G11 * (y[i + 1, j + 1] - y[i + 1, j]) ** 2 +
                                (y[i, j] - y[i + 1, j]) * (
                                        G22 * (y[i, j] - y[i + 1, j]) + 2 * G12 * (-y[i + 1, j + 1] + y[i + 1, j]))
                        ) / (x[i + 1, j] * (-y[i + 1, j + 1] + y[i, j]) + x[i, j] * (y[i + 1, j + 1] - y[i + 1, j]) + x[
                            i + 1, j] * (-y[i, j] + y[i + 1, j])) ** 3
                        Fyy = (2 * Dk * (G11 * (x[i + 1, j + 1] - x[i + 1, j]) ** 2 +
                                         (x[i, j] - x[i + 1, j]) * (
                                                 G22 * (x[i, j] - x[i + 1, j]) + 2 * G12 * (
                                                     -x[i + 1, j + 1] + x[i + 1, j]))) *
                               ((x[i + 1, j + 1] - x[i + 1, j]) ** 2 + (y[i + 1, j + 1] - y[i + 1, j]) ** 2)) / \
                              (x[i + 1, j] * (-y[i + 1, j + 1] + y[i, j]) + x[i, j] * (y[i + 1, j + 1] - y[i + 1, j]) + x[
                                  i + 1, j] * (-y[i, j] + y[i + 1, j])) ** 3
                        Fxy = (-2 * Dk * ((x[i + 1, j + 1] - x[i + 1, j]) ** 2 + (y[i + 1, j + 1] - y[i + 1, j]) ** 2) *
                               (G11 * (x[i + 1, j + 1] - x[i + 1, j]) * (y[i + 1, j + 1] - y[i + 1, j]) +
                                G22 * (x[i, j] - x[i + 1, j]) * (y[i, j] - y[i + 1, j]) +
                                G12 * (-x[i, j] * y[i + 1, j + 1] + x[i + 1, j] * y[i + 1, j + 1] - x[i + 1, j + 1] * y[
                                           i, j] + x[
                                           i, j] * y[i, j] +
                                       (x[i + 1, j + 1] + x[i, j] - 2 * x[i + 1, j]) * y[i + 1, j])) *
                               (x[i + 1, j] * (-y[i + 1, j + 1] + y[i, j]) + x[i, j] * (y[i + 1, j + 1] - y[i + 1, j]) + x[
                                   i + 1, j] * (-y[i, j] + y[i + 1, j])) ** -3)
                        Rx[i, j] += Fx / weight
                        Rxy[i, j] += Fxy / weight
                        Ry[i, j] += Fy / weight
                        Rxx[i, j] += Fxx / weight
                        Ryy[i, j] += Fyy / weight

                        # xk-1 : x[i + 1, j + 1]
                        Fx = (Dk * (-G11 * ((x[i + 1, j + 1] - x[i + 1, j]) ** 2 + (y[i + 1, j + 1] - y[i + 1, j]) ** 2) * (
                                y[i + 1, j + 1] - y[i + 1, j]) +
                                    2 * G12 * ((x[i + 1, j + 1] - x[i + 1, j]) ** 2 + (
                                            y[i + 1, j + 1] - y[i + 1, j]) ** 2) * (
                                            y[i, j] - y[i + 1, j]) +
                                    G22 * (x[i, j] ** 2 * (y[i + 1, j + 1] - y[i + 1, j]) +
                                           2 * x[i + 1, j + 1] * x[i + 1, j] * (y[i, j] - y[i + 1, j]) -
                                           (y[i + 1, j + 1] - y[i + 1, j]) * (y[i, j] - y[i + 1, j]) ** 2 +
                                           x[i + 1, j] ** 2 * (y[i + 1, j + 1] - 2 * y[i, j] + y[i + 1, j]) +
                                           2 * x[i, j] * (x[i + 1, j] * (-y[i + 1, j + 1] + y[i, j]) +
                                                          x[i + 1, j + 1] * (-y[i, j] + y[i + 1, j]))))) / \
                             (x[i + 1, j] * (-y[i + 1, j + 1] + y[i, j]) + x[i, j] * (y[i + 1, j + 1] - y[i + 1, j]) + x[
                                 i + 1, j] * (-y[i, j] + y[i + 1, j])) ** 2
                        Fy = (Dk * (-(-x[i + 1, j + 1] + x[i + 1, j]) * (
                                G11 * ((x[i + 1, j + 1] - x[i + 1, j]) ** 2 + (y[i + 1, j + 1] - y[i + 1, j]) ** 2) -
                                2 * G12 * ((x[i + 1, j + 1] - x[i + 1, j]) * (x[i, j] - x[i + 1, j]) + (
                                y[i + 1, j + 1] - y[i + 1, j]) * (
                                                   y[i, j] - y[i + 1, j])) +
                                G22 * ((x[i, j] - x[i + 1, j]) ** 2 + (y[i, j] - y[i + 1, j]) ** 2)) +
                                    2 * (x[i + 1, j] * (y[i + 1, j + 1] - y[i, j]) + x[i + 1, j + 1] * (
                                            y[i, j] - y[i + 1, j]) + x[
                                             i, j + 1] * (-y[i + 1, j + 1] + y[i + 1, j])) *
                                    (G12 * (y[i + 1, j + 1] - y[i + 1, j]) + G22 * (-y[i, j] + y[i + 1, j])))) / \
                             (x[i + 1, j] * (-y[i + 1, j + 1] + y[i, j]) + x[i, j] * (y[i + 1, j + 1] - y[i + 1, j]) + x[
                                 i + 1, j] * (-y[i, j] + y[i + 1, j])) ** 2
                        Fxx = (2 * Dk * ((x[i + 1, j + 1] - x[i + 1, j]) ** 2 + (y[i + 1, j + 1] - y[i + 1, j]) ** 2) *
                               (G11 * (y[i + 1, j + 1] - y[i + 1, j]) ** 2 + (y[i, j] - y[i + 1, j]) * (
                                       G22 * (y[i, j] - y[i + 1, j]) +
                                       2 * G12 * (-y[i + 1, j + 1] + y[i + 1, j])))) / \
                              (x[i + 1, j] * (-y[i + 1, j + 1] + y[i, j]) + x[i, j] * (y[i + 1, j + 1] - y[i + 1, j]) + x[
                                  i + 1, j] * (-y[i, j] + y[i + 1, j])) ** 3
                        Fyy = (2 * Dk * (G11 * (x[i + 1, j + 1] - x[i + 1, j]) ** 2 +
                                         (x[i, j] - x[i + 1, j]) * (
                                                 G22 * (x[i, j] - x[i + 1, j]) + 2 * G12 * (
                                                     -x[i + 1, j + 1] + x[i + 1, j]))) *
                               ((x[i + 1, j + 1] - x[i + 1, j]) ** 2 + (y[i + 1, j + 1] - y[i + 1, j]) ** 2)) / \
                              (x[i + 1, j] * (-y[i + 1, j + 1] + y[i, j]) + x[i, j] * (y[i + 1, j + 1] - y[i + 1, j]) + x[
                                  i + 1, j] * (-y[i, j] + y[i + 1, j])) ** 3
                        Fxy = -((2 * Dk * ((x[i + 1, j + 1] - x[i + 1, j]) ** 2 + (y[i + 1, j + 1] - y[i + 1, j]) ** 2) *
                                 (G11 * (x[i + 1, j + 1] - x[i + 1, j]) * (y[i + 1, j + 1] - y[i + 1, j]) +
                                  G22 * (x[i, j] - x[i + 1, j]) * (y[i, j] - y[i + 1, j]) +
                                  G12 * (-x[i, j] * y[i + 1, j + 1] + x[i + 1, j] * y[i + 1, j + 1] - x[i + 1, j + 1] * y[
                                             i, j] +
                                         x[i + 1, j] * y[i, j] + (x[i + 1, j + 1] + x[i, j] - 2 * x[i + 1, j]) * y[
                                             i + 1, j])))) / \
                              (x[i + 1, j] * (-y[i + 1, j + 1] + y[i, j]) + x[i, j] * (y[i + 1, j + 1] - y[i + 1, j]) + x[
                                  i + 1, j] * (-y[i, j] + y[i + 1, j])) ** 3
                        Rx[i + 1, j + 1] += Fx / weight
                        Rxy[i + 1, j + 1] += Fxy / weight
                        Ry[i + 1, j + 1] += Fy / weight
                        Rxx[i + 1, j + 1] += Fxx / weight
                        Ryy[i + 1, j + 1] += Fyy / weight

    if flag != 1:
        print("aproxim: ", aproximation/((nx-1)*(ny-1)*8))
    return aproximation/((nx-1)*(ny-1)*8), Rx, Ry, Rxx, Ryy, Rxy, flag


def mimimize(omega, canon, grid_shape, tau, eps, count=0):
    nx, ny = grid_shape
    Fk, Rx, Ry, Rxx, Ryy, Rxy, flag = functional2(omega, canon, grid_shape)
    x_old = omega[0]
    y_old = omega[1]
    if count >= 500:
        return np.array([x_old, y_old])
    x_new = x_old.copy()
    y_new = y_old.copy()
    maxdif = 0.0
    for i in range(2, nx-1):
        for j in range(2, ny-1):
            x_new[i, j] = x_old[i, j] - tau * (Rx[i, j] * Ryy[i, j] - Ry[i, j] * Ryy[i, j]) / (
                (Rxx[i, j] * Ryy[i, j] - Rxy[i, j] * Rxy[i, j]))
            y_new[i, j] = y_old[i, j] - tau * (Rx[i, j] * Ryy[i, j] - Ry[i, j] * Ryy[i, j]) / (
                (Rxx[i, j] * Ryy[i, j] - Rxy[i, j] * Rxy[i, j]))
            if abs(x_new[i, j] - x_old[i, j]) > maxdif:
                maxdif = abs(x_new[i, j] - x_old[i, j])
            if abs(y_new[i, j] - y_old[i, j]) > maxdif:
                maxdif = abs(y_new[i, j] - y_old[i, j])
    new_grid = np.array([x_new, y_new])
    flag = functional2(new_grid, canon, (nx, ny))[6]
    if flag == 1:
        tau_new = tau/2
        count += 1
        return mimimize(np.array([x_old, y_old]), canon, grid_shape, tau_new, eps, count)
    else:
        if maxdif <= eps:
            return np.array([x_old, y_old])
        else:
            count += 1
            return mimimize(np.array([x_new, y_new]), canon, grid_shape, tau, eps, count)


def hooke_jeeves(omega, canon, shape, step_size, tolerance):
    nx, ny = shape
    best_value = 0
    step = step_size
    for global_steps in range(20):
        for m in range(1, ny-1):
            for n in range(1, nx-1):
                current_step = step
                current_x = omega[0][m, n]
                current_y = omega[1][m, n]
                print("x: ", current_x, "y: ", current_y)
                best_x = current_x
                best_y = current_y
                current_value = functional2(omega, canon, shape)[0]
                best_value = current_value

                while current_step > tolerance:
                    flag = False
                    for i, (dx, dy) in enumerate([(current_step, 0), (-current_step, 0), (0, current_step), (0, -current_step)]):
                        new_x_pos = current_x + dx
                        new_y_pos = current_y + dy
                        omega_test = omega.copy()
                        omega_test[0][m, n] = new_x_pos
                        omega_test[1][m, n] = new_y_pos
                        new_value_pos = functional2(omega_test, canon, shape)[0]
                        smooth_flag = functional2(omega_test, canon, shape)[6]

                        if (new_value_pos < best_value) and (smooth_flag != 1):
                            best_value = new_value_pos
                            best_x = new_x_pos
                            best_y = new_y_pos
                            flag = True

                    if not flag:
                        current_step *= current_step - (current_step // 3)
                    else:
                        current_x = best_x
                        current_y = best_y

                omega[0][m, n] = current_x
                omega[1][m, n] = current_y

    return omega, best_value


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
treshhold = 1e-30
"""
test, count = winslow_without_implicit(51, Xyt, Xyb, Xyr, Xyl, treshhold)   # Уинслоу + Томпсон + явная TFI
print("кол-во итераций: ", count)
plt.plot(test[0], test[1], c="b", linewidth=1)
plt.plot(np.transpose(test[0]), np.transpose(test[1]), c="b", linewidth=1)
plt.show()

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


node2 = [11, 11, 0.6, 0.5]

nx, ny = 10, 10
# canon_grid = implicit_transfinite_interpol(nx, sq_t1, sq_b1, sq_l1, sq_r1)
# canon_grid, count = winslow_without_implicit(NX, Xyt, Xyb, Xyr, Xyl, treshhold)
# canon_grid = transfinite_interpol(nx, Xyt, Xyb, Xyl, Xyr)
# canon_grid = implicit_transfinite_interpol(nx, chevt, chevb, chevr, chevl)
# canon_grid = implicit_transfinite_interpol(nx, myt, myb, myr, myl)
canon_grid = implicit_transfinite_interpol(nx, sq_t1, sq_b1, sq_l1, sq_r1)
canon_for_min = canon_grid

for i in range(nx):
    for j in range(ny):
        """if i == (nx//2) - 1:
            canon_grid[0][i][j] += 0.04
        if j == (ny//2) - 1:
            canon_grid[1][i][j] += 0.04
        if i == (nx//2) - 1 and i != 0 and j != 0 and i != nx-1 and j != nx-1:
            canon_grid[0][i][j] -= 0.04
        if j == (ny//2) - 1 and i != 0 and j != 0 and i != nx-1 and j != nx-1:
            canon_grid[1][i][j] -= 0.04"""

"""        if i == 2:
            canon_grid[0][i][j] += 0.06
        if j == 2:
            canon_grid[1][i][j] += 0.06
        if i == nx - 5:
            canon_grid[0][i][j] -= 0.06
        if j == ny - 5:     # and i != 0 and j != 0 and i != nx-1 and j != nx-1
            canon_grid[1][i][j] -= 0.06"""
"""for i in range(nx-1, -1, -1):
    for j in range(ny):
        if j != 0 and j != ny-1:
            canon_grid[1][i][j] = canon_grid[1][i][j]**1.5 
for i in range(nx-1, -1, -1):
    for j in range(ny):
        if i != 0 and i != ny-1:
            canon_grid[0][i][j] = canon_grid[0][i][j]**1.5"""

canon_grid[0] = np.flip(canon_grid[0], axis=0)
canon_grid[1] = np.flip(canon_grid[1], axis=0)
canon_grid[0] = np.transpose(canon_grid[0])
canon_grid[1] = np.transpose(canon_grid[1])

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

plt.plot(-np.transpose(canon_grid[1]), np.transpose(canon_grid[0]), c="b", linewidth=1)
plt.plot(-canon_grid[1], canon_grid[0], c="b", linewidth=1)
plt.show()

"""print(xx)
print("x ", xx[0, 0], xx[nx-1, ny-1])
print("y ", yy[0, 0], yy[nx-1, ny-1])"""


# omega_grid2, count = winslow_without_implicit(nx, Xyt, Xyb, Xyr, Xyl, treshhold)
# omega_grid2 = implicit_transfinite_interpol(nx, myt, myb, myr, myl)
# omega_grid2 = transfinite_interpol(nx, Xyt, Xyb, Xyl, Xyr)
# omega_grid2 = implicit_transfinite_interpol(nx, chevt, chevb, chevr, chevl)
# omega_grid2 = transfinite_interpol(20, horst, horsb, horsr, horsl)
omega_grid2 = winslow_without_implicit(nx, Xyt, Xyb, Xyr, Xyl, treshhold)[0]
# omega_grid2 = implicit_transfinite_interpol(nx, sq_t1, sq_b1, sq_l1, sq_r1)
# print("\n")
# print(omega_grid2)
omega_grid2[0] = np.flip(omega_grid2[0], axis=0)
omega_grid2[1] = np.flip(omega_grid2[1], axis=0)
"""omega_grid2[0] = np.transpose(omega_grid2[0])
omega_grid2[1] = np.transpose(omega_grid2[1])"""
xx, yy = omega_grid2[0], omega_grid2[1]
omega_grid = (xx, yy)
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

plt.plot(omega_grid2[0], omega_grid2[1], c="b", linewidth=1)
plt.plot(np.transpose(omega_grid2[0]), np.transpose(omega_grid2[1]), c="b", linewidth=1)
plt.show()
"""result = var_metr_method(omega_grid, canon_grid, nx, ny)
plot_grid(result, nx, ny, 'Optimized Grid')"""
"""flag = functional2(omega_grid, canon_grid, grid_shape=(nx, ny))[6]
print("flag = ", flag)"""

### МЕЙН ВЫЗОВ ###
"""new_grid = mimimize(omega_grid, canon_grid, (nx, ny), 1, 10**(-8))"""
"""new_grid, value = hooke_jeeves(omega_grid2, canon_grid, (nx, ny), 0.2, 0.001)"""
new_grid = mimimize(omega_grid2, canon_grid, (nx, ny), 0.1, 0.0000001)
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

test, count = winslow_without_implicit(51, Xyt, Xyb, Xyr, Xyl, treshhold)   # Уинслоу + Томпсон + явная TFI
print("кол-во итераций: ", count)
plt.plot(test[0], test[1], c="b", linewidth=1)
plt.plot(np.transpose(test[0]), np.transpose(test[1]), c="b", linewidth=1)
#plt.show()"""
