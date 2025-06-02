import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree
import scipy.linalg
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from numba import njit
import time
# Параметры материала (Алюминий ГЦК)
E = 70e9  # Модуль Юнга, Па
nu = 0.33  # Коэффициент Пуассона
rho = 2700  # Плотность, кг/м^3
C11 = 108e9  # Па
C12 = 62e9  # Па
C44 = 28e9  # Па

tau_c = 20e6  # Начальное напряжение текучести, Па
gamma_0 = 1e-8  # Начальная скорость сдвига
n = 1/0.0012  # Показатель степени в законе течения

b_vectors = np.array([
    [-1/np.sqrt(2), 1/np.sqrt(2), 0],
    [1/np.sqrt(2), 0, -1/np.sqrt(2)],
    [0, -1/np.sqrt(2), 1/np.sqrt(2)],

    [1/np.sqrt(2), 0, 1/np.sqrt(2)],
    [-1/np.sqrt(2), -1/np.sqrt(2), 0],
    [0, 1/np.sqrt(2), -1/np.sqrt(2)],

    [-1/np.sqrt(2), 0, 1/np.sqrt(2)],
    [0, -1/np.sqrt(2), -1/np.sqrt(2)],
    [1/np.sqrt(2), 1/np.sqrt(2), 0],

    [1/np.sqrt(2), -1/np.sqrt(2), 0],
    [-1/np.sqrt(2), 0, -1/np.sqrt(2)],
    [0, 1/np.sqrt(2), 1/np.sqrt(2)]
])


# Векторы нормали скольжения (n1, n2, n3) для каждой системы скольжения
n_vectors = np.array([
    [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)],
    [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)],
    [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)],

    [-1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)],
    [-1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)],
    [-1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)],

    [1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)],
    [1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)],
    [1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)],

    [-1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)],
    [-1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)],
    [-1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)]
])


C = np.zeros((3, 3, 3, 3))
for i in range(3):
    for j in range(3):
        if i == j:
            C[i, i, i, i] = C11
            for k in range(3):
                if k != i:
                    C[i, i, k, k] = C12
        else:
            C[i, j, i, j] = C44


def weight_func(xi_norm, horizon=3):
    """
    Функция веса для модели.

    Parameters:
        xi_norm (float): Норма вектора ξ = x' - x.
        horizon (float): Радиус взаимодействия.
        delta (float): Параметр формы (по умолчанию 1.0).

    Returns:
        float: Весовой коэффициент.
    """
    if xi_norm > horizon:
        return 0.0
    return np.exp(-(xi_norm) ** 2)

def find_neighbors(points, horizon):
    """функция нахождения соседей частицы"""
    tree = KDTree(points)
    neighbors_dict = {}
    for i, point in enumerate(points):
        indices = tree.query_ball_point(point, horizon)
        indices.remove(i)
        neighbors_dict[i] = indices
    return neighbors_dict



def compute_shape_tensor(particles, neighbors_dict, horizon):
    """функция вычисления тензора формы в дискретной форме(16)"""
    for i, p in enumerate(particles):
        K_i = np.zeros((3, 3))
        for j in neighbors_dict[i]:
            xi_j = particles[j].x_ref - p.x_ref
            wij = weight_func(np.linalg.norm(xi_j))
            K_i += wij * np.outer(xi_j, xi_j) * (horizon ** 3)
        p.K_tensor = K_i

def compute_deformation_gradient(particles, neighbors_dict,horizon):
    """функция вычисления градиента деформации в дискретной форме(16)"""
    for i, particle in enumerate(particles):
        F_i = np.zeros((3, 3))
        xi, yi = particle.x_ref, particle.x_curr

        for j in neighbors_dict[i]:
            dy = particles[j].x_curr - yi
            xi_j = particles[j].x_ref - xi
            wij = weight_func(np.linalg.norm(xi_j))
            F_i += wij * np.outer(dy, xi_j)
        """if np.linalg.det(Crystal.K_tensor) > 1e-10:
            particle.F = F_i @ np.linalg.inv(particle.K)
        else:
            particle.F = np.eye(3)  # Резервный вариант"""
        particle.F = (F_i * horizon ** 3) @ np.linalg.inv(particle.K_tensor)


@njit
def compute_force_state_numba(P, K_inv, x_ref, neighbor_x_refs, horizon):
    T = np.zeros(3)
    for j in range(len(neighbor_x_refs)):
        xi = neighbor_x_refs[j] - x_ref
        xi_norm = np.sqrt(xi[0]**2 + xi[1]**2 + xi[2]**2)
        wij = np.exp(-(xi_norm)**2)  # Простая weight_func
        T += wij * (P @ K_inv @ xi)
    return T


def compute_forces(particles, neighbors_dict, size):
    for i, p in enumerate(particles):
        L = np.zeros(3)
        for j in neighbors_dict[i]:
            # Берем force_state из соседних частиц
            L += (p.force_state - particles[j].force_state) * (size ** 3)
        p.force = L


def compute_local_stiffness_matrix(particle, neighbors, particles):
    """Строит локальную матрицу жёсткости для частицы и её соседей."""
    n_neighbors = len(neighbors[particle.idx])
    dim = 3  # 3D
    K_local = np.zeros((dim * (n_neighbors + 1), dim * (n_neighbors + 1)))  # +1 для самой частицы

    # Заполняем диагональные блоки (сама частица)
    for d in range(dim):
        du = particle.u_curr[d] - particle.u_prev[d]
        df = particle.force[d] - particle.force_prev[d]

        K_local[d][d] = -df / du

    # Заполняем блоки для соседей (упрощённо — только диагональные элементы)
    for idx, neighbor_id in enumerate(neighbors[particle.idx]):
        neighbor = particles[neighbor_id]
        for d in range(dim):
            du = neighbor.u_curr[d] - neighbor.u_prev[d]
            df = neighbor.force[d] - neighbor.force_prev[d]

            K_local[dim * (idx + 1) + d][dim * (idx + 1) + d] = -df / du

    return K_local


def compute_damping_coefficient(particle, neighbors, particles):
    """Вычисляет c для частицы на основе локальной матрицы жёсткости."""
    K_local = compute_local_stiffness_matrix(particle, neighbors, particles)

    # Собираем вектор перемещений (частица + соседи)
    u_local = np.zeros(3 * (len(neighbors[particle.idx]) + 1))
    u_local[:3] = particle.u_curr  # Перемещение самой частицы

    for idx, neighbor_id in enumerate(neighbors[particle.idx]):
        u_local[3 * (idx + 1): 3 * (idx + 1) + 3] = particles[neighbor_id].u_curr

    numerator = abs(u_local.T @ K_local @ u_local)
    denominator = abs(u_local.T @ u_local)

    return 2 * np.sqrt(numerator / denominator)

@njit
def rotate_elastic_tensor(C, R):
    """Поворот тензора упругости 4-го ранга без np.einsum"""
    # Инициализируем повернутый тензор
    C_rotated = np.zeros_like(C)
    dim = C.shape[0]  # Размерность тензора (обычно 3 для 3D)

    # Вложенные циклы для всех индексов
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for l in range(dim):
                    for p in range(dim):
                        for q in range(dim):
                            for r in range(dim):
                                for s in range(dim):
                                    C_rotated[i, j, k, l] += (R[p, i] * R[q, j] * R[r, k] * R[s, l] * C[p, q, r, s])
    return C_rotated


"""@njit
def compute_piola_kirchhoff_stress_numba(F, F_p, C11, C12, C44):
    F_e = F @ np.linalg.inv(F_p)
    E_e = 0.5 * (F_e.T @ F_e - np.eye(3))

    # Упрощенный анизотропный тензор для ГЦК (Al)
    S = np.zeros((3, 3))
    S[0, 0] = C11 * E_e[0, 0] + C12 * (E_e[1, 1] + E_e[2, 2])
    S[1, 1] = C11 * E_e[1, 1] + C12 * (E_e[0, 0] + E_e[2, 2])
    S[2, 2] = C11 * E_e[2, 2] + C12 * (E_e[0, 0] + E_e[1, 1])
    S[0, 1] = S[1, 0] = 2 * C44 * E_e[0, 1]
    S[0, 2] = S[2, 0] = 2 * C44 * E_e[0, 2]
    S[1, 2] = S[2, 1] = 2 * C44 * E_e[1, 2]
    return F_e @ S"""
@njit
def compute_piola_kirchhoff_stress_numba(F, F_p, C):
    F_e = F @ np.linalg.inv(F_p)
    E_e = 0.5 * (F_e.T @ F_e - np.eye(3))
    # Упрощенный анизотропный тензор для ГЦК (Al)
    S = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    S[i, j] += C[i, j, k, l] * E_e[k, l]
    return F_e @ S

"""def compute_piola_kirchhoff_stress(particle):
    #Вычисляем первый тензор Пиола-Кирхгофа через анизотропную упругость
    F_e = particle.F @ np.linalg.inv(particle.F_p)

    # 2. Тензор упругих деформаций Грина-Лагранжа
    E_e = 0.5 * (F_e.T @ F_e - np.eye(3))

    S = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    S[i, j] += C[i, j, k, l] * E_e[k, l]

    # 4. Первый тензор Пиола-Кирхгофа
    P = F_e @ S
    particle.stress = P
    return P"""

def u_velocity(dt, u_next, u_prev):
    if u_next.shape != (3,) or u_prev.shape != (3,):
        print(f"ОШИБКА РАЗМЕРНОСТИ: u_next={u_next.shape}, u_prev={u_prev.shape}")
        print(f"Значения: u_next={u_next}, u_prev={u_prev}")
        raise ValueError("Некорректная размерность векторов")
    return (u_next - u_prev) / 2 * dt



def u_acceleration(dt, u_next, u_curr, u_prev):
    if u_next.shape != (3,) or u_curr.shape != (3,) or u_prev.shape != (3,):
        print(f"ОШИБКА РАЗМЕРНОСТИ: u_next={u_next.shape}, u_curr={u_curr.shape}, u_prev={u_prev.shape}")
        print(f"Значения: u_next={u_next}, u_curr={u_curr}, u_prev={u_prev}")
        raise ValueError("Некорректная размерность векторов")
    return (u_next - 2 * u_curr + u_prev) / (dt ** 2)



def apply_boundary_conditions(particles, bounds, horizon):
    for p in particles:
        # Закрепляем частицы в нижнем слое (z=0)
        if p.x_ref[2] <= bounds['zmin']:
            p.x_curr = p.x_ref.copy()
            p.velocity = np.zeros(3)
            p.u_curr = np.zeros(3)  # Обновляем u_curr
        # Прикладываем силу к верхнему слою (z=max)
        elif p.x_ref[2] >= bounds['zmax']:
            p.u_curr += np.array([-1e-20, -1e-20, -1e-10])  # Минимальное начальное смещение
            p.x_curr = p.x_ref + p.u_curr  # Обновляем x_curr
        elif p.x_ref[2] < bounds['zmax'] and p.x_ref[2] > bounds['zmin']:
            p.u_curr = np.array([1e-16, 1e-16, 1e-16])  # Минимальное начальное смещение ЕСЛИ ЧТО КИРИЛЛ ТУТ ВРОДЕ КАК БЫЛ += А НЕ =
            p.x_curr = p.x_ref + p.u_curr  # Обновляем x_curr

    neighbors_dict = find_neighbors(points, horizon)
    return neighbors_dict


"""def compute_resolved_shear_stress(particle, system_idx):
    #Вычисляет напряжение сдвига для заданной системы скольжения
    S = np.outer(b_vectors[system_idx], n_vectors[system_idx])  # Тензор Шмида
    tau = np.sum(particle.stress * S)  # Скалярное произведение
    print(tau)
    return tau"""

"""def compute_shear_stress(particle, b_vectors, n_vectors):
    tau = np.zeros(len(b_vectors))
    for k in range(len(b_vectors)):
        S = np.outer(b_vectors[k], n_vectors[k])
        tau[k] = np.tensordot(particle.stress, S)
    return tau"""

"""def compute_shear_stress(particle, b_vectors, n_vectors):
    tau = np.zeros(len(b_vectors))

    for k in range(len(b_vectors)):
        S = np.outer(b_vectors[k], n_vectors[k])
        tau[k] = np.tensordot(particle.stress, S)
    return tau"""


@njit('float64[:](float64[:,:], float64[:,:], float64[:,:])')
def compute_shear_stress_numba(stress, b_rot, n_rot):
    tau = np.zeros(len(b_rot))
    for k in range(len(b_rot)):
        S = np.outer(b_rot[k], n_rot[k])  # Внешнее произведение
        tau[k] = np.sum(stress * S)       # Аналог tensordot
    return tau



"""def update_plastic_deformation(particle, dt):
    #n можно задавать для плавности перехода
    L_p = np.zeros((3, 3))
    for i in range(len(b_vectors)):
        tau = compute_resolved_shear_stress(particle, i)
        if abs(tau) > tau_c:
            d_gamma = gamma_0 ((abs(tau) / tau_c)**83) * dt
            L_p += d_gamma * np.outer(b_vectors[i], n_vectors[i])
    # Экспоненциальное обновление F_p
    particle.F_p = scipy.linalg.expm(L_p * dt) @ particle.F_p"""


@njit
def compute_L_p_numba(stress, b_vec, n_vec, tau_c, gamma_0, n):
    tau = compute_shear_stress_numba(stress, b_vec, n_vec)
    L_p = np.zeros((3, 3))
    for k in range(len(tau)):
        if abs(tau[k]) > tau_c:
            d_gamma = gamma_0 * (abs(tau[k]) / tau_c)**n * np.sign(tau[k])
            L_p += d_gamma * np.outer(b_vec[k], n_vec[k])
    return L_p


class Particle:
    def __init__(self, idx, x_ref, material,rotation_angle):
        self.idx = idx               # Индекс частицы
        self.x_ref = x_ref           # Референсные координаты
        self.x_curr = x_ref.copy()   # Текущие координаты (y = x + u)
        self.displacement = np.zeros(3)
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.force = np.zeros(3)
        self.force_prev = np.zeros(3)
        self.F = np.eye(3)           # Градиент деформации
        self.K_tensor = np.zeros((3, 3))
        self.F_p = np.eye(3)         # Пластическая часть F
        self.stress = np.zeros((3,3))
        self.material = material     # Объект Material
        self.u_curr = np.zeros(3)
        self.u_ref = np.zeros(3)
        self.shear_stress = np.zeros(12)
        self.rotation_angle = rotation_angle  # Угол поворота
        self.R = self._build_rotation_matrix()  # Матрица поворота 3x3
        self.force_state = np.zeros(3)

    def _build_rotation_matrix(self):
        """Создает матрицу поворота вокруг оси Z на заданный угол"""
        theta = np.radians(self.rotation_angle)
        return np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

    def get_rotated_slip_systems(self):
        """Возвращает повернутые системы скольжения (b_rot, n_rot)"""
        # Глобальные системы скольжения (ваши b_vectors и n_vectors)
        b_global = np.array([
            [-1 / np.sqrt(2), 1 / np.sqrt(2), 0],
            [1 / np.sqrt(2), 0, -1 / np.sqrt(2)],
            [0, -1 / np.sqrt(2), 1 / np.sqrt(2)],

            [1 / np.sqrt(2), 0, 1 / np.sqrt(2)],
            [-1 / np.sqrt(2), -1 / np.sqrt(2), 0],
            [0, 1 / np.sqrt(2), -1 / np.sqrt(2)],

            [-1 / np.sqrt(2), 0, 1 / np.sqrt(2)],
            [0, -1 / np.sqrt(2), -1 / np.sqrt(2)],
            [1 / np.sqrt(2), 1 / np.sqrt(2), 0],

            [1 / np.sqrt(2), -1 / np.sqrt(2), 0],
            [-1 / np.sqrt(2), 0, -1 / np.sqrt(2)],
            [0, 1 / np.sqrt(2), 1 / np.sqrt(2)]
        ])
        n_global = np.array([
            [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
            [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
            [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],

            [-1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
            [-1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
            [-1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],

            [1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3)],
            [1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3)],
            [1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3)],

            [-1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3)],
            [-1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3)],
            [-1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3)]
        ])

        # Поворот каждой системы
        b_rotated = (self.R @ b_global.T).T
        n_rotated = (self.R @ n_global.T).T
        return b_rotated, n_rotated


class Crystallite:
    def __init__(self, center, rotation_angle):
        self.center = center
        self.rotation_angle = rotation_angle


class Crystal:
    def __init__(self, size, num_crystallites, rotation_range):
        self.size = size
        self.num_crystallites = num_crystallites
        self.rotation_range = rotation_range
        self.crystallites = []
        self.create_crystallites()


    def create_crystallites(self):
        np.random.seed(23)
        centers = np.random.uniform(0, self.size, (self.num_crystallites, 3))
        rotation_angles = np.random.uniform(*self.rotation_range, self.num_crystallites)
        self.crystallites = [Crystallite(centers[i], rotation_angles[i]) for i in range(self.num_crystallites)]

    def get_rotation_angles(self, points):
        crystallite_labels = np.argmin(
            np.linalg.norm(points[:, None] - [c.center for c in self.crystallites], axis=2), axis=1)
        return np.array([self.crystallites[label].rotation_angle for label in crystallite_labels])



# Параметры модели
size = 7
num_crystallites = 1
rotation_range = (-25, 25)

# Создание кристалла и сетки точек
crystal = Crystal(size, num_crystallites, rotation_range)
x, y, z = np.meshgrid(np.linspace(0, size-1, size), np.linspace(0, size-1, size), np.linspace(0, size-1, size), indexing='ij')
points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
misorientations = crystal.get_rotation_angles(points)
# Поиск соседей
h = np.linalg.norm(points[1] - points[0])  # Шаг сетки
horizon = 1 * h  # Радиус взаимодействия
particles = [Particle(idx=i, x_ref=points[i], material='Al', rotation_angle=misorientations[i])
             for i in range(len(points))]

bounds = {'zmin': 0, 'zmax': size-1}


def visualize_crystal_cube(particles, crystallites, misorientations):
    """
    Визуализирует 3D кубик, где цвет частиц зависит от угла разориентации с осью z.

    particles: np.ndarray (N, 3) - координаты частиц
    crystallites: np.ndarray (N,) - индекс кристаллита для каждой частицы
    misorientations: np.ndarray (M,) - углы разориентации кристаллитов (в градусах)
    """

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    norm = Normalize(vmin=-25, vmax=25)
    cmap = cm.jet

    colors = [cmap(norm(misorientations[p.idx])) for p in particles]

    x = np.array([p.x_ref[0] for p in particles])
    y = np.array([p.x_ref[1] for p in particles])
    z = np.array([p.x_ref[2] for p in particles])

    ax.scatter(x, y, z, c=colors, marker='o', s=5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Peridynamic Crystal Cube')

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label('Misorientation (degrees)')
    plt.show()


#visualize_crystal_cube(particles, crystal.crystallites, misorientations)
# NOVOE
sigma_u_values = []
epsilon_u_values = []
neighbors = apply_boundary_conditions(particles, bounds, horizon)


def visualize_boundary_conditions(particles, bounds, horizon):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Цвета для разных типов частиц:
    # - Обычные частицы: синий
    # - Закреплённые (нижняя грань): красный
    # - Нагружаемые (верхняя грань): зелёный
    colors = []
    for p in particles:
        if p.x_ref[2] <= bounds['zmin']:
            colors.append('red')  # Закреплённые
        elif p.x_ref[2] >= bounds['zmax']:
            colors.append('green')  # Нагружаемые
        else:
            colors.append('blue')  # Обычные

    x = np.array([p.x_ref[0] for p in particles])
    y = np.array([p.x_ref[1] for p in particles])
    z = np.array([p.x_ref[2] for p in particles])

    ax.scatter(x, y, z, c=colors, marker='o', s=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Граничные условия: красные - закреплены, зелёные - нагружаются')

    plt.show()
# visualize_boundary_conditions(particles, bounds, horizon)


def run_simulation(particles, neighbors, horizon):
    dt = (2 * h / len(particles)) * np.sqrt(rho / E)
    print(f"Time step: {dt}")
    n_steps = 1_0
    # Инициализация

    for p in particles:
        p.u_prev = np.zeros(3)
        p.u_curr = np.zeros(3)
        p.force_prev = np.zeros(3)

    compute_shape_tensor(particles, neighbors, horizon)
    nachalo = time.time()
    for step in range(n_steps):
        neighbors = apply_boundary_conditions(particles, bounds, horizon)
        #c = compute_damping_coefficient(p, neighbors, particles)
        # Обновление граничных условий
        #print(p.u_curr)
        """# 1. Обновляем пластическую деформацию
        for p in particles:
            b_rot, n_rot = p.get_rotated_slip_systems()
            L_p = compute_L_p_numba(p.stress, b_rot, b_rot, tau_c, gamma_0, n)
            p.F_p = (np.eye(3) + dt * L_p) @ p.F_p"""
        # Основной цикл ADRS
        while True:
            # 2. Вычисляем градиент деформации
            compute_deformation_gradient(particles, neighbors, horizon)
            # 1. Обновляем пластическую деформацию
            for p in particles:
                b_rot, n_rot = p.get_rotated_slip_systems()
                L_p = compute_L_p_numba(p.stress, b_rot, b_rot, tau_c, gamma_0, n)
                p.F_p = (np.eye(3) + dt * L_p) @ p.F_p
            """# 2. Вычисляем градиент деформации
            compute_deformation_gradient(particles, neighbors, horizon)"""
            # 3. Вычисляем force states для всех частиц
            for i, p in enumerate(particles):
                C_rot = rotate_elastic_tensor(C, p.R)
                P = compute_piola_kirchhoff_stress_numba(p.F, p.F_p, C_rot)
                K_inv = np.linalg.inv(p.K_tensor)
                neighbor_x_refs = np.array([particles[j].x_ref for j in neighbors[i]])
                p.force_state = compute_force_state_numba(P, K_inv, p.x_ref, neighbor_x_refs, horizon)
                p.stress = P
            #print(force_states)
            # 4. Уравнение (15): вычисляем L(x,t)
            """for p in particles:
                print("до", p.force.size)
                break"""
            compute_forces(particles, neighbors, horizon)
            """for p in particles:
                print("после", p.force.size)
                break"""
            # 5. Вычисляем коэффициент демпфирования
            c = compute_damping_coefficient(p, neighbors, particles)
            # 6. Обновляем перемещения
            sum_L2 = 0.0
            max_du = 0.0
            u_next = np.zeros(3)
            for p in particles:
                if p.x_ref[2] > bounds['zmin']:
                    # Сохраняем предыдущие значения
                    u_prev = p.u_prev.copy()
                    u_curr = p.u_curr.copy()
                    u_vel = u_velocity(dt, u_next, u_prev)
                    u_acc = u_acceleration(dt, u_next, u_curr, u_prev)
                    force = p.force.copy()

                    #print("force", force)
                    #print("u_curr", u_curr)
                    # ADRS update (уравнение 8)
                    """for i in range(len(u_prev)):
                        u_next[i] = (2 * dt ** 2 * force[i] + 4 * u_curr[i] + (c * dt - 2) * u_prev[i]) / (2 + c * dt)"""

                    u_next = (2 * dt ** 2 * force + 4 * u_curr + (c * dt - 2) * u_prev) / (2 + c * dt)
                    # Вычисляем ошибки
                    du = np.linalg.norm(u_next - u_curr)
                    sum_L2 += np.linalg.norm(p.force) ** 2
                    max_du += du ** 2

                    # Обновляем значения
                    p.u_prev = u_curr
                    p.u_curr = u_next
                    p.x_curr = p.x_ref + p.u_curr
            # Проверка сходимости
            e1 = np.sqrt(sum_L2) / len(particles)
            e2 = np.sqrt(max_du) / len(particles)
            print(f"Step {step}: e1={e1:.3e}, e2={e2:.3e}, dt={dt:.2e}, max(F_p)={max([np.max(p.F_p) for p in particles]):.2f}")

            if e1 < 1e-4 and e2 < 1e-4:
                break
        if step % 100 == 0 and step != 0:
            konec = time.time()
            print(f"Время выполнения: {konec - nachalo} секунд")
        stress_total = 0.0
        strain_total = 0.0
        count = 0
        for p in particles:
            s = np.sqrt(3 / 2 * np.tensordot(p.stress, p.stress.T, axes=2))
            Eps = 0.5 * (p.F.T @ p.F - np.eye(3))
            e = np.sqrt(2 / 3 * np.tensordot(Eps, Eps.T, axes=2))
            stress_total += s
            strain_total += e
            count += 1


        if count > 0:
            sigma_u_values.append(stress_total/1e6)
            epsilon_u_values.append(strain_total)


run_simulation(particles,neighbors,horizon)


def plot_deformation(particles):
    fig = plt.figure(figsize=(12, 6))

    # Вычисляем перемещения
    displacements = np.array([np.linalg.norm(p.x_curr - p.x_ref) for p in particles])

    # Начальная конфигурация
    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(*zip(*[p.x_ref for p in particles]), c='blue', s=5)
    ax1.set_title('Initial Configuration')

    # Деформированная конфигурация
    ax2 = fig.add_subplot(122, projection='3d')
    sc2 = ax2.scatter(*zip(*[p.x_curr for p in particles]), c=displacements, cmap='viridis', s=5)
    ax2.set_title('Deformed Configuration')
    plt.colorbar(sc2, ax=ax2, label='Displacement magnitude')

    plt.tight_layout()
    plt.show()


def plot_stress(particles):
    stresses = np.array([np.linalg.norm(p.stress) for p in particles])
    z_pos = np.array([p.x_curr[2] for p in particles])

    plt.figure(figsize=(12, 6))
    sc = plt.scatter(z_pos, stresses, c=stresses, cmap='jet',
                     vmin=0, vmax=5e10)  # Фиксированный диапазон
    plt.colorbar(sc, label='Stress (Pa)')
    plt.xlabel('Z Position', fontsize=12)
    plt.ylabel('Stress Magnitude (Pa)', fontsize=12)
    plt.title('Stress Distribution along Z-axis', fontsize=14)
    plt.grid(True)
    plt.show()


#plot_deformation(particles)
#plot_stress(particles)

def plot_pilot(epsilon_u_values,sigma_u_values):
    plt.figure(figsize=(10, 5))
    plt.plot(epsilon_u_values, sigma_u_values)
    plt.xlabel('Интенсивность деформации')
    plt.ylabel('Интенсивность напряжения (МПа)')
    plt.title('Кривая напряжение-деформация с переходом в пластику')
    plt.grid(True)
    plt.show()
plot_pilot(epsilon_u_values,sigma_u_values)
print(epsilon_u_values)
print(sigma_u_values)