import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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
        rect_x1, rect_y1, rect_x2, rect_y2, rect_x3, rect_y3, rect_x4, rect_y4,
        original_area, new_area, centroid[0], centroid[1], angle
    )


# Пример использования
x1, y1, x2, y2, x3, y3, x4, y4 = 0.7701372587635112, 0.8888888888947125, 0.8002091452414226,\
                                 0.8888888888962341, 0.8047161867342936, 0.9629629629633799,\
                                 0.7799004073644833, 0.9629629629637829


# Получаем все переменные
(rect_x1, rect_y1, rect_x2, rect_y2,
 rect_x3, rect_y3, rect_x4, rect_y4,
 original_area, new_area,
 centroid_x, centroid_y, angle) = approximate_quad_by_pca(x1, y1, x2, y2, x3, y3, x4, y4)

# Вывод результатов
print("Координаты прямоугольника (PCA-аппроксимация):")
print(f"1-я вершина: ({rect_x1:.2f}, {rect_y1:.2f})")
print(f"2-я вершина: ({rect_x2:.2f}, {rect_y2:.2f})")
print(f"3-я вершина: ({rect_x3:.2f}, {rect_y3:.2f})")
print(f"4-я вершина: ({rect_x4:.2f}, {rect_y4:.2f})")
print(f"Площадь исходного четырёхугольника: {original_area:.10f}")
print(f"Площадь аппроксимирующего прямоугольника: {new_area:.9f}")
print(f"Центроид: ({centroid_x:.2f}, {centroid_y:.2f})")
print(f"Угол поворота прямоугольника: {np.degrees(angle):.2f}°")

# Визуализация
points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
rect_points = np.array([
    [rect_x1, rect_y1],
    [rect_x2, rect_y2],
    [rect_x3, rect_y3],
    [rect_x4, rect_y4]
])

plt.figure(figsize=(8, 6))
plt.plot(np.append(points[:, 0], points[0, 0]),
         np.append(points[:, 1], points[0, 1]),
         'b-', label='Исходный четырёхугольник')
plt.plot(points[:, 0], points[:, 1], 'bo')

plt.plot(np.append(rect_points[:, 0], rect_points[0, 0]),
         np.append(rect_points[:, 1], rect_points[0, 1]),
         'r-', label='PCA-прямоугольник')
plt.plot(rect_points[:, 0], rect_points[:, 1], 'ro')

plt.plot(centroid_x, centroid_y, 'g.', markersize=10, label='Центр')
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.title(f'PCA-аппроксимация (Площадь: {original_area:.5f} → {new_area:.5f})')
plt.show()