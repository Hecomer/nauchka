import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def combined_approximation(x1, y1, x2, y2, x3, y3, x4, y4):
    # Исходные точки
    points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

    # 1. Вычисляем площадь исходного четырёхугольника
    def compute_area(points):
        x = points[:, 0]
        y = points[:, 1]
        return 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

    original_area = compute_area(points)

    # 2. Применяем PCA для определения главных осей
    pca = PCA(n_components=2)
    pca.fit(points)
    pc1, pc2 = pca.components_

    # Центроид
    centroid = np.mean(points, axis=0)

    # 3. Проецируем точки на главные оси
    projected = pca.transform(points)
    min_pc1, max_pc1 = np.min(projected[:, 0]), np.max(projected[:, 0])
    min_pc2, max_pc2 = np.min(projected[:, 1]), np.max(projected[:, 1])

    # Размеры вдоль главных осей
    length = max_pc1 - min_pc1
    width = max_pc2 - min_pc2

    # 4. Угол поворота прямоугольника
    angle = np.arctan2(pc1[1], pc1[0])

    # 5. Строим прямоугольник в PCA-пространстве
    half_length = length / 2
    half_width = width / 2
    rect_local = np.array([
        [-half_length, -half_width],
        [half_length, -half_width],
        [half_length, half_width],
        [-half_length, half_width]
    ])

    # 6. Масштабируем прямоугольник, чтобы площадь совпала с исходной
    current_area = length * width
    scale_factor = np.sqrt(original_area / current_area)
    rect_local_scaled = rect_local * scale_factor

    # 7. Поворачиваем и сдвигаем обратно в исходную систему
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    rect_rotated = np.dot(rect_local_scaled, rotation_matrix)
    rect_final = rect_rotated + centroid

    # 8. Извлекаем координаты вершин
    rect_x1, rect_y1 = rect_final[0]
    rect_x2, rect_y2 = rect_final[1]
    rect_x3, rect_y3 = rect_final[2]
    rect_x4, rect_y4 = rect_final[3]

    return (
        rect_x1, rect_y1, rect_x2, rect_y2, rect_x3, rect_y3, rect_x4, rect_y4,
        original_area, centroid[0], centroid[1], angle
    )


# Пример использования
x1, y1 = 0, 0
x2, y2 = 2, 1
x3, y3 = 3, 4
x4, y4 = 1, 3

# Получаем результаты
(rect_x1, rect_y1, rect_x2, rect_y2,
 rect_x3, rect_y3, rect_x4, rect_y4,
 area, centroid_x, centroid_y, angle) = combined_approximation(x1, y1, x2, y2, x3, y3, x4, y4)

# Вывод
print("Координаты прямоугольника (комбинированный метод):")
print(f"1-я вершина: ({rect_x1:.2f}, {rect_y1:.2f})")
print(f"2-я вершина: ({rect_x2:.2f}, {rect_y2:.2f})")
print(f"3-я вершина: ({rect_x3:.2f}, {rect_y3:.2f})")
print(f"4-я вершина: ({rect_x4:.2f}, {rect_y4:.2f})")
print(f"Площадь: {area:.2f}")
print(f"Центроид: ({centroid_x:.2f}, {centroid_y:.2f})")
print(f"Угол поворота: {np.degrees(angle):.2f}°")

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
         'r-', label='Комбинированный прямоугольник')
plt.plot(rect_points[:, 0], rect_points[:, 1], 'ro')

plt.plot(centroid_x, centroid_y, 'g*', markersize=10, label='Центроид')
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.title(f'Комбинированная аппроксимация (Площадь: {area:.2f})')
plt.show()