import numpy as np
import matplotlib.pyplot as plt


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
        rect_x1, rect_y1, rect_x2, rect_y2, rect_x3, rect_y3, rect_x4, rect_y4,
        area, centroid[0], centroid[1]
    )


# Пример использования
x1, y1 = 0, 0
x2, y2 = 1, 0
x3, y3 = 5, 11
x4, y4 = 4, 11

# Получаем все переменные
(rect_x1, rect_y1, rect_x2, rect_y2,
 rect_x3, rect_y3, rect_x4, rect_y4,
 area, centroid_x, centroid_y) = approximate_quad_to_rectangle(x1, y1, x2, y2, x3, y3, x4, y4)

# Вывод результатов
print("Координаты прямоугольника:")
print(f"1-я вершина: ({rect_x1:.2f}, {rect_y1:.2f})")
print(f"2-я вершина: ({rect_x2:.2f}, {rect_y2:.2f})")
print(f"3-я вершина: ({rect_x3:.2f}, {rect_y3:.2f})")
print(f"4-я вершина: ({rect_x4:.2f}, {rect_y4:.2f})")
print(f"Площадь: {area:.2f}")
print(f"Центроид: ({centroid_x:.2f}, {centroid_y:.2f})")

# Визуализация
quad_points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
rect_points = np.array([
    [rect_x1, rect_y1],
    [rect_x2, rect_y2],
    [rect_x3, rect_y3],
    [rect_x4, rect_y4]
])

plt.figure(figsize=(8, 6))
plt.plot(np.append(quad_points[:, 0], quad_points[0, 0]),
         np.append(quad_points[:, 1], quad_points[0, 1]),
         'b-', label='Исходный четырёхугольник')
plt.plot(quad_points[:, 0], quad_points[:, 1], 'bo')

plt.plot(np.append(rect_points[:, 0], rect_points[0, 0]),
         np.append(rect_points[:, 1], rect_points[0, 1]),
         'r-', label='Аппроксимирующий прямоугольник')
plt.plot(rect_points[:, 0], rect_points[:, 1], 'ro')

plt.plot(centroid_x, centroid_y, 'g.', markersize=10, label='Центроид')
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.title(f'Площадь сохраняется: {area:.2f}')
plt.show()