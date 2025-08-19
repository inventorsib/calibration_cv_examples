import sys
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from pyapriltags import Detector
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view

#! ----------------------------------------------------------------------------------------------------------

# Colors
colors = [
    (255, 0, 0),      # Красный
    (0, 255, 0),      # Зеленый
    (0, 0, 255),      # Синий
    (255, 255, 0),    # Желтый
    (255, 0, 255),    # Пурпурный
    (0, 255, 255),    # Голубой
    (255, 128, 0),    # Оранжевый
    (128, 0, 255),    # Фиолетовый
    (0, 128, 128),    # Бирюзовый
    (255, 0, 128)     # Розово-малиновый
    ]

#! ----------------------------------------------------------------------------------------------------------
# Settings

focus_px = 1578
fx = focus_px
fy = focus_px

camera_matrix = np.array([
[fx, 0, 691],   # fx, 0, cx
[0, fy, 462],   # 0, fy, cy
[0, 0, 1]         # 0, 0, 1
], dtype=np.float32)
          
# Коэффициенты дисторсии (пример)
dist_coeffs =  np.array([-0.409, 0.298, 0.0002, 0.001, -0.214], dtype=np.float32)

def interpolateDepthMap(depth_map,  window_size=5):
    
    if window_size % 2 == 0:
        raise ValueError("Размер окна должен быть нечетным")
    
    k = window_size // 2  # радиус окна
    corrected = depth_map.copy().astype(float)
    
    # Создаем скользящие окна
    windows = sliding_window_view(depth_map, (window_size, window_size))
    
    # Создаем маску для inf значений
    inf_mask = np.isinf(depth_map)
    for i in range(k, depth_map.shape[0] - k):
        for j in range(k, depth_map.shape[1] - k):
            if not inf_mask[i, j]:
                continue  # пропускаем если в центре не inf
            # Извлекаем окно
            window = windows[i - k, j - k].copy()
            # Заменяем inf на NaN для удобства обработки
            window[np.isinf(window)] = np.nan
            mean_val = np.nanmedian(window)
            
            # Если есть валидные значения в окне - заменяем центр
            if not np.isnan(mean_val):
                corrected[i, j] = mean_val
                
    return corrected
            

def getTagSizeInFrame(corners):
    side_lengths = []
    for i in range(4):
        # Берем соседние точки (замыкаем контур: последняя с первой)
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        # Вычисляем Евклидово расстояние между точками
        length = np.linalg.norm(p1 - p2)
        side_lengths.append(length)
    
    # Средний размер стороны метки
    avg_size = np.mean(side_lengths)
    return avg_size

#! Координаты углов на изображении
def getTagPositionInFrame(detector, gray, image):

    results = detector.detect(gray)
    print(results)

    tags_corners = []
    corners = []
    for i, detection in enumerate(results):
        # Координаты углов в формате [ [x1, y1], [x2, y2], [x3, y3], [x4, y4] ]
        
        corners = detection.corners.astype(int)

        if getTagSizeInFrame(corners)<25:
            continue

        tags_corners.append(corners)
        
        # Визуализация (опционально)
        for j in range(4):
            start = tuple(corners[j])
            end = tuple(corners[(j + 1) % 4])
            cv2.line(image, start, end, colors[i], 2)
            cv2.putText(image, str(j), start, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)

    return tags_corners

#! ----------------------------------------------------------------------------------------------------------

# Парсим аргументы командной строки
image_file_name = "1.png"
tag_families = 'tag16h5'

if len(sys.argv) == 1:
    print("imageFileName tagFamiliy")
if len(sys.argv) > 1:
    image_file_name = sys.argv[1]
if len(sys.argv) > 2:
    tag_families = sys.argv[2]
    
print("filename ", image_file_name, "/  families", tag_families)

# Создание детектора
detector = Detector(
    families=tag_families,  # Тип меток (по умолчанию)
    nthreads=1,           # Количество потоков
    quad_decimate=1.0,    # Уменьшение разрешения изображения
    quad_sigma=1.0,       # Размытие изображения
    decode_sharpening=0.7,
    refine_edges=0.1        # Точность определения границ
)

image = cv2.imread(image_file_name)
image = cv2.undistort(image, camera_matrix, dist_coeffs, None, None)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Создание детектора
detector = Detector(
    families='tag16h5',  # Тип меток (по умолчанию)
    nthreads=1,           # Количество потоков
    quad_decimate=1.0,    # Уменьшение разрешения изображения
    quad_sigma=0.0,       # Размытие изображения
    refine_edges=1        # Точность определения границ
)

tags_corners = getTagPositionInFrame(detector, gray, image)
tags_corners = np.array(tags_corners, dtype=np.float32)
plt.imshow(image)
plt.show()

#! ----------------------------------------------------------------------------------------------------------
# Модуль perspective n points

tag_size = 223  # размер метки в мм
# 3D координаты углов первой метки (в своей системе координат)
object_points = [
[-tag_size/2,  tag_size/2, 0],
[ tag_size/2,  tag_size/2, 0],
[ tag_size/2,  -tag_size/2, 0],
[-tag_size/2,  -tag_size/2, 0]]
object_points = np.array(object_points, dtype=np.float32)


for tag_corner in tags_corners:
    success, rvec, tvec = cv2.solvePnP(object_points, tag_corner, camera_matrix, dist_coeffs)
    if(np.linalg.norm(tvec)<5000):
        rotation_matrix, _ = cv2.Rodrigues(rvec)


#! ----------------------------------------------------------------------------------------------------------

# Загрузка данных из файла (формат: X Y Z R на каждой строке)
data = np.loadtxt('file0808_1.txt')

scale_factor = 4

width = 1920//scale_factor
height = 1280//scale_factor
cx = width/2
cy = height/2

focus_px = 1578//scale_factor

fx = focus_px
fy = focus_px

depth_map = np.full((height, width), np.inf)
for _, row in enumerate(data):
    xpt, ypt, zpt, r = row[:4] 

    z = xpt
    x = ypt
    y = zpt
    r = r
    #
    if z<=0 or z>10000:
        continue 
        
    u = (x * fx) / z + cx
    v = (y * fy) / z + cy
    
    #    
    u_idx = int(round(u))
    v_idx = int(round(v))
    
    #   
    if 0 <= u_idx < width and 0 <= v_idx < height:
        if r < depth_map[v_idx, u_idx]:
            depth_map[v_idx, u_idx] = r


# TODO: depth_map = interpolateDepthMap(depth_map, 5)
# 
depth_map[np.isinf(depth_map)] = 0  
depth_map[np.isnan(depth_map)] = 0

depth_visual = (255 * depth_map / np.max(depth_map))

depth_map = depth_visual.astype(np.uint8)
depth_map_img = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)
depth_map_img = cv2.convertScaleAbs(depth_map_img, alpha=2.5, beta=0)


depth_map_img = cv2.resize(
    depth_map_img,
    (int(width*scale_factor), int(height*scale_factor)), 
    interpolation=cv2.INTER_LINEAR
)
depth_map = cv2.resize(
    depth_map,
    (int(width*scale_factor), int(height*scale_factor)), 
    interpolation=cv2.INTER_LINEAR
)

tags_corners = getTagPositionInFrame(detector, depth_map, depth_map_img)


plt.imshow(depth_map_img)
plt.show()






