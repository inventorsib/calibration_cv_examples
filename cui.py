import cmd
import shlex
import os
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import cv2
from matplotlib import pyplot as plt
from pyapriltags import Detector
from scipy.spatial import KDTree

# Цвета для визуализации
COLORS = [
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

class PointCloudCLI(cmd.Cmd):
    """
    Расширенный CLI для работы с облаками точек и изображениями.
    Поддерживает загрузку, обработку и анализ данных.
    """
    prompt = ">>> "
    intro = "PointCloud CLI. Введите 'help' для списка команд или 'help <команда>' для подробной информации"
    
    # Поддерживаемые форматы файлов
    IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    POINT_CLOUD_EXTENSIONS = ['.txt', '.csv']
    
    def __init__(self):
        super().__init__()
        self.work_dir = os.getcwd()
        self.file_history: List[str] = []
        self.image_history: Dict[str, np.ndarray] = {}  # {путь: изображение}
        self.point_cloud_history: Dict[str, np.ndarray] = {}  # {путь: облако точек}
        self.current_image: Optional[np.ndarray] = None
        self.current_path: Optional[str] = None

        self.detected_tags_history: Dict[str, List[Dict[str, Any]]] = {}  # История обнаруженных меток
        self.camera_matrix: Optional[np.ndarray] = None  # Матрица камеры
        self.dist_coeffs: Optional[np.ndarray] = None    # Коэффициенты дисторсии
        self.current_image: Optional[np.ndarray] = None
        self.current_path: Optional[str] = None
    
        # Стандартные параметры камеры
        self._setup_default_camera_parameters()

    def _setup_default_camera_parameters(self):
        """Установка параметров камеры по умолчанию"""
        # Эти параметры нужно настроить под вашу камеру!
        self.camera_matrix = np.array([
            [1000.0, 0.0, 640.0],    # fx, 0, cx
            [0.0, 1000.0, 480.0],    # 0, fy, cy  
            [0.0, 0.0, 1.0]          # 0, 0, 1
        ], dtype=np.float32)
        
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)  # Без дисторсии

    def do_set_camera_parameters(self, arg: str):
        """
        Установить параметры камеры для PnP.
        
        Использование: set_camera_parameters [fx] [fy] [cx] [cy] [k1] [k2] [p1] [p2]
        
        Параметры по умолчанию: fx=1000, fy=1000, cx=640, cy=480, без дисторсии
        """
        try:
            args = shlex.split(arg) if arg else []
            
            if args:
                if len(args) < 4:
                    raise ValueError("Укажите как минимум fx, fy, cx, cy")
                    
                fx = float(args[0])
                fy = float(args[1]) if len(args) > 1 else float(args[0])
                cx = float(args[2])
                cy = float(args[3])
                
                self.camera_matrix = np.array([
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float32)
                
                # Коэффициенты дисторсии
                if len(args) >= 8:
                    self.dist_coeffs = np.array([
                        [float(args[4])],  # k1
                        [float(args[5])],  # k2  
                        [float(args[6])],  # p1
                        [float(args[7])]   # p2
                    ], dtype=np.float32)
                else:
                    self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
            
            # Показать текущие параметры
            print("Текущие параметры камеры:")
            print("Матрица камеры:")
            print(self.camera_matrix)
            print("Коэффициенты дисторсии:")
            print(self.dist_coeffs.flatten())
            
        except Exception as e:
            print(f"Ошибка установки параметров: {e}")

    def _visualize_with_matplotlib(self, points: np.ndarray, color_mode: str, title: str):
        """3D визуализация с использованием matplotlib"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Подготовка данных для визуализации
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        
        # Выбор цвета точек
        colors = self._get_point_colors(points, color_mode)
        
        # Визуализация
        scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', s=1, alpha=0.6)
        
        # Настройка графика
        ax.set_title(f'3D Облако точек: {title}\n{len(points):,} точек', fontsize=14)
        ax.set_xlabel('X (м)')
        ax.set_ylabel('Y (м)')
        ax.set_zlabel('Z (м)')
        
        # Добавление цветовой шкалы
        if color_mode != 'rgb' and len(np.unique(colors)) > 1:
            plt.colorbar(scatter, ax=ax, label=self._get_colorbar_label(color_mode))
        
        # Настройка вида
        ax.view_init(elev=30, azim=45)
        ax.grid(True)
        
        # Автоматическое масштабирование
        self._set_3d_axes_limits(ax, points)
        
        plt.tight_layout()
        plt.show()


    def _get_point_colors(self, points: np.ndarray, color_mode: str) -> np.ndarray:
        """Получение цветов точек в зависимости от режима"""
        if color_mode == 'height' and points.shape[1] >= 3:
            return points[:, 2]  # Высота (Z координата)
        
        elif color_mode == 'intensity' and points.shape[1] >= 4:
            return points[:, 3]  # Интенсивность/отражательная способность
        
        elif color_mode == 'distance' and points.shape[1] >= 3:
            # Евклидово расстояние от начала координат
            return np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
        
        elif color_mode == 'rgb' and points.shape[1] >= 6:
            # RGB цвета (предполагаем, что каналы в столбцах 3,4,5)
            return points[:, 3:6] / 255.0  # Нормализация к [0,1]
        
        else:
            # По умолчанию - высота или последовательные значения
            if points.shape[1] >= 3:
                return points[:, 2]
            else:
                return np.arange(len(points))

    def _get_colorbar_label(self, color_mode: str) -> str:
        """Получение подписи для цветовой шкалы"""
        labels = {
            'height': 'Высота (Z)',
            'intensity': 'Интенсивность',
            'distance': 'Расстояние от начала',
            'rgb': 'RGB цвет'
        }
        return labels.get(color_mode, 'Значение')

    def _set_3d_axes_limits(self, ax, points: np.ndarray):
        """Установка пределов осей для 3D графика"""
        if points.shape[1] >= 3:
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            
            # Вычисление диапазонов
            x_range = x.max() - x.min()
            y_range = y.max() - y.min()
            z_range = z.max() - z.min()
            
            max_range = max(x_range, y_range, z_range) * 0.5
            
            # Центрирование
            mid_x = (x.max() + x.min()) * 0.5
            mid_y = (y.max() + y.min()) * 0.5
            mid_z = (z.max() + z.min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

    def do_analyze_point_cloud(self, arg: str):
        """
        Анализ статистики облака точек.
        
        Использование: analyze_point_cloud [источник]
        """
        try:
            if not self.point_cloud_history:
                raise ValueError("Нет загруженных облаков точек")
                
            points, path = self._get_item_from_history(
                arg or list(self.point_cloud_history.keys())[-1],
                self.point_cloud_history, "облако точек"
            )
            
            self._print_point_cloud_stats(points, Path(path).name)
            
        except Exception as e:
            print(f"Ошибка анализа: {e}")

    def _print_point_cloud_stats(self, points: np.ndarray, filename: str):
        """Вывод статистики облака точек"""
        print("╔══════════════════════════════════════════════════╗")
        print("║              СТАТИСТИКА ОБЛАКА ТОЧЕК            ║")
        print("╠══════════════════════════════════════════════════╣")
        print(f"║ Файл: {filename:40s} ║")
        print(f"║ Количество точек: {points.shape[0]:26,d} ║")
        print(f"║ Измерений на точку: {points.shape[1]:24d} ║")
        print("╠══════════════════════════════════════════════════╣")
        
        if points.shape[1] >= 3:
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            
            stats = [
                ("X", x), ("Y", y), ("Z", z)
            ]
            
            for dim_name, data in stats:
                print(f"║ {dim_name}: min={data.min():7.2f} max={data.max():7.2f} mean={data.mean():7.2f} ║")
        
        if points.shape[1] >= 4:
            intensity = points[:, 3]
            print(f"║ Интенсивность: min={intensity.min():5.1f} max={intensity.max():5.1f} ║")
        
        print("╚══════════════════════════════════════════════════╝")

    def do_filter_point_cloud(self, arg: str):
        """
        Фильтрация облака точек по различным критериям.
        
        Использование: filter_point_cloud [источник] [тип] [min] [max] [сохранить]
        
        Параметры:
          источник: индекс или путь к облаку точек
          тип: 'x', 'y', 'z', 'intensity', 'distance'
          min: минимальное значение
          max: максимальное значение  
          сохранить: 'save' для сохранения результата
        """
        try:
            args = shlex.split(arg) if arg else []
            
            if len(args) < 4:
                raise ValueError("Укажите: источник тип min max [save]")
                
            source = args[0]
            filter_type = args[1]
            min_val = float(args[2])
            max_val = float(args[3])
            save_result = len(args) > 4 and args[4].lower() == 'save'
            
            points, path = self._get_item_from_history(
                source, self.point_cloud_history, "облако точек"
            )
            
            # Применение фильтра
            filtered_points = self._apply_filter(points, filter_type, min_val, max_val)
            
            print(f"Исходных точек: {len(points):,}")
            print(f"После фильтрации: {len(filtered_points):,}")
            print(f"Сохранилось: {len(filtered_points)/len(points)*100:.1f}%")
            
            if save_result and len(filtered_points) > 0:
                new_filename = f"filtered_{Path(path).name}"
                np.savetxt(new_filename, filtered_points)
                print(f"✓ Отфильтрованное облако сохранено как: {new_filename}")
                
            # Показ результата
            self._visualize_with_matplotlib(filtered_points, 'height', f"Filtered: {filter_type}")
            
        except Exception as e:
            print(f"Ошибка фильтрации: {e}")

    def _apply_filter(self, points: np.ndarray, filter_type: str, min_val: float, max_val: float) -> np.ndarray:
        """Применение фильтра к облаку точек"""
        if filter_type == 'x' and points.shape[1] >= 1:
            mask = (points[:, 0] >= min_val) & (points[:, 0] <= max_val)
        elif filter_type == 'y' and points.shape[1] >= 2:
            mask = (points[:, 1] >= min_val) & (points[:, 1] <= max_val)
        elif filter_type == 'z' and points.shape[1] >= 3:
            mask = (points[:, 2] >= min_val) & (points[:, 2] <= max_val)
        elif filter_type == 'intensity' and points.shape[1] >= 4:
            mask = (points[:, 3] >= min_val) & (points[:, 3] <= max_val)
        elif filter_type == 'distance' and points.shape[1] >= 3:
            distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
            mask = (distances >= min_val) & (distances <= max_val)
        else:
            raise ValueError(f"Неизвестный тип фильтра: {filter_type}")
        
        return points[mask]

    

    # ------ Утилитарные методы ------
    def _validate_path(self, path: str, check_exists: bool = True, 
                      is_file: bool = True, valid_extensions: Optional[List[str]] = None) -> Path:
        """
        Валидация пути к файлу/директории.
        
        Args:
            path: Путь для валидации
            check_exists: Проверять существование
            is_file: Ожидается файл (True) или директория (False)
            valid_extensions: Список допустимых расширений
            
        Returns:
            Path: Валидированный путь
            
        Raises:
            ValueError: При ошибках валидации
        """
        path_obj = Path(path).expanduser().absolute()
        
        if check_exists and not path_obj.exists():
            raise ValueError(f"Путь не существует: {path_obj}")
            
        if is_file and path_obj.is_dir():
            raise ValueError(f"Ожидается файл, но найдена директория: {path_obj}")
            
        if not is_file and path_obj.is_file():
            raise ValueError(f"Ожидается директория, но найден файл: {path_obj}")
            
        if valid_extensions and is_file:
            if path_obj.suffix.lower() not in valid_extensions:
                raise ValueError(f"Неподдерживаемый формат: {path_obj.suffix}. "
                               f"Допустимые: {', '.join(valid_extensions)}")
                               
        return path_obj

    def _get_item_from_history(self, arg: str, history_dict: Dict[str, Any], 
                             item_type: str = "изображение") -> Tuple[Any, str]:
        """
        Получение элемента из истории по индексу или пути.
        
        Args:
            arg: Аргумент (индекс или путь)
            history_dict: Словарь истории
            item_type: Тип элемента для сообщений об ошибках
            
        Returns:
            Tuple[элемент, путь]
        """
        if not arg:
            raise ValueError(f"Укажите индекс или путь к {item_type}")
            
        try:
            # Попытка использовать как индекс
            index = int(arg) - 1
            if 0 <= index < len(history_dict):
                path = list(history_dict.keys())[index]
                return history_dict[path], path
            else:
                raise ValueError(f"Неверный индекс: {index + 1}")
                
        except ValueError:
            # Использование как путь
            path_obj = self._validate_path(arg, check_exists=False)
            path_str = str(path_obj)
            
            if path_str not in history_dict:
                raise ValueError(f"{item_type.capitalize()} не найдено в истории: {path_str}")
                
            return history_dict[path_str], path_str

    def _complete_path(self, text: str, include_files: bool = False, 
                      include_dirs: bool = False) -> List[str]:
        """Автодополнение путей"""
        if not text:
            text = ""
            
        search_path = os.path.join(self.work_dir, text)
        dir_path = os.path.dirname(search_path)
        base_name = os.path.basename(search_path)
        
        if not os.path.exists(dir_path):
            return []
        
        matches = []
        for name in os.listdir(dir_path):
            full_path = os.path.join(dir_path, name)
            
            if include_files and os.path.isfile(full_path) and name.startswith(base_name):
                matches.append(name)
            elif include_dirs and os.path.isdir(full_path) and name.startswith(base_name):
                matches.append(name + "/")
        
        return matches

    # ------ Команды для работы с облаками точек ------
    def do_load_point_cloud(self, arg: str):
        """
        Загрузить файл с облаком точек.
        
        Использование: load_point_cloud <путь/к/файлу>
        
        Поддерживаемые форматы: .txt, .csv
        """
        if not arg:
            print("Ошибка: Укажите путь к файлу")
            return
        
        try:
            args = shlex.split(arg)
            if len(args) > 1:
                print("Ошибка: Слишком много аргументов. Используйте кавычки для путей с пробелами")
                return
                
            path_obj = self._validate_path(args[0], valid_extensions=self.POINT_CLOUD_EXTENSIONS)
            
            # Загрузка данных
            data = np.loadtxt(path_obj)
            data_shape = data.shape
            
            if data_shape[1] < 3:
                raise ValueError("Неверный формат хранения точек. Ожидается минимум 3 столбца")
            
            path_str = str(path_obj)
            self.point_cloud_history[path_str] = data
            self.current_data = data
            self.current_path = path_str
            self.file_history.append(path_str)

            print(f"Успешно загружено: {path_obj.name}")
            print(f"Размер: {data_shape[0]} точек, {data_shape[1]} измерений")
            
        except Exception as e:
            print(f"Ошибка загрузки: {e}")

    def do_generate_depth_maps(self, arg: str):
        """
        Генерация карт глубины и отражательной способности из облака точек.
        
        Использование: generate_depth_maps [источник] [высота] [ширина] [фокус] [k] [макс_расстояние]
        
        Параметры:
          источник:      индекс или путь к облаку точек (по умолчанию: последнее)
          высота:       высота выходного изображения (по умолчанию: 480)
          ширина:       ширина выходного изображения (по умолчанию: 640)  
          фокус:        фокусное расстояние (по умолчанию: 525)
          k:            количество соседей для интерполяции (по умолчанию: 5)
          макс_расстояние: максимальное расстояние для интерполяции (по умолчанию: 10)
        """
        try:
            args = shlex.split(arg) if arg else []
            
            # Параметры по умолчанию
            params = {
                'source': args[0] if len(args) > 0 else None,
                'height': int(args[1]) if len(args) > 1 else 480,
                'width': int(args[2]) if len(args) > 2 else 640,
                'focus': float(args[3]) if len(args) > 3 else 525.0,
                'k': int(args[4]) if len(args) > 4 else 5,
                'max_dist': float(args[5]) if len(args) > 5 else 10.0
            }
            
            # Получение облака точек
            if not self.point_cloud_history:
                raise ValueError("Нет загруженных облаков точек")
                
            points, path = self._get_item_from_history(
                params['source'] or list(self.point_cloud_history.keys())[-1],
                self.point_cloud_history, "облако точек"
            )
            
            # Генерация карт
            depth_map, reflect_map = self._generate_maps_from_points(
                points, params['height'], params['width'], params['focus'],
                params['k'], params['max_dist']
            )
            
            # Сохранение результатов
            base_name = Path(path).name
            depth_name = f"DEPTH_MAP_{base_name}"
            reflect_name = f"REFLECT_MAP_{base_name}"
            
            self.image_history[depth_name] = depth_map
            if reflect_map is not None:
                self.image_history[reflect_name] = reflect_map
            
            self.current_image = depth_map
            self.current_path = depth_name
            self.file_history.extend([depth_name, reflect_name] if reflect_map is not None else [depth_name])
            
            print(f"Сгенерирована карта глубины: {depth_name} ({depth_map.shape[0]}x{depth_map.shape[1]})")
            if reflect_map is not None:
                print(f"Сгенерирована карта отражательной способности: {reflect_name} ({reflect_map.shape[0]}x{reflect_map.shape[1]})")
                
        except Exception as e:
            print(f"Ошибка генерации карт: {e}")

    def _generate_maps_from_points(self, points: np.ndarray, height: int, width: int, 
                                 focus: float, k: int, max_dist: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Генерация карт из облака точек"""
        num_dimensions = points.shape[1]
        has_reflectivity = num_dimensions >= 4
        
        # Параметры камеры
        fx = fy = focus
        cx, cy = width / 2, height / 2
        
        # Инициализация карт
        depth_map = np.full((height, width), np.inf)
        reflect_map = np.full((height, width), np.inf) if has_reflectivity else None
        
        # Проецирование точек
        for point in points:
            x_lidar, y_lidar, z_lidar = point[:3]
            reflectance = point[3] if has_reflectivity else None
            
            # Переход в систему координат кадра
            z = x_lidar
            x = y_lidar
            y = z_lidar

            if z <= 0 or z > 10000:
                continue
                
            # Проецирование 3D -> 2D
            u = (x * fx) / z + cx  # x -> u (ширина)
            v = (y * fy) / z + cy  # z -> v (высота), y -> глубина
            
            u_idx = int(round(u))
            v_idx = int(round(v))
            
            # Обновление карт
            if 0 <= u_idx < width and 0 <= v_idx < height:
                if z < depth_map[v_idx, u_idx]:
                    depth_map[v_idx, u_idx] = z
                
                if has_reflectivity and reflectance < reflect_map[v_idx, u_idx]:
                    reflect_map[v_idx, u_idx] = reflectance
        
        # Обработка пропущенных значений
        depth_map = self._process_map(depth_map, k, max_dist)
        
        if has_reflectivity:
            reflect_map = self._process_map(reflect_map, k, max_dist)
        
        return depth_map, reflect_map

    def _process_map(self, data_map: np.ndarray, k: int, max_dist: float) -> np.ndarray:
        """Обработка и заполнение пропусков в карте"""
        # Замена бесконечностей и NaN
        processed_map = np.copy(data_map)
        processed_map[np.isinf(processed_map)] = 0
        processed_map[np.isnan(processed_map)] = 0
        
        # Заполнение пропусков
        if np.any(processed_map == 0):
            processed_map = self._fill_map_gaps(processed_map, k, max_dist)
        
        return processed_map

    def _fill_map_gaps(self, data_map: np.ndarray, k: int, max_distance: float) -> np.ndarray:
        """Заполнение пропусков в карте с использованием KDTree"""
        mask = data_map == 0
        known_y, known_x = np.where(~mask)
        
        if len(known_y) == 0:
            return data_map
            
        known_points = np.column_stack((known_y, known_x))
        known_values = data_map[~mask]
        unknown_y, unknown_x = np.where(mask)
        
        tree = KDTree(known_points)
        filled_map = data_map.copy()
        
        for i, (y, x) in enumerate(zip(unknown_y, unknown_x)):
            distances, indices = tree.query([y, x], k=min(k, len(known_points)))
            
            valid = distances < max_distance
            if np.any(valid):
                weights = 1.0 / (distances[valid] + 1e-8)
                weights /= np.sum(weights)
                filled_map[y, x] = np.sum(weights * known_values[indices[valid]])
        
        return filled_map

    def do_visualize_point_cloud(self, arg: str):
        """
        3D визуализация облака точек.
        
        Использование: visualize_point_cloud [источник] [цвет]
        
        Параметры:
          источник: индекс или путь к облаку точек (по умолчанию: последнее)
          цвет: 'height', 'intensity', 'distance' или 'rgb' (по умолчанию: height)
        """
        try:
            args = shlex.split(arg) if arg else []
            
            source = args[0] if len(args) > 0 else None
            color_mode = args[1] if len(args) > 1 else 'height'
            
            if not self.point_cloud_history:
                raise ValueError("Нет загруженных облаков точек")
                
            points, path = self._get_item_from_history(
                source or list(self.point_cloud_history.keys())[-1],
                self.point_cloud_history, "облако точек"
            )
            
            print(f"Визуализация облака точек: {Path(path).name}")
            print(f"Размер: {points.shape[0]} точек, {points.shape[1]} измерений")
            

            self._visualize_with_matplotlib(points, color_mode, Path(path).name)
                
        except Exception as e:
            print(f"Ошибка визуализации: {e}")

    def do_list_point_clouds(self, arg: str):
        """
        Показать историю загруженных облаков точек.
        
        Использование: list_point_clouds
        """
        if not self.point_cloud_history:
            print("История облаков точек пуста")
            return
            
        print("\nЗагруженные облака точек:")
        for i, (path, data) in enumerate(self.point_cloud_history.items(), 1):
            path_obj = Path(path)
            print(f"{i:2d}. {path_obj.name} - {data.shape[0]} точек - {path}")

    # ------ Команды для работы с изображениями ------
    def do_load_image(self, arg: str):
        """
        Загрузить изображение.
        
        Использование: load_image <путь/к/файлу>
        
        Поддерживаемые форматы: .jpg, .jpeg, .png, .bmp, .tiff
        """
        if not arg:
            print("Ошибка: Укажите путь к файлу")
            return
        
        try:
            args = shlex.split(arg)
            if len(args) > 1:
                print("Ошибка: Слишком много аргументов. Используйте кавычки для путей с пробелами")
                return
                
            path_obj = self._validate_path(args[0], valid_extensions=self.IMAGE_EXTENSIONS)
            
            # Загрузка изображения
            img = cv2.imread(str(path_obj))
            if img is None:
                raise ValueError("Не удалось загрузить изображение")
                
            # Сохранение в истории
            path_str = str(path_obj)
            self.image_history[path_str] = img
            self.current_image = img
            self.current_path = path_str
            self.file_history.append(path_str)

            height, width = img.shape[:2]
            print(f"Успешно загружено: {path_obj.name}")
            print(f"Размер: {width}x{height} пикселей")
            
        except Exception as e:
            print(f"Ошибка загрузки: {e}")

    def do_show_image(self, arg: str):
        """
        Показать изображение.
        
        Использование: show_image [источник]
        
        Параметры:
          источник: индекс или путь к изображению (по умолчанию: последнее)
        """
        try:
            if not self.image_history:
                raise ValueError("Нет загруженных изображений")
                
            image, _ = self._get_item_from_history(
                arg or list(self.image_history.keys())[-1], 
                self.image_history
            )
            
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image, 
                      cmap='gray' if len(image.shape) == 2 else None)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Ошибка отображения: {e}")

    def do_detect_apriltags(self, arg: str):
        """
        Обнаружение AprilTags на изображении.
        
        Использование: detect_apriltags [источник] [семейство_тегов]
        
        Параметры:
          источник: индекс или путь к изображению (по умолчанию: последнее)
          семейство_тегов: тип меток (по умолчанию: tag16h5)
        """
        try:
            args = shlex.split(arg) if arg else []
            
            source = args[0] if len(args) > 0 else None
            tag_family = args[1] if len(args) > 1 else 'tag16h5'
            save_tags = len(args) > 2 and args[2].lower() == 'save'

            if not self.image_history:
                raise ValueError("Нет загруженных изображений")
                
            image, path = self._get_item_from_history(
                source or list(self.image_history.keys())[-1], 
                self.image_history
            )
            
            print(f"Обнаружение меток семейства: {tag_family}")
            
            # Подготовка детектора
            detector = Detector(
                families=tag_family,
                nthreads=1,
                quad_decimate=1.0,
                quad_sigma=1.0,
                decode_sharpening=0.7,
                refine_edges=0.1
            )
            
            # Подготовка изображения
            if len(image.shape) == 3 and image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.astype(np.uint8)
                image = self._convert_to_bgr(image)
            
            # Обнаружение меток
            result_image = np.copy(image)
            tags_info = self._detect_and_draw_tags(detector, gray, result_image)
            
            # Сохранение координат меток
            if save_tags and tags_info:
                self.detected_tags_history[path] = tags_info
                print(f"+ Координаты {len(tags_info)} меток сохранены для изображения: {Path(path).name}")

            self._print_tags_info(tags_info)

            # Отображение результатов
            plt.figure(figsize=(12, 10))
            plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            plt.title(f"Обнаружено меток: {len(tags_info)}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Ошибка обнаружения меток: {e}")

    def _print_tags_info(self, tags_info: List[Dict[str, Any]]) -> None:
        """Красивый вывод информации о метках в консоль"""
        if not tags_info:
            print("╔══════════════════════════════════════════════════╗")
            print("║              МЕТКИ НЕ ОБНАРУЖЕНЫ                ║")
            print("╚══════════════════════════════════════════════════╝")
            return
        
        print("╔═════════════════════════════════════════════════════════════════════════════════════╗")
        print("║                        ОБНАРУЖЕННЫЕ APRILTAGS                                       ║")
        print("╠═════════════════════════════════════════════════════════════════════════════════════╣")
        print("║ ID │   Центр (x,y)      │   Размер   │           Углы (координаты x,y)              ║")
        print("╠════╪════════════════════╪════════════╪══════════════════════════════════════════════╣")
        
        for tag in tags_info:
            center_x, center_y = tag['center']
            corners_str = ""
            
            # Форматирование координат углов
            for j, corner in enumerate(tag['corners']):
                corner_x, corner_y = corner
                corners_str += f"{j}:({corner_x:3d},{corner_y:3d}) "
            
            print(f"║ {tag['id']:2d} │ ({center_x:6.1f}, {center_y:6.1f})   │ {tag['size']:7.1f}px │ {corners_str}")
            
            
            if tag != tags_info[-1]:  # Разделитель между метками
                print("╠════╪════════════════════╪════════════╪══════════════════════════════════════════════╣")
        
        print("╚═══╪══════════════════════╪════════════╪═════════════════════════════════════════════╝")
        print(f"Всего обнаружено меток: {len(tags_info)}")
        
        # Статистика
        if len(tags_info) > 1:
            avg_size = np.mean([tag['size'] for tag in tags_info])
            min_size = np.min([tag['size'] for tag in tags_info])
            max_size = np.max([tag['size'] for tag in tags_info])
            print(f"Средний размер: {avg_size:.1f}px (min: {min_size:.1f}px, max: {max_size:.1f}px)")

    def _detect_and_draw_tags(self, detector: Detector, gray_image: np.ndarray, 
                            output_image: np.ndarray) -> List[Dict[str, Any]]:
        """Обнаружение и отрисовка AprilTags с возвратом информации о метках"""
        results = detector.detect(gray_image)
        tags_info = []
        
        for i, detection in enumerate(results):
            corners = detection.corners.astype(int)
            
            # Фильтрация маленьких меток
            if self._calculate_tag_size(corners) < 25:
                continue
                
            # Сохранение информации о метке (центр оставляем как float для точности)
            tag_info = {
                'id': detection.tag_id,
                'corners': corners,
                'center': detection.center,  # Оставляем как float
                'center_int': detection.center.astype(int),  # Целочисленная версия для отрисовки
                'size': self._calculate_tag_size(corners),
                'hamming': detection.hamming,
                'decision_margin': detection.decision_margin
            }
            tags_info.append(tag_info)
            
            # Отрисовка метки (используем целочисленные координаты)
            for j in range(4):
                start = tuple(corners[j])
                end = tuple(corners[(j + 1) % 4])
                center = tuple(tag_info['center_int'])
                
                cv2.line(output_image, start, end, COLORS[i % len(COLORS)], 2)
                cv2.putText(output_image, str(j), start, cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, COLORS[i % len(COLORS)], 2)
                cv2.putText(output_image, f"id={detection.tag_id}", center, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[i % len(COLORS)], 2)
        
        return tags_info

    def _calculate_tag_size(self, corners: np.ndarray) -> float:
        """Вычисление среднего размера стороны метки"""
        side_lengths = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            length = np.linalg.norm(p1 - p2)
            side_lengths.append(length)
        
        return np.min(side_lengths)

    def _convert_to_bgr(self, image: np.ndarray) -> np.ndarray:
        """Безопасное преобразование в BGR формат"""
        if image is None:
            return None
        
        if image.dtype in [np.float64, np.float32]:
            # Нормализация float изображений
            if image.max() <= 1.0:
                image_8bit = (image * 255).astype(np.uint8)
            else:
                image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            return cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2BGR)
        else:
            try:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            except cv2.error:
                image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                return cv2.cvtColor(image_normalized, cv2.COLOR_GRAY2BGR)

    def do_list_images(self, arg: str):
        """
        Показать историю загруженных изображений.
        
        Использование: list_images
        """
        if not self.image_history:
            print("История изображений пуста")
            return
            
        print("\nЗагруженные изображения:")
        for i, (path, img) in enumerate(self.image_history.items(), 1):
            path_obj = Path(path)
            height, width = img.shape[:2]
            print(f"{i:2d}. {path_obj.name} ({width}x{height}) - {path}")

    def do_remove_image(self, arg: str):
        """
        Удалить изображение из истории.
        
        Использование: remove_image [индекс|путь]
        """
        if not arg:
            print("Укажите индекс или путь изображения")
            return
            
        try:
            path_to_remove = None
            
            # Попытка удаления по индексу
            try:
                index = int(arg) - 1
                if 0 <= index < len(self.image_history):
                    path_to_remove = list(self.image_history.keys())[index]
                else:
                    print("Ошибка: Неверный индекс")
                    return
                    
            except ValueError:
                # Удаление по пути
                path_obj = self._validate_path(arg, check_exists=False)
                path_to_remove = str(path_obj)
                
            if path_to_remove in self.image_history:
                del self.image_history[path_to_remove]
                print(f"Изображение удалено: {Path(path_to_remove).name}")
                
                # Обновление текущего изображения
                if self.current_path == path_to_remove:
                    self.current_image = None
                    self.current_path = None
                    if self.image_history:
                        self.current_path = list(self.image_history.keys())[-1]
                        self.current_image = self.image_history[self.current_path]
            else:
                print("Ошибка: Изображение не найдено в истории")
                
        except Exception as e:
            print(f"Ошибка удаления: {e}")


    def do_solve_pnp(self, arg: str):
        '''
        Решение PnP задачи для преобразования 2D координат в 3D.
        
        Использование: solve_pnp [изображение] [размер_метки] [метод]
        
        Параметры:
        изображение: индекс или путь к изображению с сохраненными метками
        размер_метки: физический размер метки в метрах (по умолчанию: 0.1)
        метод: метод решения PnP (ITERATIVE, P3P, EPnP, etc.) (по умолчанию: ITERATIVE)
        '''
        try:
            args = shlex.split(arg) if arg else []
            
            source = args[0] if len(args) > 0 else None
            tag_size = float(args[1]) if len(args) > 1 else 0.1
            method = args[2] if len(args) > 2 else 'ITERATIVE'
            
            if not self.detected_tags_history:
                raise ValueError("Нет сохраненных координат меток. Используйте 'detect_apriltags save'")
            
            # Получение изображения с метками
            image_path = source or list(self.detected_tags_history.keys())[-1]
            if image_path not in self.detected_tags_history:
                raise ValueError(f"Нет сохраненных меток для изображения: {image_path}")
            
            tags_info = self.detected_tags_history[image_path]
            
            if not tags_info:
                raise ValueError("Нет меток для решения PnP")
            
            # Подготовка 3D модельных точек и 2D изображенных точек
            object_points = []
            image_points = []
            
            for tag in tags_info:
                # 3D координаты углов метки в системе координат метки
                half_size = tag_size / 2.0
                tag_object_points = np.array([
                    [-half_size, -half_size, 0.0],  # Левый нижний
                    [ half_size, -half_size, 0.0],  # Правый нижний
                    [ half_size,  half_size, 0.0],  # Правый верхний
                    [-half_size,  half_size, 0.0]   # Левый верхний
                ], dtype=np.float32)
                
                object_points.extend(tag_object_points)
                image_points.extend(tag['corners'])
            
            object_points = np.array(object_points, dtype=np.float32)
            image_points = np.array(image_points, dtype=np.float32)
            
            # Решение PnP
            success, rvec, tvec = self._solve_pnp_problem(
                object_points, image_points, method
            )
            
            if success:
                # Преобразование результатов
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                euler_angles = self._rotation_matrix_to_euler(rotation_matrix)
                
                # Вывод результатов
                print("╔══════════════════════════════════════════════════╗")
                print("║               РЕЗУЛЬТАТЫ PnP                    ║")
                print("╠══════════════════════════════════════════════════╣")
                print(f"║ Метод: {method:38s} ║")
                print(f"║ Количество точек: {len(object_points):26d} ║")
                print("╠══════════════════════════════════════════════════╣")
                print(f"║ Вектор перемещения (tvec):                      ║")
                print(f"║   X: {tvec[0, 0]:8.3f} m                         ║")
                print(f"║   Y: {tvec[1, 0]:8.3f} m                         ║")
                print(f"║   Z: {tvec[2, 0]:8.3f} m                         ║")
                print("╠══════════════════════════════════════════════════╣")
                print(f"║ Углы Эйлера (в градусах):                       ║")
                print(f"║   Roll (X):  {euler_angles[0]:6.1f}°               ║")
                print(f"║   Pitch (Y): {euler_angles[1]:6.1f}°               ║")
                print(f"║   Yaw (Z):   {euler_angles[2]:6.1f}°               ║")
                print("╚══════════════════════════════════════════════════╝")
                
                # Сохранение результатов
                self._save_pnp_results(image_path, rvec, tvec, rotation_matrix)
                
            else:
                print("Не удалось решить PnP задачу")
                
        except Exception as e:
            print(f"Ошибка решения PnP: {e}")

    def _solve_pnp_problem(self, object_points: np.ndarray, image_points: np.ndarray, 
                        method: str = 'ITERATIVE') -> Tuple[bool, np.ndarray, np.ndarray]:
        """Решение PnP задачи"""
        try:
            # Выбор метода решения
            pnp_methods = {
                'ITERATIVE': cv2.SOLVEPNP_ITERATIVE,
                'P3P': cv2.SOLVEPNP_P3P,
                'EPNP': cv2.SOLVEPNP_EPNP,
                'IPPE': cv2.SOLVEPNP_IPPE
            }
            
            method_flag = pnp_methods.get(method.upper(), cv2.SOLVEPNP_ITERATIVE)
            
            # Решение PnP
            success, rvec, tvec = cv2.solvePnP(
                object_points, image_points, 
                self.camera_matrix, self.dist_coeffs,
                flags=method_flag
            )
            
            return success, rvec, tvec
            
        except Exception as e:
            print(f"Ошибка в solvePnP: {e}")
            return False, None, None

    def _rotation_matrix_to_euler(self, R: np.ndarray) -> np.ndarray:
        """Преобразование матрицы вращения в углы Эйлера"""
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        # Конвертация в градусы
        return np.degrees([x, y, z])

    def _save_pnp_results(self, image_path: str, rvec: np.ndarray, tvec: np.ndarray, 
                        rotation_matrix: np.ndarray) -> None:
        """Сохранение результатов PnP в файл"""
        try:
            results_dir = Path("pnp_results")
            results_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pnp_{Path(image_path).stem}_{timestamp}.txt"
            filepath = results_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Результаты PnP для изображения: {image_path}\n")
                f.write(f"Время анализа: {datetime.now()}\n\n")
                
                f.write("Параметры камеры:\n")
                f.write(f"Матрица камеры:\n{self.camera_matrix}\n")
                f.write(f"Коэффициенты дисторсии:\n{self.dist_coeffs.flatten()}\n\n")
                
                f.write("Вектор вращения (rvec):\n")
                f.write(f"{rvec.flatten()}\n\n")
                
                f.write("Вектор перемещения (tvec):\n")
                f.write(f"X: {tvec[0, 0]:.6f} m\n")
                f.write(f"Y: {tvec[1, 0]:.6f} m\n")
                f.write(f"Z: {tvec[2, 0]:.6f} m\n\n")
                
                f.write("Матрица вращения:\n")
                f.write(f"{rotation_matrix}\n\n")
                
                euler_angles = self._rotation_matrix_to_euler(rotation_matrix)
                f.write("Углы Эйлера (градусы):\n")
                f.write(f"Roll (X):  {euler_angles[0]:.2f}°\n")
                f.write(f"Pitch (Y): {euler_angles[1]:.2f}°\n")
                f.write(f"Yaw (Z):   {euler_angles[2]:.2f}°\n")
            
            print(f"✓ Результаты сохранены в: {filepath}")
            
        except Exception as e:
            print(f"Ошибка сохранения результатов: {e}")

    def do_list_saved_tags(self, arg: str):
        """
        Показать список изображений с сохраненными метками.
        
        Использование: list_saved_tags
        """
        if not self.detected_tags_history:
            print("Нет сохраненных координат меток")
            return
        
        print("╔══════════════════════════════════════════════════╗")
        print("║         ИЗОБРАЖЕНИЯ С СОХРАНЕННЫМИ МЕТКАМИ       ║")
        print("╠══════════════════════════════════════════════════╣")
        
        for i, (path, tags) in enumerate(self.detected_tags_history.items(), 1):
            path_obj = Path(path)
            print(f"║ {i:2d}. {path_obj.name:45s} ║")
            print(f"║    Количество меток: {len(tags):26d} ║")
            if i < len(self.detected_tags_history):
                print("║----------------------------------------------------║")
        
        print("╚══════════════════════════════════════════════════╝")

    # ------ Команды для работы с файловой системой ------
    def do_list_directory(self, arg: str):
        """
        Показать содержимое директории.
        
        Использование: list_directory [путь]
        """
        path = arg if arg else self.work_dir
        try:
            path_obj = self._validate_path(path, is_file=False)
            
            print(f"\nСодержимое {path_obj}:")
            items = list(path_obj.iterdir())
            
            for item in sorted(items, key=lambda x: (not x.is_dir(), x.name.lower())):
                if item.is_dir():
                    print(f"{item.name}/")
                else:
                    size = item.stat().st_size
                    print(f"{item.name} ({size} байт)")
                    
        except Exception as e:
            print(f"Ошибка: {e}")

    def do_change_directory(self, arg: str):
        """
        Сменить текущую директорию.
        
        Использование: change_directory <путь>
        """
        if not arg:
            print(f"Текущая директория: {self.work_dir}")
            return
            
        try:
            path_obj = self._validate_path(arg, is_file=False)
            
            self.work_dir = str(path_obj)
            os.chdir(self.work_dir)
            print(f"Текущая директория: {self.work_dir}")
            
        except Exception as e:
            print(f"Ошибка: {e}")

    # ------ Команды для работы с историей ------
    def do_history(self, arg: str):
        """
        Показать историю операций.
        
        Использование: history
        """
        if not self.file_history:
            print("История операций пуста")
            return
            
        print("\nИстория операций:")
        for i, path in enumerate(self.file_history, 1):
            path_obj = Path(path)
            print(f"{i:2d}. {path_obj.name} - {path}")

    def do_clear_history(self, arg: str):
        """
        Очистить историю операций.
        
        Использование: clear_history
        """
        self.file_history.clear()
        print("История операций очищена")

    # ------ Системные команды ------
    def do_exit(self, arg: str):
        """
        Выйти из приложения.
        
        Использование: exit
        """
        print("Сессия завершена")
        print(f"Всего операций: {len(self.file_history)}")
        return True

    # ------ Автодополнение ------
    def complete_load_image(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        return self._complete_path(text, include_files=True)

    def complete_load_point_cloud(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        return self._complete_path(text, include_files=True)

    def complete_change_directory(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        return self._complete_path(text, include_dirs=True)

    # ------ Синонимы команд ------
    do_quit = do_exit
    do_q = do_exit
    do_ls = do_list_directory
    do_cd = do_change_directory
    do_lsp = do_list_point_clouds
    do_lsi = do_list_images
    do_rmi = do_remove_image

    # ------ Хуки ------
    def precmd(self, line: str) -> str:
        """Логирование перед выполнением команды"""
        if line and line.split()[0] not in ["exit", "quit", "q"]:
            print("-" * 60)
        return line

    def postloop(self) -> None:
        """Действия после завершения цикла"""
        print(f"\nСессия завершена. Всего операций: {len(self.file_history)}")

if __name__ == "__main__":
    PointCloudCLI().cmdloop()