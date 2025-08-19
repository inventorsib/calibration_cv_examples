import cmd
import shlex
import os
import glob
from datetime import datetime

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import cv2
from pyapriltags import Detector

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

class FileCLI(cmd.Cmd):
    prompt = ">>> "
    intro = "Файловый менеджер CLI. Напишите 'help' для списка команд"
    
    def __init__(self):
        super().__init__()
        self.work_dir = os.getcwd()
        self.file_history = []
        self.image_history = {}  # {путь: изображение}
        self.points_history = {}  # {путь: изображение}
        self.current_image = None
        self.current_path = None

    # ------ Команды для работы с облаком точек ------
    def do_load_points(self, arg):
        """Загрузить файл c облаком точек: loadpoints <путь/к/файлу>"""
        if not arg:
            print("Ошибка: Укажите путь к файлу")
            return
        
        try:
            # Обработка путей с пробелами
            args = shlex.split(arg)
            if len(args) > 1:
                print("Ошибка: Слишком много аргументов. Используйте кавычки для путей с пробелами")
                return
                
            path = Path(args[0]).expanduser().absolute()
            if not os.path.exists(path):
                print(f"Ошибка: Файл не существует: {path}")
                return
                
            if os.path.isdir(path):
                print(f"Ошибка: {path} является директорией")
                return
            
            valid_ext = ['.txt','.csv']
            if path.suffix.lower() not in valid_ext:
                print(f"Ошибка: Неподдерживаемый формат изображения: {path.suffix}")
                return
            
            data = np.loadtxt(path)
            data_size = np.shape(data)
            if data_size[1]<3:
                print(f"Ошибка: неверный формат хранения точек")
                return

            self.points_history[str(path)] = data
            self.current_data = data
            self.current_path = path
    
            self.file_history.append(path)

            print(f"Размер: {data_size[0]}x{data_size[1]}")
            
        except Exception as e:
            print(f"Ошибка загрузки: {str(e)}")

    def do_calc_maps(self, arg):
        try:
            args = shlex.split(arg)

            points, path = self._get_points(arg[0])

            height = int(args[1])
            width = int(args[2])
            focus = int(args[3])
            fx = focus 
            fy = focus
            cx = width/2
            cy = height/2        
            num_in_row = np.shape(points)[1]
            depth_map = np.full((height, width), np.inf)
            reflect_map = np.full((height, width), np.inf)
            for iterator, row in enumerate(points):
                
                if num_in_row==4:
                    xpt, ypt, zpt, r = row[:4] 
                else:
                    xpt, ypt, zpt = row[:3]

                z = xpt
                x = ypt
                y = zpt

                if num_in_row==4:
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
                    if z < depth_map[v_idx, u_idx]:
                        depth_map[v_idx, u_idx] = z
                    if num_in_row==4:
                        if r < reflect_map[v_idx, u_idx]:
                            reflect_map[v_idx, u_idx] = r

            depth_map[np.isinf(depth_map)] = 0  
            depth_map[np.isnan(depth_map)] = 0  
            self.depth_visual = np.copy(255 * depth_map / np.max(depth_map))

            if num_in_row==4:
                reflect_map[np.isinf(reflect_map)] = 0  
                reflect_map[np.isnan(reflect_map)] = 0  
                self.reflect_visual = np.copy(255 * reflect_map / np.max(reflect_map))

            name = str(Path(path).name)
            # Сохранение в истории
            self.image_history["DEPTH_MAP_"+name] = depth_map
            self.current_image = depth_map
            self.current_path = "DEPTH_MAP_"+name
            image_size = np.shape(depth_map)
            self.file_history.append("DEPTH_MAP_"+name)
            print("DEPTH_MAP_"+name+f"  Размер: {image_size[0]}x{image_size[1]}")

            if num_in_row==4:
                # Сохранение в истории
                self.image_history["REFLECT_MAP_"+name] = reflect_map
                self.current_image = reflect_map
                self.current_path = "REFLECT_MAP_"+name
                image_size = np.shape(depth_map)
                self.file_history.append("REFLECT_MAP_"+name)
                print("REFLECT_MAP_"+name+f"  Размер: {image_size[0]}x{image_size[1]}")

        except Exception as e:
            print(e)

    '''         
    def do_show_depthmap(self, arg):
        plt.imshow(self.depth_visual)
        plt.show()

    def do_show_reflectmap(self, arg):
        plt.imshow(self.reflect_visual)
        plt.show()
    '''

    def do_histpoints(self, arg):
            """Показать историю загруженных файлов точек: histpoints"""
            if not self.points_history:
                print("История файлов точек пуста")
                return
                
            print("\nИстория загруженных файлов точек:")
            for i, path in enumerate(self.points_history.keys(), 1):
                print(f"{i:2d}. {Path(path).name}  - {path}")
    
    # ------ Команды для работы с изображениями ------
    def do_load_image(self, arg):
        """Загрузить изображение: loadimg <путь/к/файлу>"""
        if not arg:
            print("Ошибка: Укажите путь к файлу")
            return
        
        try:
            args = shlex.split(arg)
            if len(args) > 1:
                print("Ошибка: Слишком много аргументов")
                return
                
            path = Path(args[0]).expanduser().absolute()
            
            if not path.exists():
                print(f"Ошибка: Файл не существует: {path}")
                return
                
            if path.is_dir():
                print(f"Ошибка: {path} является директорией")
                return
                
            # Проверка формата файла
            valid_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            if path.suffix.lower() not in valid_ext:
                print(f"Ошибка: Неподдерживаемый формат изображения: {path.suffix}")
                return
                
            # Загрузка изображения
            img = cv2.imread(str(path))
            if img is None:
                print(f"Ошибка: Не удалось загрузить изображение")
                return
                
            # Сохранение в истории
            self.image_history[str(path)] = img
            self.current_image = img
            self.current_path = path
        
            image_size = np.shape(img)

            self.file_history.append(path)
            print(f"Размер: {image_size[0]}x{image_size[1]}")

        except Exception as e:
            print(f"Ошибка загрузки: {str(e)}")

    def do_show_image(self, arg):
        try:
            image = self._get_image(arg)

            plt.imshow(image)
            plt.show()
        
        except Exception as e:
            print("Загрузите изображение")

    def do_detect_tag(self, arg):
        try:
            args = shlex.split(arg)
            image = self._get_image(args[0])

            if len(args)==1:
                tag_families = 'tag16h5'
            else:
                tag_families = args[1]

            
            print("tag families:", tag_families)
            detector = Detector(
                families=tag_families,  # Тип меток (по умолчанию)
                nthreads=1,           # Количество потоков
                quad_decimate=1.0,    # Уменьшение разрешения изображения
                quad_sigma=1.0,       # Размытие изображения
                decode_sharpening=0.7,
                refine_edges=0.1        # Точность определения границ
            )

            if len(image.shape) == 3 and image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.astype(np.uint8)
                image = self._safe_convert_gray_to_bgr(image)
                #///image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            tags_on_image = np.copy(image)

            # TODO: use tags_corners
            tags_corners = self._getTagPositionInFrame(detector, gray, tags_on_image)

            plt.imshow(tags_on_image)
            plt.show()
            
        
        except Exception as e:
            print("Ошибка", e)

    def do_histimg(self, arg):
        """Показать историю загруженных изображений: histimg"""
        if not self.image_history:
            print("История изображений пуста")
            return
            
        print("\nИстория загруженных изображений:")
        for i, path in enumerate(self.image_history.keys(), 1):
            img = self.image_history[path]
            h, w = img.shape[:2]
            print(f"{i:2d}. {Path(path).name} ({w}x{h}) - {path}")

    def do_delimg(self, arg):
            """Удалить изображение из истории: delimg [индекс|путь]"""
            if not arg:
                print("Укажите индекс или путь изображения")
                return
                
            try:
                # Удаление по индексу
                index = int(arg) - 1
                if 0 <= index < len(self.image_history):
                    path = list(self.image_history.keys())[index]
                    del self.image_history[path]
                    print(f"Изображение удалено из истории: {Path(path).name}")
                else:
                    print("Ошибка: Неверный индекс")
            except ValueError:
                # Удаление по пути
                path = str(Path(arg).expanduser().absolute())
                if path in self.image_history:
                    del self.image_history[path]
                    print(f"Изображение удалено из истории: {Path(path).name}")
                else:
                    print("Ошибка: Изображение не найдено в истории")

    # ------ Команды для работы с файлами ------
    def do_dir(self, arg):
        """Показать содержимое директории: dir [путь]"""
        path = arg if arg else self.work_dir
        try:
            if not os.path.exists(path):
                print(f"Директория не существует: {path}")
                return
                
            print(f"\nСодержимое {os.path.abspath(path)}:")
            for item in os.listdir(path):
                full_path = os.path.join(path, item)
                if os.path.isdir(full_path):
                    print(f"{item}/")
                else:
                    size = os.path.getsize(full_path)
                    print(f"{item} ({size} байт)")
            print()
            
        except Exception as e:
            print(f"Ошибка: {str(e)}")

    def do_cd(self, arg):
        """Сменить директорию: cd <путь>"""
        if not arg:
            print(f"Текущая директория: {self.work_dir}")
            return
            
        try:
            new_dir = os.path.abspath(os.path.join(self.work_dir, arg))
            if not os.path.exists(new_dir):
                print(f"Директория не существует: {new_dir}")
                return
                
            if not os.path.isdir(new_dir):
                print(f"Ошибка: {new_dir} не является директорией")
                return
                
            self.work_dir = new_dir
            os.chdir(self.work_dir)
            print(f"Текущая директория: {self.work_dir}")
        except Exception as e:
            print(f"Ошибка: {str(e)}")

    # ------ Автодополнение ------
    def complete_load_image(self, text, line, begidx, endidx):
        """Автодополнение для команды loadimg"""
        return self._complete_path(text, include_files=True)
    
    def complete_load_points(self, text, line, begidx, endidx):
        """Автодополнение для команды loadpoins"""
        return self._complete_path(text, include_files=True)

    def complete_cd(self, text, line, begidx, endidx):
        """Автодополнение для команды cd"""
        return self._complete_path(text, include_dirs=True)

    def _complete_path(self, text, include_files=False, include_dirs=False):
        """Универсальный метод автодополнения путей"""
        if not text:
            text = ""
            
        # Подготовка пути для поиска
        search_path = os.path.join(self.work_dir, text)
        dir_path = os.path.dirname(search_path)
        base_name = os.path.basename(search_path)
        
        if not os.path.exists(dir_path):
            return []
        
        # Поиск совпадений
        matches = []
        for name in os.listdir(dir_path):
            full_path = os.path.join(dir_path, name)
            
            # Фильтрация по типу
            if include_files and os.path.isfile(full_path) and name.startswith(base_name):
                matches.append(name)
            elif include_dirs and os.path.isdir(full_path) and name.startswith(base_name):
                matches.append(name + "/")
        
        return matches

    # ------ История операций ------
    def do_history(self, arg):
        """Показать историю загруженных файлов: history"""
        if not self.file_history:
            print("История пуста")
            return
            
        print("\nИстория загрузок:")
        for i, path in enumerate(self.file_history, 1):
            print(f"{i:2d}. {path}")
        print()

    def do_clear_history(self, arg):
        """Очистить историю: clear_history"""
        self.file_history = []
        print("История очищена")

    # ------ Системные команды ------
    def do_exit(self, arg):
        """Выйти из приложения: exit"""
        print("Сессия завершена")
        return True

    # ------ Специальные методы ------

    def precmd(self, line):
        """Логирование команд"""
        if line and line.split()[0] not in ["exit", "quit"]:
            print("-" * 50)
        return line

    def postloop(self):
        """Финал работы"""
        print(f"\nВсего загружено файлов: {len(self.file_history)}")


    def _display_image(self, img, title):
            """Отображение изображения с помощью OpenCV"""
            # Масштабирование для больших изображений
            h, w = img.shape[:2]
            max_size = 800
            
            if w > max_size or h > max_size:
                scale = max_size / max(w, h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img_resized = cv2.resize(img, (new_w, new_h))
            else:
                img_resized = img
            
            # Показ изображения
            cv2.imshow(title, img_resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def _list_images(self):
        """Список изображений в истории"""
        if not self.image_history:
            print("История изображений пуста")
            return
            
        print("\nДоступные изображения (используйте 'showimg <номер>'):")
        for i, path in enumerate(self.image_history.keys(), 1):
            img = self.image_history[path]
            h, w = img.shape[:2]
            print(f"{i:2d}. {Path(path).name} ({w}x{h})")


    def _getTagSizeInFrame(self, corners):
        side_lengths = []
        for i in range(4):
            # Берем соседние точки (замыкаем контур: последняя с первой)
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            # Вычисляем Евклидово расстояние между точками
            length = np.linalg.norm(p1 - p2)
            side_lengths.append(length)
        
        # Средний размер стороны метки
        avg_size = np.min(side_lengths)
        return avg_size

    #! Координаты углов на изображении
    def _getTagPositionInFrame(self, detector, gray, image):

        results = detector.detect(gray)

        tags_corners = []
        corners = []
        for i, detection in enumerate(results):
            # Координаты углов в формате [ [x1, y1], [x2, y2], [x3, y3], [x4, y4] ]
            
            corners = detection.corners.astype(int)

            if self._getTagSizeInFrame(corners)<25:
                continue

            tags_corners.append(corners)
            
            # Визуализация (опционально)
            for j in range(4):
                start = tuple(corners[j])
                end = tuple(corners[(j + 1) % 4])
                center = tuple(detection.center.astype(int))
                cv2.line(image, start, end, colors[i], 2)
                cv2.putText(image, str(j), start, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
                cv2.putText(image, "id="+str(detection.tag_id), center, cv2.FONT_HERSHEY_SIMPLEX, 1, colors[i], 2)

        return tags_corners

    def _get_image(self, arg):
        """Получить загруженное изображение: [индекс|путь]"""
        if not arg:
            self._list_images()
            return
        img = []
        try:
            # Попытка использовать как индекс
            index = int(arg) - 1
            if 0 <= index < len(self.image_history):
                path = list(self.image_history.keys())[index]
                img = self.image_history[path]
            else:
                print("Ошибка: Неверный индекс")
                return
        except ValueError:
            # Использование как путь
            if not arg.endswith(".txt"):
                path = str(Path(arg).expanduser().absolute())
            else:
                path = arg 

            if path not in self.image_history:
                print(f"Ошибка: Изображение не найдено в истории")
                return
            img = self.image_history[path]
        
        return img
    
    def _get_points(self, arg):
        """Получить облако точек: [индекс|путь]"""
        if not arg:
            self._list_images()
            return
        points = []
        try:
            # Попытка использовать как индекс
            index = int(arg) - 1
            if 0 <= index < len(self.points_history):
                path = list(self.points_history.keys())[index]
                points = self.points_history[path]
            else:
                print("Ошибка: Неверный индекс")
                return
        except ValueError:
            # Использование как путь
            path = str(Path(arg).expanduser().absolute())
            if path not in self.points_history:
                print(f"Ошибка: Облако точек не найдено в истории")
                return
            points = self.points_history[path]
        
        return points, path


    def _safe_convert_gray_to_bgr(self, image):
        """
        Безопасное преобразование grayscale в BGR с обработкой разных типов данных
        """
        if image is None:
            return None
        
        # Проверяем тип данных
        if image.dtype == np.float64:
            # Преобразуем 64-bit float в 8-bit
            # Нормализуем к диапазону [0, 255]
            if image.max() <= 1.0:  # если значения в [0, 1]
                image_8bit = (image * 255).astype(np.uint8)
            else:
                image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            return cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2BGR)
        
        elif image.dtype == np.float32:
            # 32-bit float
            if image.max() <= 1.0:
                image_8bit = (image * 255).astype(np.uint8)
            else:
                image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            return cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2BGR)
        
        else:
            # Для других типов (uint8, uint16) пробуем прямое преобразование
            try:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            except cv2.error:
                # Если не получается, нормализуем и преобразуем
                image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                return cv2.cvtColor(image_normalized, cv2.COLOR_GRAY2BGR)

    # Синонимы
    do_quit = do_exit
    do_q = do_exit

if __name__ == "__main__":
    FileCLI().cmdloop()