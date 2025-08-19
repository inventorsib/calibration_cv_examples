import cmd
import shlex
import os
import cv2
import numpy as np
from pathlib import Path

class ImageCLI(cmd.Cmd):
    prompt = "🖼️> "
    intro = "Image Manager CLI. Напишите 'help' для списка команд"
    
    def __init__(self):
        super().__init__()
        self.work_dir = os.getcwd()
        self.image_history = {}  # {путь: изображение}
        self.current_image = None
        self.current_path = None

    # ------ Команды для работы с изображениями ------
    def do_loadimg(self, arg):
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
            
            # Вывод информации
            h, w = img.shape[:2]
            channels = "цветное" if len(img.shape) == 3 else "ч/б"
            print(f"✅ Изображение загружено: {path.name}")
            print(f"Размер: {w}x{h} пикселей, {channels}")
            print(f"Путь: {path}")
            
        except Exception as e:
            print(f"Ошибка загрузки: {str(e)}")

    def do_showimg(self, arg):
        """Показать изображение из истории: showimg [индекс|путь]"""
        if not arg:
            self._list_images()
            return
            
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
            path = str(Path(arg).expanduser().absolute())
            if path not in self.image_history:
                print(f"Ошибка: Изображение не найдено в истории")
                return
            img = self.image_history[path]
        
        # Показ изображения
        self._display_image(img, path)

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

    # ------ Автодополнение ------
    def complete_loadimg(self, text, line, begidx, endidx):
        """Автодополнение для загрузки изображений"""
        return self._complete_image_paths(text)
    
    def complete_showimg(self, text, line, begidx, endidx):
        """Автодополнение для показа изображений"""
        return self._complete_image_history(text)
    
    def complete_delimg(self, text, line, begidx, endidx):
        """Автодополнение для удаления изображений"""
        return self._complete_image_history(text)

    def _complete_image_paths(self, text):
        """Автодополнение путей к изображениям"""
        path = Path(text or ".").expanduser()
        dir_path = str(path.parent) if text else str(Path.cwd())
        
        if not Path(dir_path).is_dir():
            return []
        
        # Допустимые расширения
        valid_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        completions = []
        for item in Path(dir_path).iterdir():
            if not text or item.name.startswith(path.name):
                if item.is_dir():
                    completions.append(f"{item.name}/")
                elif item.is_file() and item.suffix.lower() in valid_ext:
                    completions.append(str(item))
        return completions

    def _complete_image_history(self, text):
        """Автодополнение истории изображений"""
        if not text:
            return list(self.image_history.keys())
        
        # Поиск по имени файла или пути
        completions = []
        for path in self.image_history:
            if text in path or text in Path(path).name:
                completions.append(path)
        return completions

    # ------ Вспомогательные методы ------
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

    # ------ Системные команды ------
    def do_exit(self, arg):
        """Выйти из приложения: exit"""
        cv2.destroyAllWindows()
        print("Сессия завершена")
        return True

    # Синонимы
    do_quit = do_exit
    do_q = do_exit

if __name__ == "__main__":
    ImageCLI().cmdloop()