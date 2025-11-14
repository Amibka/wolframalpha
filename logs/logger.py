import logging
import os
from functools import wraps


# === Базовая настройка логирования ===
def setup_logger(name="wolfram_pyqt", log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Формат сообщений
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # === Вывод в консоль ===
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # === Запись в файл (app.log) ===
    file_handler = logging.FileHandler(os.path.join(log_dir, "app.log"), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # === Запись в отдельный txt-файл (app.txt) ===
    txt_handler = logging.FileHandler(os.path.join(log_dir, "app.txt"), encoding="utf-8")
    txt_handler.setLevel(logging.DEBUG)
    txt_handler.setFormatter(formatter)
    logger.addHandler(txt_handler)

    # === Запись ошибок отдельно (errors.log) ===
    error_handler = logging.FileHandler(os.path.join(log_dir, "errors.log"), encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    return logger


# === Инициализируем логгер ===
logger = setup_logger()


# === Универсальный декоратор логирования ===
def log_call(func):
    """
    Универсальный декоратор для логирования вызовов функций.
    Логирует:
      - входные аргументы
      - успешное выполнение
      - ошибки с трассировкой
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        arg_str = ", ".join([repr(a) for a in args] +
                            [f"{k}={v!r}" for k, v in kwargs.items()])
        logger.info(f"Вызов {func.__name__}({arg_str})")

        try:
            result = func(*args, **kwargs)
            logger.info(f"{func.__name__} успешно выполнена. Результат: {result}")
            return result
        except Exception as e:
            logger.exception(f"Ошибка в {func.__name__}: {e}")
            raise

    return wrapper
