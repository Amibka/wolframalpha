"""
Нормализация входного текста перед токенизацией
"""
import re


class TextNormalizer:
    def __init__(self):
        # Замены символов
        self.replacements = {
            'х': 'x',  # русская х → английская x
            '×': '*',
            '÷': '/',
            '−': '-',
            '·': '*',
            '^': '**',
            '√': 'sqrt',
        }

        # Замены слов
        self.word_replacements = {
            # Русские варианты
            'икс': 'x',
            'игрек': 'y',
            'зет': 'z',
            'квадрат': '**2',
            'куб': '**3',
            'в степени': '**',
            'умножить на': '*',
            'плюс': '+',
            'минус': '-',
            'делить на': '/',
            'равно': '=',

            # Английские варианты
            'squared': '**2',
            'cubed': '**3',
            'times': '*',
            'plus': '+',
            'minus': '-',
            'divided by': '/',
            'equals': '=',
        }

    def normalize(self, text: str) -> str:
        """
        Нормализует текст

        Примеры:
        "2х + 1" → "2*x + 1"
        "x в квадрате" → "x**2"
        "√x" → "sqrt(x)"
        """
        text = text.lower().strip()

        # Замена символов
        for old, new in self.replacements.items():
            text = text.replace(old, new)

        # Замена слов
        for old, new in self.word_replacements.items():
            text = re.sub(r'\b' + old + r'\b', new, text)

        # Добавляем * между числом и переменной: 2x → 2*x
        text = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', text)

        # Убираем лишние пробелы
        text = re.sub(r'\s+', ' ', text).strip()

        return text