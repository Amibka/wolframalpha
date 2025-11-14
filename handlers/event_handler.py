"""
handlers/event_handler.py - ИСПРАВЛЕННАЯ ВЕРСИЯ
КРИТИЧНО: Добавлен return результата!
"""

from logs.logger import log_call
from core.parser import get_text
from utils.error_handler import math_error_handler


@log_call
@math_error_handler
def on_enter_pressed(input_field, output_widget):
    """
    Обрабатывает нажатие Enter в поле ввода

    Args:
        input_field: QLineEdit с пользовательским вводом
        output_widget: MathOutputWidget для отображения результата

    Returns:
        result: Результат вычисления (может быть dict, list, str, SymPy объект и т.д.)
    """
    user_input = input_field.text().strip()

    if not user_input:
        return None

    print(f"DEBUG: Начинаем обработку '{user_input}'")

    try:
        # Получаем результат из парсера
        result = get_text(user_input)

        print(f"DEBUG: Получен результат: {result}")
        print(f"DEBUG: Тип результата: {type(result)}")

        # Отображаем результат в виджете
        print(f"🔍 DEBUG: Вызываем display_result с результатом: {result}")
        output_widget.display_result(result)
        print("DEBUG: Результат успешно отображен")

        # КРИТИЧНО: ВОЗВРАЩАЕМ РЕЗУЛЬТАТ!
        return result

    except Exception as e:
        import traceback
        error_message = f"❌ Ошибка обработки:\n{str(e)}\n\n{traceback.format_exc()}"
        print(f"DEBUG: Произошла ошибка: {error_message}")
        output_widget.setPlainText(error_message)

        # Возвращаем ошибку в виде dict
        return {
            'type': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }