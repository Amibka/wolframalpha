import functools
import traceback

# Импорты для обработки ошибок SymPy
try:
    from sympy.parsing.sympy_parser import TokenError
except ImportError:
    # В некоторых версиях SymPy может не быть TokenError
    TokenError = SyntaxError

try:
    from sympy.core.sympify import SympifyError
except ImportError:
    # Альтернативный импорт для старых версий
    try:
        from sympy import SympifyError
    except ImportError:
        # Если вообще нет, создаем заглушку
        class SympifyError(Exception):
            pass


def math_error_handler(func):
    """
    Декоратор для обработки математических ошибок с человекочитаемыми сообщениями

    Перехватывает:
    - Ошибки парсинга (неправильный синтаксис)
    - Математические ошибки (деление на ноль, логарифм отрицательного и т.д.)
    - Ошибки SymPy (некорректные операции)
    - Неожиданные исключения

    Возвращает понятное сообщение вместо краха программы
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)

        except (SympifyError, TokenError) as e:
            # Ошибки парсинга выражений
            expr = args[0] if args else "выражение"
            return f"Ошибка синтаксиса: не могу понять '{expr}'\n💡 Проверьте правильность записи"

        except ZeroDivisionError:
            return "Ошибка: деление на ноль"

        except ValueError as e:
            error_msg = str(e).lower()

            # Логарифм отрицательного числа
            if 'log' in error_msg or 'logarithm' in error_msg:
                return "Ошибка: логарифм отрицательного числа или нуля"

            # Корень из отрицательного
            if 'sqrt' in error_msg or 'negative' in error_msg:
                return "Ошибка: корень из отрицательного числа (используйте комплексные числа)"

            # Невалидное значение
            if 'invalid' in error_msg:
                return f"Некорректное значение: {e}"

            return f"Математическая ошибка: {e}"

        except TypeError as e:
            error_msg = str(e).lower()

            # Неправильное количество аргументов
            if 'argument' in error_msg:
                return f"Ошибка: неправильное количество аргументов\n💡 {e}"

            # Неподдерживаемая операция
            if 'unsupported' in error_msg:
                return f"Неподдерживаемая операция: {e}"

            return f"Ошибка типа данных: {e}"

        except AttributeError as e:
            return f"Ошибка: функция или метод не найдены\n💡 Возможно, опечатка в названии"

        except RecursionError:
            return "Ошибка: слишком сложное выражение (рекурсия)"

        except MemoryError:
            return "Ошибка: недостаточно памяти для вычисления"

        except KeyboardInterrupt:
            return "⚠Вычисление прервано пользователем"

        except TimeoutError:
            return "Превышено время ожидания (слишком долгие вычисления)"

        except NotImplementedError as e:
            return f"Эта операция пока не реализована: {e}"

        except Exception as e:
            # Для остальных ошибок показываем общее сообщение
            func_name = func.__name__

            # Извлекаем класс ошибки
            error_type = type(e).__name__

            # Формируем понятное сообщение
            message = f"Непредвиденная ошибка в {func_name}\n"
            message += f"Тип: {error_type}\n"
            message += f"Детали: {str(e)}"

            # Опционально: добавляем traceback для отладки (можно отключить в продакшене)
            if kwargs.get('debug', False):
                message += f"\n\nTraceback:\n{traceback.format_exc()}"

            return message

    return wrapper


# ============================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================

if __name__ == "__main__":
    from sympy import sympify, solve, diff, integrate, symbols


    @math_error_handler
    def solve_equation_safe(equation: str, variable: str = 'x'):
        """Безопасное решение уравнения"""
        var = symbols(variable)

        if '=' in equation:
            left, right = equation.split('=')
            expr = sympify(left) - sympify(right)
        else:
            expr = sympify(equation)

        return solve(expr, var)


    @math_error_handler
    def derivative_safe(expression: str, variable: str = 'x'):
        """Безопасное взятие производной"""
        expr = sympify(expression)
        var = symbols(variable)
        return diff(expr, var)


    @math_error_handler
    def integrate_safe(expression: str, variable: str = 'x'):
        """Безопасное интегрирование"""
        expr = sympify(expression)
        var = symbols(variable)
        return integrate(expr, var)


    # Тесты
    print("=" * 60)
    print("ТЕСТЫ ДЕКОРАТОРА")
    print("=" * 60)

    # Правильные выражения
    print("\nКорректные выражения:")
    print(f"solve: x^2 - 4 = 0 → {solve_equation_safe('x^2 - 4 = 0')}")
    print(f"derivative: x^3 → {derivative_safe('x**3')}")
    print(f"integrate: x^2 → {integrate_safe('x**2')}")

    # Ошибки парсинга
    print("\nОшибки синтаксиса:")
    print(solve_equation_safe('x^2 + + 4'))  # Двойной +
    print(derivative_safe('sin(x'))  # Незакрытая скобка
    print(integrate_safe('x y z'))  # Некорректный синтаксис

    # Математические ошибки
    print("\nМатематические ошибки:")
    print(solve_equation_safe('1/0'))  # Деление на ноль
    print(derivative_safe('log(-5)'))  # Логарифм отрицательного

    # Ошибки типов
    print("\nОшибки типов:")
    print(solve_equation_safe('x + "text"'))  # Смешение типов

    print("\n" + "=" * 60)