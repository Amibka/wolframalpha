# utils/math_formatter.py
from sympy import pretty, N, simplify


def format_solutions(result):
    """
    Форматирует результаты в красивый текст с использованием Unicode математики.
    """
    if isinstance(result, str):
        return result

    try:
        if isinstance(result, (list, tuple)) and len(result) > 0:
            # Список решений
            lines = []
            lines.append("РЕШЕНИЯ:")
            lines.append("=" * 80)
            lines.append("")

            for i, sol in enumerate(result, 1):
                # Упрощаем решение
                try:
                    sol_simplified = simplify(sol)
                except:
                    sol_simplified = sol

                # Pretty print символьного выражения
                pretty_sol = pretty(sol_simplified, use_unicode=True)

                # Разбиваем на строки если многострочное
                sol_lines = pretty_sol.split('\n')

                # Выводим решение
                lines.append(f"x{subscript(i)} = {sol_lines[0]}")
                for line in sol_lines[1:]:
                    lines.append(f"     {line}")

                # Численное приближение
                try:
                    numeric = N(sol_simplified, 8)  # Увеличил точность до 8 знаков
                    numeric_pretty = pretty(numeric, use_unicode=True)
                    numeric_lines = numeric_pretty.split('\n')

                    lines.append(f"   ≈ {numeric_lines[0]}")
                    for line in numeric_lines[1:]:
                        lines.append(f"     {line}")
                except:
                    pass

                # Пустая строка между решениями
                if i < len(result):
                    lines.append("")

            lines.append("")
            lines.append("=" * 80)

            return '\n'.join(lines)

        else:
            # Одиночный результат
            lines = []
            lines.append("РЕЗУЛЬТАТ:")
            lines.append("=" * 80)
            lines.append("")

            # Pretty print
            try:
                result_simplified = simplify(result)
            except:
                result_simplified = result

            pretty_result = pretty(result_simplified, use_unicode=True)
            result_lines = pretty_result.split('\n')

            for line in result_lines:
                lines.append(line)

            # Численное приближение
            try:
                numeric = N(result_simplified, 8)
                try:
                    numeric_val = complex(numeric)
                    result_val = complex(result_simplified)
                    show_numeric = abs(numeric_val - result_val) > 1e-10
                except:
                    show_numeric = str(numeric) != str(result_simplified)

                if show_numeric:
                    lines.append("")
                    numeric_pretty = pretty(numeric, use_unicode=True)
                    numeric_lines = numeric_pretty.split('\n')
                    lines.append(f"≈ {numeric_lines[0]}")
                    for line in numeric_lines[1:]:
                        lines.append(f"  {line}")
            except:
                pass

            lines.append("")
            lines.append("=" * 80)

            return '\n'.join(lines)

    except Exception as e:
        print(f"❌ Ошибка форматирования: {e}")
        import traceback
        traceback.print_exc()
        return str(result)


def subscript(n):
    """Конвертирует число в нижний индекс Unicode."""
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(n).translate(subscripts)
