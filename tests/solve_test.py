import pytest
from core.parser import get_text


class TestSolveFunction:
    """Тесты для функции solve с различными переменными и синтаксисом"""

    # ========================================================================
    # БАЗОВЫЕ ТЕСТЫ С АНГЛИЙСКИМ СИНТАКСИСОМ (at)
    # ========================================================================

    def test_solve_linear_at_x(self):
        """Линейное уравнение с переменной x через 'at'"""
        result = get_text("solve 4x + 1 = 10 at x")
        assert result == [9/4] or str(result) == "[9/4]"

    def test_solve_linear_at_y(self):
        """Линейное уравнение с переменной y через 'at'"""
        result = get_text("solve 4x + y = 10 at y")
        assert "10 - 4*x" in str(result) or result == [10 - 4*sympy.Symbol('x')]

    def test_solve_linear_at_z(self):
        """Линейное уравнение с переменной z через 'at'"""
        result = get_text("solve 3z - 5 = 7 at z")
        assert result == [4] or str(result) == "[4]"

    def test_solve_quadratic_at_x(self):
        """Квадратное уравнение через 'at'"""
        result = get_text("solve x^2 - 5x + 6 = 0 at x")
        assert len(result) == 2
        assert 2 in result and 3 in result

    def test_solve_quadratic_at_t(self):
        """Квадратное уравнение с переменной t через 'at'"""
        result = get_text("solve t^2 + 4t + 4 = 0 at t")
        assert result == [-2] or str(result) == "[-2]"

    # ========================================================================
    # ТЕСТЫ С РУССКИМ СИНТАКСИСОМ (по)
    # ========================================================================

    def test_solve_linear_po_x(self):
        """Линейное уравнение с переменной x через 'по'"""
        result = get_text("solve 2x + 3 = 11 по x")
        assert result == [4] or str(result) == "[4]"

    def test_solve_linear_po_y(self):
        """Линейное уравнение с переменной y через 'по'"""
        result = get_text("solve 4x + y = 10 по y")
        assert "10 - 4*x" in str(result)

    def test_solve_linear_po_z(self):
        """Линейное уравнение с переменной z через 'по'"""
        result = get_text("solve 5z + 2 = 17 по z")
        assert result == [3] or str(result) == "[3]"

    def test_solve_quadratic_po_x(self):
        """Квадратное уравнение через 'по'"""
        result = get_text("solve x^2 - 9 = 0 по x")
        assert len(result) == 2
        assert -3 in result and 3 in result

    def test_solve_quadratic_po_t(self):
        """Квадратное уравнение с переменной t через 'по'"""
        result = get_text("solve t^2 - 2t + 1 = 0 по t")
        assert result == [1] or str(result) == "[1]"

    # ========================================================================
    # ТЕСТЫ С АЛЬТЕРНАТИВНЫМ СИНТАКСИСОМ (by, in)
    # ========================================================================

    def test_solve_linear_by_x(self):
        """Линейное уравнение через 'by'"""
        result = get_text("solve 3x - 7 = 2 by x")
        assert result == [3] or str(result) == "[3]"

    def test_solve_linear_in_y(self):
        """Линейное уравнение через 'in'"""
        result = get_text("solve 2y + 5 = 15 in y")
        assert result == [5] or str(result) == "[5]"

    # ========================================================================
    # ТЕСТЫ С РАЗНЫМИ ПЕРЕМЕННЫМИ
    # ========================================================================

    def test_solve_with_variable_a(self):
        """Уравнение с переменной a"""
        result = get_text("solve 2a + 3 = 9 по a")
        assert result == [3] or str(result) == "[3]"

    def test_solve_with_variable_b(self):
        """Уравнение с переменной b"""
        result = get_text("solve 4b - 8 = 0 at b")
        assert result == [2] or str(result) == "[2]"

    def test_solve_with_variable_m(self):
        """Уравнение с переменной m"""
        result = get_text("solve m^2 - 16 = 0 по m")
        assert len(result) == 2
        assert -4 in result and 4 in result

    def test_solve_with_variable_n(self):
        """Уравнение с переменной n"""
        result = get_text("solve 3n + 7 = 22 at n")
        assert result == [5] or str(result) == "[5]"

    def test_solve_with_variable_p(self):
        """Уравнение с переменной p"""
        result = get_text("solve p^2 + 6p + 9 = 0 по p")
        assert result == [-3] or str(result) == "[-3]"

    # ========================================================================
    # ТЕСТЫ БЕЗ ЯВНОГО УКАЗАНИЯ ПЕРЕМЕННОЙ
    # ========================================================================

    def test_solve_without_variable_single_var(self):
        """Уравнение без указания переменной (только одна переменная в выражении)"""
        result = get_text("solve x^2 - 4 = 0")
        assert len(result) == 2
        assert -2 in result and 2 in result

    def test_solve_expression_only(self):
        """Решение выражения (приравнивание к нулю)"""
        result = get_text("solve 2x + 6")
        assert result == [-3] or str(result) == "[-3]"

    # ========================================================================
    # СЛОЖНЫЕ ВЫРАЖЕНИЯ
    # ========================================================================

    def test_solve_rational_equation(self):
        """Рациональное уравнение"""
        result = get_text("solve x/2 + 3 = 7 at x")
        assert result == [8] or str(result) == "[8]"

    def test_solve_with_multiple_terms(self):
        """Уравнение с несколькими членами"""
        result = get_text("solve 2x + 3y - 5 = 0 по x")
        assert "y" in str(result)
        assert "(5 - 3*y)/2" in str(result) or "5/2 - 3*y/2" in str(result)

    def test_solve_cubic_equation(self):
        """Кубическое уравнение"""
        result = get_text("solve x^3 - 8 = 0 at x")
        assert 2 in result or any(abs(float(r) - 2) < 0.01 for r in result)

    def test_solve_with_parentheses(self):
        """Уравнение со скобками"""
        result = get_text("solve 2(x + 3) = 14 по x")
        assert result == [4] or str(result) == "[4]"

    def test_solve_exponential(self):
        """Уравнение с показательной функцией"""
        result = get_text("solve 2^x = 8 at x")
        assert result == [3] or any(abs(float(r) - 3) < 0.01 for r in result)

    # ========================================================================
    # КРАЕВЫЕ СЛУЧАИ
    # ========================================================================

    def test_solve_identity(self):
        """Тождество (бесконечно много решений)"""
        result = get_text("solve x = x at x")
        # Может вернуть True, все действительные числа или специальное сообщение
        assert result is not None

    def test_solve_contradiction(self):
        """Противоречие (нет решений)"""
        result = get_text("solve x + 1 = x at x")
        # Должен вернуть пустой список или сообщение об отсутствии решений
        assert result == [] or "решения" in str(result).lower()

    def test_solve_zero_coefficient(self):
        """Уравнение с нулевым коэффициентом"""
        result = get_text("solve 0x + 5 = 5 at x")
        # x может быть любым числом
        assert result is not None

    # ========================================================================
    # ТЕСТЫ С ТРИГОНОМЕТРИЧЕСКИМИ ФУНКЦИЯМИ
    # ========================================================================

    def test_solve_sin_equation(self):
        """Уравнение с синусом"""
        result = get_text("solve sin(x) = 0 at x")
        # Должен найти хотя бы один корень (0, pi, -pi, ...)
        assert result is not None
        assert len(result) > 0

    def test_solve_cos_equation(self):
        """Уравнение с косинусом"""
        result = get_text("solve cos(x) = 1 at x")
        assert 0 in result or any(abs(float(r)) < 0.01 for r in result)

    # ========================================================================
    # ТЕСТЫ С ЛОГАРИФМАМИ
    # ========================================================================

    def test_solve_log_equation(self):
        """Уравнение с логарифмом"""
        result = get_text("solve log(x) = 2 at x")
        # log(x) = 2 => x = e^2 или 10^2 в зависимости от основания
        assert result is not None
        assert len(result) > 0

    # ========================================================================
    # СМЕШАННЫЕ ТЕСТЫ
    # ========================================================================

    def test_solve_russian_and_english_variables(self):
        """Уравнение с русским синтаксисом и английскими переменными"""
        result = get_text("solve 3x + 2y = 12 по x")
        assert "y" in str(result)

    def test_solve_complex_expression(self):
        """Сложное выражение с несколькими операциями"""
        result = get_text("solve (x + 2)^2 - 9 = 0 at x")
        assert len(result) == 2
        assert -5 in result and 1 in result

    def test_solve_fraction_coefficients(self):
        """Уравнение с дробными коэффициентами"""
        result = get_text("solve x/3 + 2 = 5 по x")
        assert result == [9] or str(result) == "[9]"

    # ========================================================================
    # ТЕСТЫ НА ОШИБКИ И ВАЛИДАЦИЮ
    # ========================================================================

    def test_solve_empty_expression(self):
        """Пустое выражение"""
        result = get_text("solve")
        assert "ошибка" in str(result).lower() or result == 'Пожалуйста, введите математическое выражение.'

    def test_solve_invalid_syntax(self):
        """Неверный синтаксис"""
        result = get_text("solve 2x ++ 3 = 5 at x")
        # Должен обработать ошибку
        assert result is not None

    def test_solve_multiple_variables_no_specification(self):
        """Несколько переменных без указания какую решать"""
        result = get_text("solve x + y = 5")
        # Должен либо решить по x (по умолчанию), либо попросить уточнить
        assert result is not None


if __name__ == "__main__":
    # Запуск тестов с подробным выводом
    pytest.main([__file__, "-v", "--tb=short"])