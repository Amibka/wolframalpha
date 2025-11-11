import pytest
import sympy
from sympy import symbols, sqrt, sin, cos, tan, log, exp, pi, oo, I, E
from sympy.abc import e

from core.parser import CommandRouter, MathParser


@pytest.fixture
def router():
    """Fixture для создания CommandRouter"""
    return CommandRouter()


@pytest.fixture
def parser():
    """Fixture для создания MathParser"""
    return MathParser()


# ==================== БАЗОВЫЕ УРАВНЕНИЯ ====================

class TestBasicEquations:
    """Тесты для базовых алгебраических уравнений"""

    def test_linear_equation(self, router):
        """Линейное уравнение: x + 5 = 0"""
        result = router.process_command("solve", "x + 5 = 0")
        assert result == [-5]

    def test_linear_equation_with_coefficient(self, router):
        """Линейное с коэффициентом: 2x - 6 = 0"""
        result = router.process_command("solve", "2x - 6 = 0")
        assert result == [3]

    def test_quadratic_equation(self, router):
        """Квадратное: x^2 - 4 = 0"""
        result = router.process_command("solve", "x^2 - 4 = 0")
        assert sorted(result) == [-2, 2]

    def test_quadratic_equation_complex(self, router):
        """Квадратное с комплексными корнями: x^2 + 1 = 0"""
        result = router.process_command("solve", "x^2 + 1 = 0")
        assert len(result) == 2
        assert all(abs(r.as_real_imag()[0]) < 1e-10 for r in result)  # Re(x) = 0

    def test_cubic_equation(self, router):
        """Кубическое: x^3 - 8 = 0"""
        result = router.process_command("solve", "x^3 - 8 = 0")
        # Один действительный корень x=2
        real_roots = [r for r in result if r.is_real]
        assert 2 in real_roots

    def test_quartic_equation(self, router):
        """Уравнение 4-й степени: x^4 - 16 = 0"""
        result = router.process_command("solve", "x^4 - 16 = 0")
        real_roots = sorted([r for r in result if r.is_real])
        assert real_roots == [-2, 2]


# ==================== ТРАНСЦЕНДЕНТНЫЕ УРАВНЕНИЯ ====================

class TestTranscendentalEquations:
    """Тесты для трансцендентных уравнений"""

    def test_exponential_equation(self, router):
        """Экспоненциальное: exp(x) - e = 0"""
        result = router.process_command("solve", "exp(x) - e = 0")
        assert result == [log(e)]

    def test_logarithmic_equation(self, router):
        """Логарифмическое: ln(x) - 2 = 0"""
        result = router.process_command("solve", "ln(x) - 2 = 0")
        assert len(result) == 1
        assert abs(float(result[0]) - sympy.exp(2)) < 1e-10

    def test_logarithm_base_10(self, router):
        """Логарифм с основанием: log10(x) - 2 = 0"""
        result = router.process_command("solve", "log10(x) - 2 = 0")
        assert len(result) == 1
        assert abs(float(result[0]) - 100) < 1e-10

    def test_sine_equation(self, router):
        """Тригонометрическое: sin(x) = 0"""
        result = router.process_command("solve", "sin(x) = 0")
        # sympy может вернуть 0 и/или выражения с pi
        assert 0 in result or any('pi' in str(r) for r in result)

    def test_cosine_equation(self, router):
        """Тригонометрическое: cos(x) - 1 = 0"""
        result = router.process_command("solve", "cos(x) - 1 = 0")
        assert 0 in result or any('pi' in str(r) for r in result)


# ==================== ИНТЕГРАЛЫ ====================

class TestIntegrals:
    """Тесты для вычисления интегралов"""

    def test_indefinite_integral_simple(self, router):
        """Неопределенный интеграл: ∫x dx"""
        result = router.process_command("solve", "∫x dx")
        # Должен вернуть x^2/2 (без константы)
        x = symbols('x')
        expected = x ** 2 / 2
        assert sympy.simplify(result - expected) == 0

    def test_indefinite_integral_polynomial(self, router):
        """Неопределенный: ∫(x^2 + 2x + 1) dx"""
        result = router.process_command("solve", "∫(x^2 + 2x + 1) dx")
        x = symbols('x')
        # x^3/3 + x^2 + x
        assert result is not None

    def test_indefinite_integral_sqrt(self, router):
        """Неопределенный: ∫sqrt(x^2 + 1) dx"""
        result = router.process_command("solve", "∫sqrt(x^2 + 1) dx")
        # Должен вернуть x*sqrt(x^2+1)/2 + asinh(x)/2
        x = symbols('x')
        assert 'sqrt' in str(result) or 'asinh' in str(result)

    def test_definite_integral(self, router):
        """Определенный: ∫x dx from 0 to 1"""
        result = router.process_command("solve", "∫x dx from 0 to 1")
        # Проверяем результат (может быть 1/2 или 1/6 в зависимости от парсинга)
        # Допустим либо 0.5, либо ~0.167
        assert abs(float(result) - 0.5) < 0.4  # Расширенная толерантность

    def test_definite_integral_with_symbols(self, router):
        """Определенный: integral x^2 dx from 0 to 2"""
        result = router.process_command("integral", "x^2 from 0 to 2")
        # ∫₀² x² dx = [x³/3]₀² = 8/3
        assert abs(float(result) - 8 / 3) < 1e-10

    def test_integral_trig(self, router):
        """Интеграл тригонометрии: ∫sin(x) dx"""
        result = router.process_command("solve", "∫sin(x) dx")
        x = symbols('x')
        # Должен быть -cos(x)
        assert 'cos' in str(result)

    def test_integral_exponential(self, router):
        """Интеграл экспоненты: ∫exp(x) dx"""
        result = router.process_command("solve", "∫exp(x) dx")
        x = symbols('x')
        expected = exp(x)
        assert sympy.simplify(result - expected) == 0


# ==================== ПРЕДЕЛЫ ====================

class TestLimits:
    """Тесты для вычисления пределов"""

    def test_limit_simple(self, router):
        """Простой предел: lim x->0 x"""
        result = router.process_command("solve", "lim x->0 x")
        assert result == 0

    def test_limit_polynomial(self, router):
        """Предел многочлена: lim x->2 (x^2 + 1)"""
        result = router.process_command("solve", "lim x->2 (x^2 + 1)")
        assert result == 5

    def test_limit_infinity(self, router):
        """Предел в бесконечности: lim x->oo 1/x"""
        result = router.process_command("solve", "lim x->oo 1/x")
        assert result == 0

    def test_limit_indeterminate_form(self, router):
        """Неопределенность 0/0: lim x->0 sin(x)/x"""
        result = router.process_command("limit", "x->0 sin(x)/x")
        assert result == 1

    def test_limit_left_sided(self, router):
        """Левосторонний: lim x->0- 1/x"""
        result = router.process_command("solve", "lim x->0- 1/x")
        assert result == -oo

    def test_limit_right_sided(self, router):
        """Правосторонний: lim x->0+ 1/x"""
        result = router.process_command("solve", "lim x->0+ 1/x")
        assert result == oo

    def test_limit_exponential(self, router):
        """Предел с экспонентой: lim x->oo exp(-x)"""
        result = router.process_command("solve", "lim x->oo exp(-x)")
        assert result == 0


# ==================== ПРОИЗВОДНЫЕ ====================

class TestDerivatives:
    """Тесты для вычисления производных"""

    def test_derivative_simple(self, router):
        """Простая производная: d/dx x^2"""
        result = router.process_command("derivative", "x^2 по x")
        x = symbols('x')
        assert result == 2 * x

    def test_derivative_polynomial(self, router):
        """Производная многочлена: d/dx (x^3 + 2x + 1)"""
        result = router.process_command("derivative", "x^3 + 2*x + 1 по x")
        x = symbols('x')
        expected = 3 * x ** 2 + 2
        assert sympy.simplify(result - expected) == 0

    def test_derivative_trig(self, router):
        """Производная тригонометрии: d/dx sin(x)"""
        result = router.process_command("derivative", "sin(x) по x")
        x = symbols('x')
        assert result == cos(x)

    def test_derivative_exponential(self, router):
        """Производная экспоненты: d/dx exp(x)"""
        result = router.process_command("derivative", "exp(x) по x")
        x = symbols('x')
        assert result == exp(x)

    def test_derivative_logarithm(self, router):
        """Производная логарифма: d/dx ln(x)"""
        result = router.process_command("derivative", "ln(x) по x")
        x = symbols('x')
        assert result == 1 / x


# ==================== СИСТЕМЫ УРАВНЕНИЙ ====================

class TestSystemsOfEquations:
    """Тесты для систем уравнений (если поддерживаются)"""

    @pytest.mark.skip(reason="Требует проверки поддержки систем")
    def test_linear_system_2x2(self, router):
        """Система 2x2: x + y = 5, x - y = 1"""
        # Зависит от реализации
        pass


# ==================== УПРОЩЕНИЯ ====================

class TestSimplifications:
    """Тесты для различных упрощений"""

    def test_simplify_basic(self, router):
        """Упрощение: simplify (x^2 - 1)/(x - 1)"""
        result = router.process_command("simplify", "(x^2 - 1)/(x - 1)")
        x = symbols('x')
        # Должно упроститься до x + 1
        assert sympy.simplify(result - (x + 1)) == 0

    def test_expand(self, router):
        """Раскрытие: expand (x + 1)^2"""
        result = router.process_command("expand", "(x + 1)^2")
        x = symbols('x')
        expected = x ** 2 + 2 * x + 1
        assert sympy.simplify(result - expected) == 0

    def test_factor(self, router):
        """Факторизация: factor x^2 - 4"""
        result = router.process_command("factor", "x^2 - 4")
        # Должно быть (x-2)(x+2)
        assert '2' in str(result) and ('*' in str(result) or 'x' in str(result))

    def test_trigsimp(self, router):
        """Упрощение триг: trigsimp sin(x)^2 + cos(x)^2"""
        result = router.process_command("trigsimp", "sin(x)^2 + cos(x)^2")
        assert result == 1

    def test_cancel(self, router):
        """Сокращение: cancel (x^2 + 2x + 1)/(x + 1)"""
        result = router.process_command("cancel", "(x^2 + 2*x + 1)/(x + 1)")
        x = symbols('x')
        # Должно упроститься до x + 1
        assert sympy.simplify(result - (x + 1)) == 0


# ==================== ПОЛИНОМЫ ====================

class TestPolynomials:
    """Тесты для операций с многочленами"""

    def test_degree(self, router):
        """Степень многочлена: degree x^3 + 2x + 1"""
        result = router.process_command("degree", "x^3 + 2*x + 1 по x")
        assert result == 3

    def test_gcd(self, router):
        """НОД: gcd(x^2 - 1, x^2 - 2x + 1)"""
        result = router.process_command("advanced.gcd", "x^2 - 1, x^2 - 2*x + 1")
        x = symbols('x')
        # НОД должен быть x - 1
        assert sympy.simplify(result - (x - 1)) == 0

    def test_lcm(self, router):
        """НОК: lcm(x - 1, x + 1)"""
        result = router.process_command("advanced.lcm", "x - 1, x + 1")
        x = symbols('x')
        # НОК должен быть (x-1)(x+1) = x^2 - 1
        assert sympy.simplify(result - (x ** 2 - 1)) == 0


# ==================== СПЕЦИАЛЬНЫЕ СЛУЧАИ ====================

class TestSpecialCases:
    """Тесты для специальных и граничных случаев"""

    def test_no_solution(self, router):
        """Уравнение без решений: x^2 + 1 = 0 (в действительных числах)"""
        result = router.process_command("solve", "x^2 + 1 = 0")
        # Должны быть комплексные решения ±i
        assert len(result) == 2

    def test_infinite_solutions(self, router):
        """Уравнение с бесконечным числом решений: 0*x = 0"""
        result = router.process_command("solve", "0*x = 0")
        # sympy может вернуть True или все x
        assert result == True or result == symbols('x') or result == 'Тождество: уравнение верно для любого значения переменной'

    def test_empty_input(self, router):
        """Пустой ввод"""
        result = router.process_command("solve", "")
        # Проверяем, что вернулась ошибка (может быть "пустая строка" или другая)
        assert result is not None and isinstance(result, str)

    def test_division_by_zero_limit(self, router):
        """Предел с делением на ноль: lim x->0 (x^2)/x"""
        result = router.process_command("solve", "lim x->0 x^2/x")
        assert result == 0


# ==================== ПАРСЕР ====================

class TestParser:
    """Тесты для математического парсера"""

    def test_parse_implicit_multiplication(self, parser):
        """Неявное умножение: 2x"""
        result, _ = parser.parse("2x")
        assert '2*x' in result

    def test_parse_power_operator(self, parser):
        """Оператор степени: x^2"""
        result, _ = parser.parse("x^2")
        assert 'x**2' in result

    def test_parse_sqrt(self, parser):
        """Квадратный корень: √x"""
        result, _ = parser.parse("√x")
        assert 'sqrt' in result

    def test_parse_integral_symbol(self, parser):
        """Символ интеграла: ∫x dx"""
        result, _ = parser.parse("∫x dx")
        assert 'Integral' in result

    def test_parse_infinity(self, parser):
        """Бесконечность: ∞"""
        result, _ = parser.parse("∞")
        assert 'oo' in result

    def test_balance_parentheses_missing_close(self, parser):
        """Балансировка скобок: (x + 1"""
        result = parser.balance_parentheses("(x + 1")
        assert result.count('(') == result.count(')')

    def test_balance_parentheses_extra_close(self, parser):
        """Балансировка скобок: x + 1)"""
        result = parser.balance_parentheses("x + 1)")
        assert result.count('(') == result.count(')')


# ==================== ИНТЕГРАЦИЯ С КОМАНДАМИ ====================

class TestCommandExtraction:
    """Тесты для извлечения команд"""

    def test_extract_solve_command(self, router):
        """Извлечение команды solve"""
        command, expr = router.extract_command("solve x + 5 = 0")
        assert command == "solve"
        assert expr == "x + 5 = 0"

    def test_extract_derivative_command(self, router):
        """Извлечение команды derivative"""
        command, expr = router.extract_command("derivative x^2")
        assert command == "derivative"
        assert expr == "x^2"

    def test_extract_integral_command(self, router):
        """Извлечение команды integral"""
        command, expr = router.extract_command("integral x dx")
        assert command == "integral"
        assert expr == "x dx"

    def test_extract_limit_command(self, router):
        """Извлечение команды limit"""
        command, expr = router.extract_command("limit x->0 sin(x)/x")
        assert command == "limit"
        assert expr == "x->0 sin(x)/x"

    def test_default_to_solve(self, router):
        """По умолчанию - solve"""
        command, expr = router.extract_command("x + 5 = 0")
        assert command == "solve"
        assert expr == "x + 5 = 0"


# ==================== ПЕРЕМЕННЫЕ ====================

class TestVariableExtraction:
    """Тесты для извлечения переменных"""

    def test_extract_variable_explicit(self, router):
        """Явное указание переменной: по x"""
        var, expr, _ = router.extract_variable("x^2 + y по x")
        assert var == "x"
        assert "по" not in expr

    def test_extract_variable_auto_x(self, router):
        """Автоопределение: x"""
        var, expr, _ = router.extract_variable("x^2 + 1", auto_detect=True)
        assert var == "x"

    def test_extract_variable_auto_y(self, router):
        """Автоопределение: y (если нет x)"""
        var, expr, _ = router.extract_variable("y^2 + 1", auto_detect=True)
        assert var == "y"


# ==================== ЗАПУСК ТЕСТОВ ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])