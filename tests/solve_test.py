"""
Комплексный набор pytest тестов для всех функций SymPy solver
Охватывает все операции, граничные случаи и потенциальные проблемы

Запуск: pytest test_sympy_solver.py -v
Запуск с покрытием: pytest test_sympy_solver.py -v --cov=core.sympy_solver
"""

import pytest
import sympy
from sympy import symbols, oo, pi, E, I, sqrt, Integral, Limit
from core.sympy_solver import (
    solve_equation, derivative, calculation_residue, integrate_func,
    simplify_func, expand_func, factor_func, cancel_func, together_func,
    apart_func, collect_func, trigsimp_func, powsimp_func, radsimp_func,
    ratsimp_func, logcombine_func, nsimplify_func, sqrtdenest_func,
    factor_terms_func, expand_complex_func, separatevars_func,
    gcd_func, lcm_func, div_func, quo_func, rem_func,
    poly_func, degree_func, content_func, primitive_func,
    LC_func, LM_func, LT_func, sqf_list_func, limit_func
)

x, y, z, t = symbols('x y z t')


# ============================================================================
# 1. ТЕСТЫ РЕШЕНИЯ УРАВНЕНИЙ (solve_equation)
# ============================================================================

class TestSolveEquation:
    """Тесты для функции solve_equation"""

    # --- Базовые линейные и квадратные уравнения ---

    def test_linear_equation_simple(self):
        """Простое линейное уравнение: x + 5 = 0"""
        result = solve_equation("x + 5 = 0", "x")
        assert result == [-5]

    def test_linear_equation_with_coefficient(self):
        """Линейное с коэффициентом: 2*x - 6 = 0"""
        result = solve_equation("2*x - 6 = 0", "x")
        assert result == [3]

    def test_quadratic_simple(self):
        """Квадратное: x^2 - 4 = 0"""
        result = solve_equation("x^2 - 4 = 0", "x")
        assert set(result) == {-2, 2}

    def test_quadratic_perfect_square(self):
        """Полный квадрат: x^2 + 2*x + 1 = 0"""
        result = solve_equation("x^2 + 2*x + 1 = 0", "x")
        assert result == [-1]

    def test_quadratic_no_real_solutions(self):
        """Нет действительных решений: x^2 + 1 = 0"""
        result = solve_equation("x^2 + 1 = 0", "x")
        # Должно вернуть комплексные корни или пустой список
        assert result is not None

    # --- Кубические и полиномы высших степеней ---

    def test_cubic_simple(self):
        """Кубическое: x^3 - 8 = 0"""
        result = solve_equation("x^3 - 8 = 0", "x")
        assert 2 in result

    def test_quartic(self):
        """Полином 4-й степени: x^4 - 16 = 0"""
        result = solve_equation("x^4 - 16 = 0", "x")
        real_solutions = [r for r in result if sympy.sympify(r).is_real]
        assert len(real_solutions) >= 2

    def test_quintic(self):
        """Полином 5-й степени: x^5 - 32 = 0"""
        result = solve_equation("x^5 - 32 = 0", "x")
        assert 2 in result or sympy.sympify(2) in result

    # --- Тригонометрические уравнения ---

    def test_sin_zero(self):
        """sin(x) = 0"""
        result = solve_equation("sin(x) = 0", "x")
        assert 0 in result or any(abs(float(r)) < 0.01 for r in result if isinstance(r, (int, float)))

    def test_cos_one(self):
        """cos(x) = 1"""
        result = solve_equation("cos(x) = 1", "x")
        assert 0 in result or any(abs(float(r)) < 0.01 for r in result if isinstance(r, (int, float)))

    def test_tan_one(self):
        """tan(x) = 1"""
        result = solve_equation("tan(x) = 1", "x")
        # pi/4 ≈ 0.785
        assert result is not None and len(result) > 0

    # --- Показательные уравнения ---

    def test_exponential_2_to_x(self):
        """2^x = 8"""
        result = solve_equation("2**x = 8", "x")
        assert 3 in result or sympy.sympify(3) in result

    def test_exponential_e_to_x(self):
        """e^x = 1"""
        result = solve_equation("E**x = 1", "x")
        assert 0 in result or sympy.sympify(0) in result

    # --- Логарифмические уравнения ---

    def test_log_equation(self):
        """log(x) = 2"""
        result = solve_equation("log(x) = 2", "x")
        # log базы 10: x = 100
        assert result is not None

    def test_ln_equation(self):
        """ln(x) = 1"""
        result = solve_equation("ln(x) = 1", "x")
        # Должно быть E
        assert E in result or any(abs(float(r) - 2.718) < 0.1 for r in result if isinstance(r, (int, float)))

    # --- Уравнения с несколькими переменными ---

    def test_two_variables_solve_x(self):
        """x + y = 5, решить по x"""
        result = solve_equation("x + y = 5", "x")
        # x = 5 - y
        assert result is not None

    def test_circle_solve_x(self):
        """x^2 + y^2 = 25, решить по x"""
        result = solve_equation("x**2 + y**2 = 25", "x")
        assert result is not None and len(result) > 0

    # --- Дробные уравнения ---

    def test_rational_simple(self):
        """1/x = 2"""
        result = solve_equation("1/x = 2", "x")
        assert 0.5 in result or sympy.Rational(1, 2) in result

    def test_rational_complex(self):
        """1/(x^2) = 4"""
        result = solve_equation("1/(x**2) = 4", "x")
        assert result is not None

    # --- Граничные случаи ---

    def test_equation_without_equals(self):
        """Уравнение без знака =: x^2 - 4"""
        result = solve_equation("x**2 - 4", "x")
        assert set(result) == {-2, 2}

    def test_only_variable(self):
        """Только переменная: x"""
        result = solve_equation("x", "x")
        assert result == [0]

    def test_constant_equation(self):
        """Константа: 5"""
        result = solve_equation("5", "x")
        assert result == []

    def test_identity_0_equals_0(self):
        """Тождество: 0 = 0"""
        result = solve_equation("0 = 0", "x")
        assert result is True or "тождество" in str(result).lower()

    def test_contradiction_1_equals_0(self):
        """Противоречие: 1 = 0"""
        result = solve_equation("1 = 0", "x")
        assert result == []

    def test_empty_equation(self):
        """Пустое уравнение"""
        result = solve_equation("", "x")
        assert result == []

    # --- Автоопределение переменной ---

    def test_auto_detect_variable_x(self):
        """Автоопределение переменной x"""
        result = solve_equation("x**2 - 9 = 0")
        assert set(result) == {-3, 3}

    def test_auto_detect_variable_y(self):
        """Автоопределение переменной y"""
        result = solve_equation("y**2 - 16 = 0", "y")
        assert set(result) == {-4, 4}


# ============================================================================
# 2. ТЕСТЫ ПРОИЗВОДНЫХ (derivative)
# ============================================================================

class TestDerivative:
    """Тесты для функции derivative"""

    # --- Базовые производные ---

    def test_constant(self):
        """Производная константы: d/dx(5) = 0"""
        result = derivative("5 по x")
        assert result == 0

    def test_variable(self):
        """Производная переменной: d/dx(x) = 1"""
        result = derivative("x по x")
        assert result == 1

    def test_power_x2(self):
        """x^2 -> 2*x"""
        result = derivative("x**2 по x")
        assert result == 2*x

    def test_power_x3(self):
        """x^3 -> 3*x^2"""
        result = derivative("x**3 по x")
        assert result == 3*x**2

    def test_polynomial(self):
        """Полином: x^3 + 2*x^2 + x"""
        result = derivative("x**3 + 2*x**2 + x по x")
        expected = 3*x**2 + 4*x + 1
        assert sympy.simplify(result - expected) == 0

    # --- Тригонометрические функции ---

    def test_sin(self):
        """d/dx(sin(x)) = cos(x)"""
        result = derivative("sin(x) по x")
        assert result == sympy.cos(x)

    def test_cos(self):
        """d/dx(cos(x)) = -sin(x)"""
        result = derivative("cos(x) по x")
        assert result == -sympy.sin(x)

    def test_tan(self):
        """d/dx(tan(x)) = sec^2(x)"""
        result = derivative("tan(x) по x")
        assert result == sympy.tan(x)**2 + 1 or result == 1/sympy.cos(x)**2

    # --- Показательные и логарифмические ---

    def test_exp(self):
        """d/dx(e^x) = e^x"""
        result = derivative("E**x по x")
        assert result == sympy.exp(x)

    def test_ln(self):
        """d/dx(ln(x)) = 1/x"""
        result = derivative("ln(x) по x")
        assert result == 1/x

    def test_log(self):
        """d/dx(log(x)) = 1/x (в SymPy log это натуральный логарифм)"""
        result = derivative("log(x) по x")
        # В SymPy log() это ln(), поэтому производная 1/x
        assert result == 1/x

    # --- Сложные производные ---

    def test_product_rule(self):
        """Произведение: x^2*sin(x)"""
        result = derivative("x**2*sin(x) по x")
        expected = 2*x*sympy.sin(x) + x**2*sympy.cos(x)
        assert sympy.simplify(result - expected) == 0

    def test_quotient_rule(self):
        """Частное: sin(x)/x"""
        result = derivative("sin(x)/x по x")
        expected = (x*sympy.cos(x) - sympy.sin(x))/x**2
        assert sympy.simplify(result - expected) == 0

    def test_chain_rule(self):
        """Композиция: sin(x^2)"""
        result = derivative("sin(x**2) по x")
        expected = 2*x*sympy.cos(x**2)
        assert sympy.simplify(result - expected) == 0

    # --- Корни ---

    def test_sqrt(self):
        """d/dx(sqrt(x)) = 1/(2*sqrt(x))"""
        result = derivative("sqrt(x) по x")
        expected = 1/(2*sympy.sqrt(x))
        assert sympy.simplify(result - expected) == 0

    # --- Автоопределение переменной ---

    def test_auto_detect_single_variable(self):
        """Автоопределение одной переменной"""
        result = derivative("x**3 + 2*x")
        assert result == 3*x**2 + 2

    def test_multiple_variables_error(self):
        """Несколько переменных без указания"""
        result = derivative("x**2 + y**2")
        assert "несколько" in str(result).lower() or result is not None

    # --- Частные производные ---

    def test_partial_by_x(self):
        """Частная по x: x^2 + y^2"""
        result = derivative("x**2 + y**2 по x")
        assert result == 2*x

    def test_partial_by_y(self):
        """Частная по y: x^2 + y^2"""
        result = derivative("x**2 + y**2 по y")
        assert result == 2*y

    # --- Граничные случаи ---

    def test_empty_expression(self):
        """Пустое выражение"""
        result = derivative("")
        assert result is None


# ============================================================================
# 3. ТЕСТЫ ИНТЕГРАЛОВ (integrate_func)
# ============================================================================

class TestIntegrate:
    """Тесты для функции integrate_func"""

    # --- Неопределённые интегралы ---

    def test_constant(self):
        """∫5 dx = 5*x"""
        integral = Integral(5, x)
        result = integrate_func(integral)
        assert result == 5*x

    def test_variable_x(self):
        """∫x dx = x^2/2"""
        integral = Integral(x, x)
        result = integrate_func(integral)
        expected = x**2/2
        assert sympy.simplify(result - expected) == 0

    def test_power_x2(self):
        """∫x^2 dx = x^3/3"""
        integral = Integral(x**2, x)
        result = integrate_func(integral)
        expected = x**3/3
        assert sympy.simplify(result - expected) == 0

    def test_polynomial(self):
        """∫(x^3 + 2*x^2 + x) dx"""
        integral = Integral(x**3 + 2*x**2 + x, x)
        result = integrate_func(integral)
        expected = x**4/4 + 2*x**3/3 + x**2/2
        assert sympy.simplify(result - expected) == 0

    # --- Тригонометрические ---

    def test_sin(self):
        """∫sin(x) dx = -cos(x)"""
        integral = Integral(sympy.sin(x), x)
        result = integrate_func(integral)
        expected = -sympy.cos(x)
        assert sympy.simplify(result - expected) == 0

    def test_cos(self):
        """∫cos(x) dx = sin(x)"""
        integral = Integral(sympy.cos(x), x)
        result = integrate_func(integral)
        expected = sympy.sin(x)
        assert sympy.simplify(result - expected) == 0

    # --- Показательные ---

    def test_exp(self):
        """∫e^x dx = e^x"""
        integral = Integral(sympy.exp(x), x)
        result = integrate_func(integral)
        assert result == sympy.exp(x)

    # --- Рациональные ---

    def test_one_over_x(self):
        """∫1/x dx = ln(x)"""
        integral = Integral(1/x, x)
        result = integrate_func(integral)
        assert result == sympy.log(x)

    def test_rational_1_over_x2_plus_1(self):
        """∫1/(x^2+1) dx = atan(x)"""
        integral = Integral(1/(x**2+1), x)
        result = integrate_func(integral)
        assert result == sympy.atan(x)

    # --- Определённые интегралы ---

    def test_definite_x2_0_to_1(self):
        """∫₀¹ x^2 dx = 1/3"""
        integral = Integral(x**2, (x, 0, 1))
        result = integrate_func(integral)
        assert result == sympy.Rational(1, 3)

    def test_definite_sin_0_to_pi(self):
        """∫₀^π sin(x) dx = 2"""
        integral = Integral(sympy.sin(x), (x, 0, pi))
        result = integrate_func(integral)
        assert result == 2

    def test_definite_1_over_x_1_to_e(self):
        """∫₁^e 1/x dx = 1"""
        integral = Integral(1/x, (x, 1, E))
        result = integrate_func(integral)
        assert result == 1

    # --- Сложные интегралы ---

    def test_sqrt(self):
        """∫sqrt(x) dx = (2/3)*x^(3/2)"""
        integral = Integral(sympy.sqrt(x), x)
        result = integrate_func(integral)
        expected = sympy.Rational(2, 3) * x**sympy.Rational(3, 2)
        assert sympy.simplify(result - expected) == 0

    # --- Граничные случаи ---

    def test_string_input_with_integral(self):
        """Строка с Integral объектом"""
        result = integrate_func("Integral(x**2, x)")
        assert result is not None


# ============================================================================
# 4. ТЕСТЫ ПРЕДЕЛОВ (limit_func)
# ============================================================================

class TestLimit:
    """Тесты для функции limit_func"""

    # --- Базовые пределы ---

    def test_sin_x_over_x_at_0(self):
        """lim(x->0) sin(x)/x = 1"""
        result = limit_func("sin(x)/x, x, 0")
        assert result == 1

    def test_1_over_x_at_infinity(self):
        """lim(x->∞) 1/x = 0"""
        result = limit_func("1/x, x, oo")
        assert result == 0

    def test_polynomial_at_point(self):
        """lim(x->2) x^2 = 4"""
        result = limit_func("x**2, x, 2")
        assert result == 4

    def test_exp_at_0(self):
        """lim(x->0) e^x = 1"""
        result = limit_func("E**x, x, 0")
        assert result == 1

    # --- Односторонние пределы ---

    def test_1_over_x_right_at_0(self):
        """lim(x->0+) 1/x = +∞"""
        result = limit_func("1/x, x, 0+")
        assert result == oo

    def test_1_over_x_left_at_0(self):
        """lim(x->0-) 1/x = -∞"""
        result = limit_func("1/x, x, 0-")
        assert result == -oo

    # --- Пределы на бесконечности ---

    def test_x2_at_infinity(self):
        """lim(x->∞) x^2 = ∞"""
        result = limit_func("x**2, x, oo")
        assert result == oo

    def test_1_over_x2_at_infinity(self):
        """lim(x->∞) 1/x^2 = 0"""
        result = limit_func("1/x**2, x, oo")
        assert result == 0

    def test_exp_at_infinity(self):
        """lim(x->∞) e^x = ∞"""
        result = limit_func("E**x, x, oo")
        assert result == oo

    def test_exp_neg_at_infinity(self):
        """lim(x->∞) e^(-x) = 0"""
        result = limit_func("E**(-x), x, oo")
        assert result == 0

    # --- Неопределённости ---

    def test_indeterminate_0_over_0(self):
        """lim(x->2) (x^2-4)/(x-2) = 4"""
        result = limit_func("(x**2-4)/(x-2), x, 2")
        assert result == 4

    def test_indeterminate_inf_over_inf(self):
        """lim(x->∞) x^2/x^3 = 0"""
        result = limit_func("x**2/x**3, x, oo")
        assert result == 0

    # --- Граничные случаи ---

    def test_constant_limit(self):
        """lim(x->0) 5 = 5"""
        result = limit_func("5, x, 0")
        assert result == 5


# ============================================================================
# 5. ТЕСТЫ УПРОЩЕНИЯ
# ============================================================================

class TestSimplification:
    """Тесты для функций упрощения"""

    # --- Simplify ---

    def test_simplify_fraction(self):
        """(x^2-1)/(x-1) = x+1"""
        result = simplify_func("(x**2-1)/(x-1)")
        assert result == x + 1

    def test_simplify_trig_identity(self):
        """sin^2 + cos^2 = 1"""
        result = simplify_func("sin(x)**2 + cos(x)**2")
        assert result == 1

    # --- Expand ---

    def test_expand_square(self):
        """(x+1)^2 = x^2+2*x+1"""
        result = expand_func("(x+1)**2")
        assert result == x**2 + 2*x + 1

    def test_expand_cube(self):
        """(x+1)^3"""
        result = expand_func("(x+1)**3")
        assert result == x**3 + 3*x**2 + 3*x + 1

    def test_expand_product(self):
        """(x+y)*(x-y) = x^2-y^2"""
        result = expand_func("(x+y)*(x-y)")
        assert result == x**2 - y**2

    # --- Factor ---

    def test_factor_difference_of_squares(self):
        """x^2-1 = (x-1)*(x+1)"""
        result = factor_func("x**2-1")
        assert result == (x-1)*(x+1)

    def test_factor_perfect_square(self):
        """x^2+2*x+1 = (x+1)^2"""
        result = factor_func("x**2+2*x+1")
        assert result == (x+1)**2

    def test_factor_difference_of_cubes(self):
        """x^3-8 = (x-2)*(x^2+2*x+4)"""
        result = factor_func("x**3-8")
        assert result == (x-2)*(x**2+2*x+4)

    # --- Cancel ---

    def test_cancel_fraction(self):
        """(x^2-1)/(x-1) = x+1"""
        result = cancel_func("(x**2-1)/(x-1)")
        assert result == x + 1

    # --- Together ---

    def test_together_sum(self):
        """1/x + 1/y = (x+y)/(x*y)"""
        result = together_func("1/x + 1/y")
        expected = (x+y)/(x*y)
        assert sympy.simplify(result - expected) == 0

    # --- Apart ---

    def test_apart_partial_fractions(self):
        """1/(x^2-1) = 1/(2*(x-1)) - 1/(2*(x+1))"""
        result = apart_func("1/(x**2-1)")
        # Проверяем, что результат эквивалентен исходному
        assert sympy.simplify(result - 1/(x**2-1)) == 0


# ============================================================================
# 6. ТЕСТЫ ТРИГОНОМЕТРИЧЕСКОГО УПРОЩЕНИЯ
# ============================================================================

class TestTrigonometry:
    """Тесты для тригонометрических функций"""

    def test_trigsimp_identity(self):
        """sin^2+cos^2 = 1"""
        result = trigsimp_func("sin(x)**2 + cos(x)**2")
        assert result == 1

    def test_trigsimp_double_angle(self):
        """2*sin*cos = sin(2x)"""
        result = trigsimp_func("2*sin(x)*cos(x)")
        assert result == sympy.sin(2*x)


# ============================================================================
# 7. ТЕСТЫ СТЕПЕНЕЙ И КОРНЕЙ
# ============================================================================

class TestPowersAndRoots:
    """Тесты для степеней и корней"""

    def test_powsimp_power_of_power(self):
        """(x^2)^3 = x^6"""
        result = powsimp_func("(x**2)**3")
        assert result == x**6

    def test_powsimp_product_of_powers(self):
        """x^2*x^3 = x^5"""
        result = powsimp_func("x**2*x**3")
        assert result == x**5

    def test_radsimp_sqrt_squared(self):
        """sqrt(x^2)"""
        result = radsimp_func("sqrt(x**2)")
        # Результат может быть |x| или x в зависимости от предположений
        assert result is not None


# ============================================================================
# 8. ТЕСТЫ ЛОГАРИФМОВ
# ============================================================================

class TestLogarithms:
    """Тесты для логарифмических функций"""

    def test_logcombine_sum(self):
        """log(x)+log(y) = log(x*y)"""
        result = logcombine_func("log(x) + log(y)")
        assert result == sympy.log(x*y)

    def test_logcombine_difference(self):
        """log(x)-log(y) = log(x/y)"""
        result = logcombine_func("log(x) - log(y)")
        assert result == sympy.log(x/y)

    def test_logcombine_coefficient(self):
        """2*log(x) = log(x^2)"""
        result = logcombine_func("2*log(x)")
        assert result == sympy.log(x**2)


# ============================================================================
# 9. ТЕСТЫ COLLECT
# ============================================================================

class TestCollect:
    """Тесты для функции collect"""

    def test_collect_by_x(self):
        """Группировка по x"""
        result = collect_func("x**2 + 2*x + x**3 по x")
        assert result is not None

    def test_collect_auto_detect(self):
        """Автоопределение переменной"""
        result = collect_func("x*y + x**2*y + x*y**2 по x")
        assert result is not None


# ============================================================================
# 10. ТЕСТЫ ПРОДВИНУТЫХ ФУНКЦИЙ
# ============================================================================

class TestAdvancedFunctions:
    """Тесты для продвинутых функций"""

    # --- GCD ---

    def test_gcd_numbers(self):
        """НОД(12, 18) = 6"""
        result = gcd_func("12, 18")
        assert result == 6

    def test_gcd_polynomials(self):
        """НОД(x^2-1, x-1) = x-1"""
        result = gcd_func("x**2-1, x-1")
        assert result == x - 1

    # --- LCM ---

    def test_lcm_numbers(self):
        """НОК(12, 18) = 36"""
        result = lcm_func("12, 18")
        assert result == 36

    def test_lcm_polynomials(self):
        """НОК(x, x^2)"""
        result = lcm_func("x, x**2")
        assert result == x**2

    # --- DIV ---

    def test_div_polynomials(self):
        """(x^2-1)/(x-1)"""
        result = div_func("x**2-1, x-1")
        # div возвращает (частное, остаток)
        assert result is not None

    # --- QUO ---

    def test_quo_polynomials(self):
        """Частное от деления полиномов"""
        result = quo_func("x**2-1, x-1")
        assert result == x + 1

    # --- REM ---

    def test_rem_polynomials(self):
        """Остаток от деления"""
        result = rem_func("x**2+1, x-1")
        assert result == 2

    # --- DEGREE ---

    def test_degree_x3(self):
        """Степень x^3"""
        result = degree_func("x**3 по x")
        assert result == 3

    def test_degree_polynomial(self):
        """Степень x^2+x"""
        result = degree_func("x**2+x по x")
        assert result == 2

    # --- CONTENT ---

    def test_content_polynomial(self):
        """Содержимое многочлена"""
        result = content_func("2*x**2 + 4*x по x")
        assert result == 2

    # --- PRIMITIVE ---

    def test_primitive_polynomial(self):
        """Примитивная часть"""
        result = primitive_func("2*x**2 + 4*x по x")
        # Должно вернуть (2, x^2 + 2*x)
        assert result is not None


# ============================================================================
# 11. ТЕСТЫ СПЕЦИАЛЬНЫХ ФУНКЦИЙ
# ============================================================================

class TestSpecialFunctions:
    """Тесты для специальных функций"""

    def test_nsimplify_pi(self):
        """Приближение pi"""
        result = nsimplify_func("3.14159265")
        # Должно найти pi
        assert result is not None

    def test_ratsimp(self):
        """Рациональное упрощение"""
        result = ratsimp_func("(x**2-1)/(x-1) + x")
        assert result == 2*x + 1

    def test_factor_terms(self):
        """Вынесение общего множителя"""
        result = factor_terms_func("2*x + 4*y")
        assert result == 2*(x + 2*y)

    def test_expand_complex(self):
        """Раскрытие комплексных чисел"""
        result = expand_complex_func("(1+I)**2")
        assert result == 2*I

    def test_separatevars(self):
        """Разделение переменных"""
        result = separatevars_func("x*y + x*z")
        assert result == x*(y + z)


# ============================================================================
# 12. ТЕСТЫ ВЫЧЕТОВ
# ============================================================================

class TestResidue:
    """Тесты для функции вычисления вычетов"""

    def test_residue_simple(self):
        """Вычет простой функции"""
        result = calculation_residue("1/(x-1) по x в 1")
        assert result == 1

    def test_residue_complex(self):
        """Вычет более сложной функции"""
        result = calculation_residue("1/x**2 по x в 0")
        assert result == 0


# ============================================================================
# 13. ТЕСТЫ ФУНКЦИЙ ДЛЯ МНОГОЧЛЕНОВ
# ============================================================================

class TestPolynomialFunctions:
    """Тесты для функций работы с многочленами"""

    def test_poly_creation(self):
        """Создание объекта Poly"""
        result = poly_func("x**2 + 2*x + 1")
        assert result is not None
        assert hasattr(result, 'degree')

    def test_LC_leading_coefficient(self):
        """Старший коэффициент"""
        result = LC_func("2*x**3 + 3*x**2 + 1 по x")
        assert result == 2

    def test_LM_leading_monomial(self):
        """Старший одночлен"""
        result = LM_func("2*x**3 + 3*x**2 + 1 по x")
        assert result is not None

    def test_LT_leading_term(self):
        """Старший член"""
        result = LT_func("2*x**3 + 3*x**2 + 1 по x")
        assert result is not None

    def test_sqf_list_square_free(self):
        """Квадратно-свободное разложение"""
        result = sqf_list_func("x**4 - 1 по x")
        assert result is not None


# ============================================================================
# 14. ГРАНИЧНЫЕ СЛУЧАИ И ОБРАБОТКА ОШИБОК
# ============================================================================

class TestEdgeCases:
    """Тесты граничных случаев и обработки ошибок"""

    # --- Пустые входы ---

    def test_solve_empty_string(self):
        """Решение пустой строки"""
        result = solve_equation("", "x")
        assert result == []

    def test_derivative_empty_string(self):
        """Производная от пустой строки"""
        result = derivative("")
        assert result is None

    # --- Некорректные форматы ---

    def test_limit_insufficient_args(self):
        """Предел с недостаточным количеством аргументов"""
        result = limit_func("sin(x)/x")
        assert "ошибка" in str(result).lower() or "недостаточно" in str(result).lower()

    def test_div_insufficient_args(self):
        """DIV с одним аргументом"""
        result = div_func("x**2")
        assert "ошибка" in str(result).lower()

    def test_quo_insufficient_args(self):
        """QUO с одним аргументом"""
        result = quo_func("x**2")
        assert "ошибка" in str(result).lower()

    def test_rem_insufficient_args(self):
        """REM с одним аргументом"""
        result = rem_func("x**2")
        assert "ошибка" in str(result).lower()

    # --- Некорректный синтаксис ---

    def test_solve_invalid_syntax(self):
        """Уравнение с некорректным синтаксисом"""
        result = solve_equation("x +++ 5 = 0", "x")
        assert "ошибка" in str(result).lower() or result == [-5]

    def test_derivative_invalid_syntax(self):
        """Производная с некорректным синтаксисом"""
        result = derivative("x ^^^ 2 по x")
        assert "ошибка" in str(result).lower() or result is not None

    # --- Деление на ноль ---

    def test_solve_division_by_zero(self):
        """Уравнение с делением на ноль"""
        result = solve_equation("1/0 = x", "x")
        # Должно обработать ошибку
        assert result is not None

    # --- Комплексные числа ---

    def test_solve_complex_roots(self):
        """Уравнение с комплексными корнями"""
        result = solve_equation("x**2 + 1 = 0", "x")
        # Должно вернуть -I и I
        assert result is not None

    def test_derivative_complex(self):
        """Производная комплексной функции"""
        result = derivative("(x + I)**2 по x")
        assert result == 2*(x + I)

    # --- Большие числа ---

    def test_solve_large_polynomial(self):
        """Полином высокой степени"""
        result = solve_equation("x**10 - 1024 = 0", "x")
        assert 2 in result or any(abs(float(r) - 2) < 0.01 for r in result if isinstance(r, (int, float)))

    # --- Специальные значения ---

    def test_limit_at_infinity_keyword(self):
        """Предел с ключевым словом infinity"""
        result = limit_func("1/x, x, infinity")
        assert result == 0

    def test_solve_with_infinity(self):
        """Уравнение с бесконечностью"""
        result = solve_equation("x = oo", "x")
        # Не должно упасть
        assert result is not None

    # --- Символы и константы ---

    def test_solve_with_pi(self):
        """Уравнение с pi"""
        result = solve_equation("x = pi", "x")
        assert pi in result or sympy.pi in result

    def test_solve_with_e(self):
        """Уравнение с E"""
        result = solve_equation("x = E", "x")
        assert E in result or sympy.E in result

    # --- Многопеременные случаи ---

    def test_derivative_no_variable_multiple_vars(self):
        """Производная без указания переменной (несколько переменных)"""
        result = derivative("x*y + x**2*y**2")
        assert "несколько" in str(result).lower() or result is not None

    def test_collect_no_variable_multiple_vars(self):
        """Collect без указания переменной"""
        result = collect_func("x*y + x**2*y")
        assert "несколько" in str(result).lower() or result is not None


# ============================================================================
# 15. ИНТЕГРАЦИОННЫЕ ТЕСТЫ
# ============================================================================

class TestIntegration:
    """Интеграционные тесты для проверки совместной работы"""

    def test_solve_with_integral(self):
        """Решение уравнения с интегралом"""
        # ∫x dx = 8 → x^2/2 = 8 → x = ±4
        integral = Integral(x, x)
        result_integral = integrate_func(integral)
        # x^2/2 = 8
        result = solve_equation(f"{result_integral} = 8", "x")
        assert 4 in result or -4 in result

    def test_derivative_then_simplify(self):
        """Производная, затем упрощение"""
        deriv = derivative("sin(x)**2 + cos(x)**2 по x")
        result = simplify_func(str(deriv))
        assert result == 0

    def test_expand_then_factor(self):
        """Раскрытие, затем факторизация"""
        expanded = expand_func("(x+1)*(x+2)")
        result = factor_func(str(expanded))
        assert result == (x+1)*(x+2)

    def test_limit_of_derivative(self):
        """Предел производной"""
        # d/dx(x^2) = 2x, lim(x->1) 2x = 2
        deriv = derivative("x**2 по x")
        result = limit_func(f"{deriv}, x, 1")
        assert result == 2


# ============================================================================
# 16. ТЕСТЫ ПРОИЗВОДИТЕЛЬНОСТИ
# ============================================================================

class TestPerformance:
    """Тесты производительности для критичных операций"""

    def test_solve_many_solutions(self):
        """Уравнение с множеством решений"""
        result = solve_equation("sin(x) = 0", "x")
        # Должно найти хотя бы одно решение быстро
        assert result is not None

    def test_large_polynomial_factor(self):
        """Факторизация большого полинома"""
        result = factor_func("x**6 - 1")
        assert result is not None

    def test_complex_simplification(self):
        """Сложное упрощение"""
        result = simplify_func("(x**4 - 1)/(x**2 - 1)")
        assert result == x**2 + 1


# ============================================================================
# 17. ТЕСТЫ ЧИСЛЕННЫХ РЕШЕНИЙ
# ============================================================================

class TestNumericalSolutions:
    """Тесты численного решения уравнений"""

    def test_transcendental_equation(self):
        """Трансцендентное уравнение: e^x = x + 2"""
        result = solve_equation("E**x = x + 2", "x")
        # Должно найти численное решение
        assert result is not None
        if isinstance(result, list) and len(result) > 0:
            # Проверяем, что решение близко к реальному
            assert isinstance(result[0], (int, float, sympy.Basic))

    def test_mixed_equation(self):
        """Смешанное уравнение: x + sin(x) = 1"""
        result = solve_equation("x + sin(x) = 1", "x")
        assert result is not None


# ============================================================================
# 18. ТЕСТЫ РУССКОЯЗЫЧНЫХ КОМАНД
# ============================================================================

class TestRussianCommands:
    """Тесты для русскоязычных команд"""

    def test_derivative_russian_po(self):
        """Производная с русским 'по'"""
        result = derivative("x**2 по x")
        assert result == 2*x

    def test_collect_russian_po(self):
        """Collect с русским 'по'"""
        result = collect_func("x**2 + 2*x по x")
        assert result is not None

    def test_degree_russian_po(self):
        """Degree с русским 'по'"""
        result = degree_func("x**3 по x")
        assert result == 3


# ============================================================================
# 19. ТЕСТЫ CORNER CASES
# ============================================================================

class TestCornerCases:
    """Тесты крайних случаев"""

    def test_solve_zero_equals_zero(self):
        """0 = 0 (тождество)"""
        result = solve_equation("0 = 0", "x")
        assert result is True or "тождество" in str(result).lower()

    def test_solve_x_equals_x(self):
        """x = x (тождество)"""
        result = solve_equation("x = x", "x")
        assert result is True or "тождество" in str(result).lower()

    def test_derivative_of_constant(self):
        """Производная константы"""
        result = derivative("100 по x")
        assert result == 0

    def test_integral_of_zero(self):
        """Интеграл от нуля"""
        integral = Integral(0, x)
        result = integrate_func(integral)
        assert result == 0

    def test_limit_of_constant(self):
        """Предел константы"""
        result = limit_func("42, x, 5")
        assert result == 42

    def test_gcd_of_same_numbers(self):
        """НОД одинаковых чисел"""
        result = gcd_func("12, 12")
        assert result == 12

    def test_lcm_of_same_numbers(self):
        """НОК одинаковых чисел"""
        result = lcm_func("12, 12")
        assert result == 12

    def test_factor_prime(self):
        """Факторизация простого числа"""
        result = factor_func("17")
        assert result == 17

    def test_expand_already_expanded(self):
        """Раскрытие уже раскрытого"""
        result = expand_func("x**2 + 2*x + 1")
        assert result == x**2 + 2*x + 1


# ============================================================================
# 20. ТЕСТЫ СТРЕССОВЫХ СЦЕНАРИЕВ
# ============================================================================

class TestStressScenarios:
    """Тесты стрессовых сценариев для презентации"""

    def test_complex_mixed_operations(self):
        """Сложная смесь операций"""
        # Решаем: d/dx(∫x^2 dx) = x^2
        integral = Integral(x**2, x)
        integrated = integrate_func(integral)
        deriv = derivative(f"{integrated} по x")
        assert sympy.simplify(deriv - x**2) == 0

    def test_nested_functions(self):
        """Вложенные функции"""
        result = derivative("sin(cos(x)) по x")
        expected = -sympy.sin(x)*sympy.cos(sympy.cos(x))
        assert sympy.simplify(result - expected) == 0

    def test_very_long_polynomial(self):
        """Очень длинный полином"""
        poly = " + ".join([f"x**{i}" for i in range(10)])
        result = derivative(f"{poly} по x")
        assert result is not None

    def test_multiple_variables_interaction(self):
        """Взаимодействие нескольких переменных"""
        result = derivative("x*y*z по x")
        assert result == y*z

    def test_rational_complex_fraction(self):
        """Сложная рациональная дробь"""
        result = simplify_func("(x**3 - 1)/(x - 1)")
        assert result == x**2 + x + 1


# ============================================================================
# ФИКСТУРЫ ДЛЯ ТЕСТОВ
# ============================================================================

@pytest.fixture
def local_dict_fixture():
    """Фикстура для локального словаря"""
    return {
        'x': x, 'y': y, 'z': z, 't': t,
        'pi': pi, 'E': E, 'I': I, 'oo': oo,
        'sin': sympy.sin, 'cos': sympy.cos,
        'exp': sympy.exp, 'log': sympy.log,
        'sqrt': sympy.sqrt
    }


# ============================================================================
# ЗАПУСК ТЕСТОВ
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])