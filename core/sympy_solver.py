import numpy as np
import sympy
import re
from sympy import symbols, sympify, Eq, solve, nsolve, diff, simplify, Integral, Symbol, oo, limit
from sympy.abc import (
    a, b, c, d, e, f, g, h, i, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z
)

from logs.logger import log_call


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ЧИСЛЕННОГО РЕШЕНИЯ
# ============================================================================

def _is_equation_complex(expr):
    """Определяет, содержит ли уравнение трансцендентные функции"""
    expr_str = str(expr)
    transcendental = ['log', 'exp', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh']
    return any(func in expr_str for func in transcendental)


def _find_domain_boundaries(expr, var):
    """Находит границы области определения функции"""
    critical_points = set()

    # Ищем знаменатели
    denominators = []
    if expr.is_Mul or expr.is_Add:
        for arg in expr.args:
            if arg.is_Pow and arg.exp.is_negative:
                denominators.append(arg.base)

    # Ищем аргументы логарифмов
    log_args = []
    for subexpr in sympy.preorder_traversal(expr):
        if isinstance(subexpr, sympy.log):
            log_args.append(subexpr.args[0])

    # Находим нули знаменателей
    for denom in denominators:
        try:
            zeros = solve(denom, var)
            for z in zeros:
                if z.is_real:
                    critical_points.add(float(z))
        except:
            pass

    # Находим точки, где аргументы логарифмов <= 0
    for log_arg in log_args:
        try:
            zeros = solve(log_arg, var)
            for z in zeros:
                if z.is_real:
                    critical_points.add(float(z))
        except:
            pass

    return sorted(critical_points)


def _get_valid_guesses(expr, var, guesses):
    """Фильтрует начальные точки, где функция определена"""
    try:
        import warnings
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        f = sympy.lambdify(var, expr, 'numpy')
    except:
        return guesses

    valid_guesses = []
    for g in guesses:
        try:
            val = f(g)
            if not (np.isnan(val) or np.isinf(val)):
                valid_guesses.append(g)
        except:
            continue

    return valid_guesses if valid_guesses else guesses


def _verify_solution(expr, var, sol_float):
    """Проверяет, является ли найденное решение корнем"""
    try:
        import warnings
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        f = sympy.lambdify(var, expr, 'numpy')
        check_val = abs(f(sol_float))
        return check_val <= 0.01
    except:
        return True


def _generate_smart_guesses(expr, var):
    """Генерирует умные начальные точки с учетом области определения"""
    critical_points = _find_domain_boundaries(expr, var)
    guesses = []

    if not critical_points:
        guesses = list(np.linspace(-100, 100, 201))
    else:
        sorted_critical = sorted(critical_points)

        # Слева от первой критической точки
        if sorted_critical[0] > -100:
            guesses.extend(np.linspace(-100, sorted_critical[0] - 0.1, 20))

        # Между критическими точками
        for i in range(len(sorted_critical) - 1):
            left = sorted_critical[i] + 0.1
            right = sorted_critical[i + 1] - 0.1
            if right > left:
                guesses.extend(np.linspace(left, right, 20))

        # Справа от последней критической точки
        if sorted_critical[-1] < 100:
            guesses.extend(np.linspace(sorted_critical[-1] + 0.1, 100, 20))

    return guesses


def _solve_numerically_basic(expr, var, num_points=21, precision=15):
    """Базовое численное решение"""
    guesses = list(np.linspace(-10, 10, num_points))
    valid_guesses = _get_valid_guesses(expr, var, guesses)

    solutions = set()
    for guess in valid_guesses:
        try:
            sol = nsolve(expr, var, guess, verify=False)
            sol_float = float(sol)

            if _verify_solution(expr, var, sol_float):
                sol_float = round(sol_float, precision)
                solutions.add(sol_float)
        except Exception:
            continue

    return solutions


def _solve_numerically_extended(expr, var, precision=15):
    """Расширенное численное решение для сложных уравнений"""
    extended_guesses = _generate_smart_guesses(expr, var)
    valid_guesses = _get_valid_guesses(expr, var, extended_guesses)

    solutions = set()
    for guess in valid_guesses:
        try:
            sol = nsolve(expr, var, guess, verify=False)
            sol_float = float(sol)

            if _verify_solution(expr, var, sol_float):
                sol_float = round(sol_float, precision)
                solutions.add(sol_float)
        except Exception:
            continue

    return solutions


# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ РЕШЕНИЯ УРАВНЕНИЙ
# ============================================================================

@log_call
def solve_equation(equation: str, variable: str = None, local_dict=None, numeric_guesses=None, precision=15):
    """
    Решает алгебраическое уравнение

    :param equation: Строка с уравнением (expr = 0 или expr)
    :param variable: Переменная для решения
    :param local_dict: Локальный словарь для sympify
    :param numeric_guesses: Список стартовых значений для численного решения
    :param precision: Точность округления (знаков после запятой)
    :return: Список решений
    """
    if not equation:
        return []

    local_dict = local_dict or {}

    if variable is None:
        variable = 'x'

    var = symbols(variable)

    # Парсинг уравнения
    if '=' in equation:
        left_side, right_side = equation.split('=', 1)
    else:
        left_side, right_side = equation, '0'

    try:
        lhs = sympify(left_side, locals=local_dict)
        rhs = sympify(right_side, locals=local_dict)
        expr = lhs - rhs
    except Exception as e:
        return f"Ошибка парсинга: {e}"

    # Проверка наличия переменной
    if var not in expr.free_symbols:
        try:
            result = expr.evalf()
            if abs(result) < 1e-10:
                return "Тождество: уравнение верно для любого значения переменной"
            else:
                return []
        except:
            return expr

    # Проверка на тождество/противоречие
    try:
        simplified_expr = simplify(expr)

        if simplified_expr == 0 or (hasattr(simplified_expr, 'is_zero') and simplified_expr.is_zero):
            return True  # ИЗМЕНЕНО: возвращаем True вместо строки

        if simplified_expr.is_number and simplified_expr != 0:
            return []
    except:
        pass

    # Символьное решение
    try:
        solutions = solve(Eq(lhs, rhs), var)
        if solutions and isinstance(solutions, list) and len(solutions) > 0:
            # Упрощаем каждое решение
            simplified = []
            for sol in solutions:
                # Заменяем log(e) -> 1 и log(E) -> 1
                sol = sol.subs(sympy.log(sympy.E), 1)
                sol = sol.subs(sympy.log(sympy.exp(1)), 1)

                # Упрощаем
                simplified_sol = sympy.simplify(sol)

                # Пытаемся развернуть логарифмы
                try:
                    expanded = sympy.expand_log(simplified_sol, force=True)
                    if expanded != simplified_sol:
                        simplified_sol = sympy.simplify(expanded)
                except:
                    pass

                # Вычисляем числовые значения
                try:
                    if simplified_sol.is_number:
                        computed = sympy.nsimplify(simplified_sol)
                        simplified.append(computed)
                    else:
                        simplified.append(simplified_sol)
                except:
                    simplified.append(simplified_sol)
            return simplified
    except Exception:
        pass

    # Численное решение с пользовательскими точками
    if numeric_guesses is not None:
        valid_guesses = _get_valid_guesses(expr, var, numeric_guesses)
        numeric_solutions = set()

        for guess in valid_guesses:
            try:
                sol = nsolve(expr, var, guess, verify=False)
                sol_float = float(sol)

                if _verify_solution(expr, var, sol_float):
                    sol_float = round(sol_float, precision)
                    numeric_solutions.add(sol_float)
            except Exception:
                continue

        if numeric_solutions:
            return sorted(numeric_solutions)
    else:
        # Базовое численное решение
        numeric_solutions = _solve_numerically_basic(expr, var, num_points=21, precision=precision)

        if numeric_solutions:
            return sorted(numeric_solutions)

    # Расширенное численное решение для сложных уравнений
    if _is_equation_complex(expr):
        numeric_solutions = _solve_numerically_extended(expr, var, precision=precision)

        if numeric_solutions:
            return sorted(numeric_solutions)

    return f"Не удалось найти решения для: {equation}"


# ============================================================================
# ПРОИЗВОДНАЯ
# ============================================================================

@log_call
def derivative(expression: str, local_dict=None):
    """
    Вычисляет производную по указанной переменной

    :param expression: "expr по x" или просто "expr"
    :param local_dict: Локальный словарь
    :return: Производная
    """
    if not expression:
        return None

    # Ищем "по <variable>"
    match = re.search(r'по\s+([a-zA-Zа-яА-Я])\b', expression, re.I)
    if match:
        variable = match.group(1)
        expression = re.sub(r'по\s+[a-zA-Zа-яА-Я]\b', '', expression, flags=re.I).strip()
    else:
        variable = None

    # Парсим выражение
    try:
        expr = sympify(expression, locals=local_dict)
    except Exception as e:
        return f"Ошибка парсинга: {e}"

    # Определяем переменную
    free_vars = list(expr.free_symbols)

    if variable is None:
        if len(free_vars) == 1:
            var = free_vars[0]
        elif len(free_vars) == 0:
            return 0
        else:
            return f"Несколько переменных: {free_vars}. Укажите через 'по x'"
    else:
        var = symbols(variable)

    # Вычисляем производную
    derivative_expr = diff(expr, var)

    return derivative_expr


# ============================================================================
# ВЫЧЕТ
# ============================================================================

@log_call
def calculation_residue(expression: str, variable: str = 'x', local_dict=None):
    """
    Вычисляет вычет функции в точке
    Формат: "функция по переменная в точка"
    """
    pattern = r'(.+?)\s+по\s+(\w+)\s+в\s+(.+)'
    match = re.match(pattern, expression.strip(), re.I)

    if not match:
        return "Ошибка: формат должен быть 'функция по переменная в точка'"

    func_str = match.group(1)
    var_str = match.group(2)
    point_str = match.group(3)

    try:
        f = sympify(func_str, locals=local_dict)
        var = symbols(var_str)
        pt = sympify(point_str, locals=local_dict)

        return sympy.residue(f, var, pt)
    except Exception as e:
        return f"Ошибка вычисления вычета: {e}"


# ============================================================================
# ИНТЕГРИРОВАНИЕ
# ============================================================================

@log_call
def integrate_func(integral_or_expr, local_dict=None):
    """
    Вычисляет интеграл

    :param integral_or_expr: Либо Integral объект, либо выражение
    :param local_dict: Локальный словарь
    :return: Результат интегрирования
    """
    if isinstance(integral_or_expr, Integral):
        # Если это объект Integral, просто вычисляем
        try:
            result = integral_or_expr.doit()
            return result
        except Exception as e:
            return f"Ошибка интегрирования: {e}"
    else:
        # Если это строка или выражение
        try:
            if isinstance(integral_or_expr, str):
                expr = sympify(integral_or_expr, locals=local_dict)
            else:
                expr = integral_or_expr

            # Ищем все Integral в выражении
            integrals = list(expr.atoms(Integral))

            if not integrals:
                # Нет интегралов - возвращаем как есть
                return expr

            # Вычисляем каждый интеграл
            for integral in integrals:
                computed = integral.doit()
                expr = expr.xreplace({integral: computed})

            return expr

        except Exception as e:
            return f"Ошибка интегрирования: {e}"


# ============================================================================
# ПРОСТЫЕ ФУНКЦИИ УПРОЩЕНИЯ
# ============================================================================

@log_call
def simplify_func(expression: str, local_dict=None):
    expr = sympify(expression, locals=local_dict)
    return sympy.simplify(expr)


@log_call
def expand_func(expression: str, local_dict=None):
    expr = sympify(expression, locals=local_dict)
    return sympy.expand(expr)


@log_call
def factor_func(expression: str, local_dict=None):
    expr = sympify(expression, locals=local_dict)
    return sympy.factor(expr)


@log_call
def cancel_func(expression: str, local_dict=None):
    expr = sympify(expression, locals=local_dict)
    return sympy.cancel(expr)


@log_call
def together_func(expression: str, local_dict=None):
    expr = sympify(expression, locals=local_dict)
    return sympy.together(expr)


@log_call
def apart_func(expression: str, local_dict=None):
    expr = sympify(expression, locals=local_dict)
    return sympy.apart(expr)


@log_call
def collect_func(expression: str, local_dict=None):
    # Извлекаем переменную из "по <var>"
    match = re.search(r'по\s+([a-zA-Zа-яА-Я])\b', expression)
    if match:
        var_name = match.group(1)
        expression = re.sub(r'по\s+[a-zA-Zа-яА-Я]\b', '', expression).strip()
    else:
        # Автоопределение
        expr = sympify(expression, locals=local_dict)
        free_vars = list(expr.free_symbols)
        if len(free_vars) == 0:
            return "Выражение не содержит переменных"
        elif len(free_vars) == 1:
            var_name = str(free_vars[0])
        else:
            return f"Несколько переменных: {free_vars}. Укажите через 'по x'"

    expr = sympify(expression, locals=local_dict)
    var = symbols(var_name)
    return sympy.collect(expr, var)


@log_call
def trigsimp_func(expression: str, local_dict=None):
    expr = sympify(expression, locals=local_dict)
    return sympy.trigsimp(expr)


@log_call
def powsimp_func(expression: str, local_dict=None):
    expr = sympify(expression, locals=local_dict)
    return sympy.powsimp(expr)


@log_call
def radsimp_func(expression: str, local_dict=None):
    expr = sympify(expression, locals=local_dict)
    return sympy.radsimp(expr)


@log_call
def ratsimp_func(expression: str, local_dict=None):
    expr = sympify(expression, locals=local_dict)
    return sympy.ratsimp(expr)


@log_call
def logcombine_func(expression: str, local_dict=None):
    expr = sympify(expression, locals=local_dict)
    return sympy.logcombine(expr)


@log_call
def nsimplify_func(expression: str, local_dict=None):
    expr = sympify(expression, locals=local_dict)
    return sympy.nsimplify(expr)


@log_call
def sqrtdenest_func(expression: str, local_dict=None):
    expr = sympify(expression, locals=local_dict)
    return sympy.sqrtdenest(expr)


@log_call
def factor_terms_func(expression: str, local_dict=None):
    expr = sympify(expression, locals=local_dict)
    return sympy.factor_terms(expr)


@log_call
def expand_complex_func(expression: str, local_dict=None):
    expr = sympify(expression, locals=local_dict)
    return sympy.expand_complex(expr)


@log_call
def separatevars_func(expression: str, local_dict=None):
    expr = sympify(expression, locals=local_dict)
    return sympy.separatevars(expr)


# ============================================================================
# ФУНКЦИИ С НЕСКОЛЬКИМИ АРГУМЕНТАМИ
# ============================================================================

@log_call
def gcd_func(expression: str, local_dict=None):
    parts = [p.strip() for p in expression.split(',')]
    exprs = [sympify(p, locals=local_dict) for p in parts]
    return sympy.gcd(*exprs)


@log_call
def lcm_func(expression: str, local_dict=None):
    parts = [p.strip() for p in expression.split(',')]
    exprs = [sympify(p, locals=local_dict) for p in parts]
    return sympy.lcm(*exprs)


@log_call
def div_func(expression: str, local_dict=None):
    parts = [p.strip() for p in expression.split(',')]
    if len(parts) != 2:
        return "Ошибка: div требует ровно 2 аргумента"
    expr1 = sympify(parts[0], locals=local_dict)
    expr2 = sympify(parts[1], locals=local_dict)
    return sympy.div(expr1, expr2)


@log_call
def quo_func(expression: str, local_dict=None):
    parts = [p.strip() for p in expression.split(',')]
    if len(parts) != 2:
        return "Ошибка: quo требует ровно 2 аргумента"
    expr1 = sympify(parts[0], locals=local_dict)
    expr2 = sympify(parts[1], locals=local_dict)
    return sympy.quo(expr1, expr2)


@log_call
def rem_func(expression: str, local_dict=None):
    parts = [p.strip() for p in expression.split(',')]
    if len(parts) != 2:
        return "Ошибка: rem требует ровно 2 аргумента"
    expr1 = sympify(parts[0], locals=local_dict)
    expr2 = sympify(parts[1], locals=local_dict)
    return sympy.rem(expr1, expr2)


# ============================================================================
# ФУНКЦИИ ДЛЯ РАБОТЫ С МНОГОЧЛЕНАМИ
# ============================================================================

@log_call
def poly_func(expression: str, local_dict=None):
    expr = sympify(expression, locals=local_dict)
    return sympy.Poly(expr)


@log_call
def degree_func(expression: str, local_dict=None):
    match = re.search(r'по\s+([a-zA-Zа-яА-Я])\b', expression)
    if match:
        var_name = match.group(1)
        expression = re.sub(r'по\s+[a-zA-Zа-яА-Я]\b', '', expression).strip()
        var = symbols(var_name)
    else:
        var = None

    expr = sympify(expression, locals=local_dict)

    try:
        poly = sympy.Poly(expr, var) if var else sympy.Poly(expr)
        return poly.degree()
    except Exception as e:
        if var is None:
            return f"Для многочленов с несколькими переменными укажите переменную через 'по x' ({e})"
        raise


@log_call
def content_func(expression: str, local_dict=None):
    match = re.search(r'по\s+([a-zA-Zа-яА-Я])\b', expression)
    if match:
        var_name = match.group(1)
        expression = re.sub(r'по\s+[a-zA-Zа-яА-Я]\b', '', expression).strip()
        var = symbols(var_name)
    else:
        var = None

    expr = sympify(expression, locals=local_dict)
    poly = sympy.Poly(expr, var) if var else sympy.Poly(expr)
    return poly.content()


@log_call
def primitive_func(expression: str, local_dict=None):
    match = re.search(r'по\s+([a-zA-Zа-яА-Я])\b', expression)
    if match:
        var_name = match.group(1)
        expression = re.sub(r'по\s+[a-zA-Zа-яА-Я]\b', '', expression).strip()
        var = symbols(var_name)
    else:
        var = None

    expr = sympify(expression, locals=local_dict)
    poly = sympy.Poly(expr, var) if var else sympy.Poly(expr)
    return poly.primitive()


@log_call
def LC_func(expression: str, local_dict=None):
    """Leading Coefficient - старший коэффициент"""
    match = re.search(r'по\s+([a-zA-Zа-яА-Я])\b', expression)
    if match:
        var_name = match.group(1)
        expression = re.sub(r'по\s+[a-zA-Zа-яА-Я]\b', '', expression).strip()
        var = symbols(var_name)
    else:
        var = None

    expr = sympify(expression, locals=local_dict)
    poly = sympy.Poly(expr, var) if var else sympy.Poly(expr)
    return poly.LC()


@log_call
def LM_func(expression: str, local_dict=None):
    """Leading Monomial - старший одночлен"""
    match = re.search(r'по\s+([a-zA-Zа-яА-Я])\b', expression)
    if match:
        var_name = match.group(1)
        expression = re.sub(r'по\s+[a-zA-Zа-яА-Я]\b', '', expression).strip()
        var = symbols(var_name)
    else:
        var = None

    expr = sympify(expression, locals=local_dict)
    poly = sympy.Poly(expr, var) if var else sympy.Poly(expr)
    return poly.LM()


@log_call
def LT_func(expression: str, local_dict=None):
    """Leading Term - старший член"""
    match = re.search(r'по\s+([a-zA-Zа-яА-Я])\b', expression)
    if match:
        var_name = match.group(1)
        expression = re.sub(r'по\s+[a-zA-Zа-яА-Я]\b', '', expression).strip()
        var = symbols(var_name)
    else:
        var = None

    expr = sympify(expression, locals=local_dict)
    poly = sympy.Poly(expr, var) if var else sympy.Poly(expr)
    return poly.LT()


@log_call
def sqf_list_func(expression: str, local_dict=None):
    """Квадратно-свободное разложение"""
    match = re.search(r'по\s+([a-zA-Zа-яА-Я])\b', expression)
    if match:
        var_name = match.group(1)
        expression = re.sub(r'по\s+[a-zA-Zа-яА-Я]\b', '', expression).strip()
        var = symbols(var_name)
    else:
        var = None

    expr = sympify(expression, locals=local_dict)
    poly = sympy.Poly(expr, var) if var else sympy.Poly(expr)
    return poly.sqf_list()


@log_call
def limit_func(expression: str, local_dict=None):
    """
    Вычисляет предел функции

    Поддерживаемые форматы:
    - "f(x), x, point" - предел f(x) при x -> point
    - "Limit(f(x), x, point)" - объект Limit

    Направление предела:
    - point - двусторонний предел
    - "point+" или "point-" для односторонних пределов

    Примеры:
        limit_func("1/x, x, 0+")  # +∞
        limit_func("1/x, x, 0-")  # -∞
        limit_func("sin(x)/x, x, 0")  # 1
        limit_func("(1+1/x)**x, x, oo")  # e
    """
    if local_dict is None:
        local_dict = {}

    try:
        # Пытаемся распарсить как Limit объект
        expr = sympify(expression, locals=local_dict)

        # Если это уже Limit, вычисляем
        if hasattr(expr, 'doit'):
            result = expr.doit()
            return result

        # Иначе парсим как "expr, var, point"
        parts = [p.strip() for p in expression.split(',')]

        if len(parts) < 3:
            return "Ошибка: недостаточно аргументов. Формат: f(x), x, point"

        func_expr = sympify(parts[0], locals=local_dict)
        var_str = parts[1].strip()
        point_str = parts[2].strip()

        # Определяем переменную
        var = sympify(var_str, locals=local_dict)
        if not isinstance(var, Symbol):
            return f"Ошибка: '{var_str}' не является переменной"

        # Определяем точку и направление
        direction = '+-'  # по умолчанию двусторонний

        if point_str.endswith('+'):
            direction = '+'
            point_str = point_str[:-1].strip()
        elif point_str.endswith('-'):
            direction = '-'
            point_str = point_str[:-1].strip()

        # Парсим точку (может быть число, oo, -oo и т.д.)
        if point_str.lower() in ('oo', 'inf', 'infinity', '∞'):
            point = oo
        elif point_str.lower() in ('-oo', '-inf', '-infinity'):
            point = -oo
        else:
            point = sympify(point_str, locals=local_dict)

        # Вычисляем предел
        result = limit(func_expr, var, point, dir=direction)

        return result

    except Exception as e:
        return f"Ошибка при вычислении предела: {e}"