import numpy as np
import sympy
import re
from sympy import symbols, sympify, Eq, solve, nsolve, SympifyError, diff
from sympy.abc import (
    a, b, c, d, e, f, g, h, i, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z
)

from logs.logger import log_call


def find_root_nsolve(equation_str, variable='x', scan_range=(-5, 5), scan_step=0.01):
    x = sympy.symbols(variable)

    # Преобразуем строку в символьное выражение
    expr = sympy.sympify(equation_str)

    # Лямбда для быстрого вычисления значений
    f = sympy.lambdify(x, expr, 'numpy')

    # Сканируем диапазон для поиска изменения знака
    xs = np.arange(scan_range[0], scan_range[1], scan_step)
    x0 = None
    for xi, xi_next in zip(xs, xs[1:]):
        try:
            val_i = f(xi)
            val_next = f(xi_next)
            if np.isnan(val_i) or np.isnan(val_next) or np.isinf(val_i) or np.isinf(val_next):
                continue
            if val_i * val_next < 0:
                x0 = (xi + xi_next) / 2  # берем середину интервала
                break
        except:
            continue

    if x0 is None:
        raise ValueError("Не удалось найти стартовое x0. Попробуйте увеличить диапазон scan_range.")

    # Используем nsolve для численного решения
    try:
        root = sympy.nsolve(expr, x, x0)
        return float(root)
    except Exception as e:
        raise ValueError(f"nsolve не смог найти корень: {e}")


def smart_nsolve(expr_str, variable='x', local_dict=None, guesses=None):
    """
    Численно решает уравнение, перебирая несколько стартовых значений.

    :param expr_str: Строка с уравнением (expr = 0)
    :param variable: Переменная для решения
    :param local_dict: Локальные функции/константы
    :param guesses: Список стартовых значений для nsolve
    :return: Список найденных численных решений
    """
    local_dict = local_dict or {}
    var = symbols(variable)
    expr = sympify(expr_str, locals=local_dict)

    # Если нет стартовых значений, используем диапазон от -10 до 10
    if guesses is None:
        guesses = np.linsympyace(-10, 10, 21)  # 21 точка

    solutions = set()

    for g in guesses:
        try:
            sol = nsolve(expr, var, g)
            # округляем до 15 знаков, чтобы убрать почти одинаковые решения
            sol = round(float(sol), 15)
            solutions.add(sol)
        except Exception:
            continue

    return sorted(solutions)


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ЧИСЛЕННОГО РЕШЕНИЯ
# ============================================================================

def _is_equation_complex(expr):
    """
    Определяет, является ли уравнение сложным (содержит трансцендентные функции).

    :param expr: Символьное выражение SymPy
    :return: True если уравнение сложное, False если простое
    """
    expr_str = str(expr)
    transcendental_functions = ['log', 'exp', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh']
    return any(func in expr_str for func in transcendental_functions)


def _find_domain_boundaries(expr, var):
    """
    Пытается определить границы области определения функции.
    Ищет точки разрыва (деление на ноль, логарифмы отрицательных чисел и т.д.)

    :param expr: Символьное выражение
    :param var: Переменная
    :return: Список критических точек (границ области определения)
    """
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
            zeros = sympy.solve(denom, var)
            for z in zeros:
                if z.is_real:
                    critical_points.add(float(z))
        except:
            pass

    # Находим точки, где аргументы логарифмов становятся <= 0
    for log_arg in log_args:
        try:
            zeros = sympy.solve(log_arg, var)
            for z in zeros:
                if z.is_real:
                    critical_points.add(float(z))
        except:
            pass

    return sorted(critical_points)


def _get_valid_guesses(expr, var, guesses):
    """
    Фильтрует начальные точки, оставляя только те, где функция определена.

    :param expr: Символьное выражение
    :param var: Переменная
    :param guesses: Список начальных значений
    :return: Список валидных начальных значений
    """
    try:
        # Подавляем warnings от numpy
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
    """
    Проверяет, является ли найденное решение действительным корнем уравнения.

    :param expr: Символьное выражение
    :param var: Переменная
    :param sol_float: Найденное решение
    :return: True если решение верное, False если нет
    """
    try:
        import warnings
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        f = sympy.lambdify(var, expr, 'numpy')
        check_val = abs(f(sol_float))
        return check_val <= 0.01  # допустимая погрешность
    except:
        return True  # если не можем проверить, считаем что верно


def _generate_smart_guesses(expr, var):
    """
    Генерирует умные начальные точки для поиска корней,
    учитывая область определения функции.

    :param expr: Символьное выражение
    :param var: Переменная
    :return: Список начальных точек
    """
    # Находим критические точки (границы области определения)
    critical_points = _find_domain_boundaries(expr, var)

    guesses = []

    if not critical_points:
        # Если нет критических точек, используем широкий диапазон
        guesses = list(np.linsympyace(-100, 100, 201))
    else:
        # Генерируем точки в каждом интервале между критическими точками
        sorted_critical = sorted(critical_points)

        # Слева от первой критической точки
        if sorted_critical[0] > -100:
            guesses.extend(np.linsympyace(-100, sorted_critical[0] - 0.1, 20))

        # Между критическими точками
        for i in range(len(sorted_critical) - 1):
            left = sorted_critical[i] + 0.1
            right = sorted_critical[i + 1] - 0.1
            if right > left:
                guesses.extend(np.linsympyace(left, right, 20))

        # Справа от последней критической точки
        if sorted_critical[-1] < 100:
            guesses.extend(np.linsympyace(sorted_critical[-1] + 0.1, 100, 20))

    return guesses


def _solve_numerically_basic(expr, var, num_points=21, precision=15):
    """
    Базовое численное решение с ограниченным набором начальных точек.
    Используется для простых уравнений.

    :param expr: Символьное выражение
    :param var: Переменная
    :param num_points: Количество начальных точек для перебора
    :param precision: Количество знаков после запятой (по умолчанию 15)
    :return: Множество найденных решений
    """
    guesses = list(np.linsympyace(-10, 10, num_points))
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
    """
    Расширенное численное решение с большим количеством начальных точек.
    Используется для сложных трансцендентных уравнений.

    :param expr: Символьное выражение
    :param var: Переменная
    :param precision: Количество знаков после запятой (по умолчанию 15)
    :return: Множество найденных решений
    """
    # Используем умную генерацию начальных точек с учетом области определения
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
def solve_equation(equation: str, variable: str = 'x', local_dict=None, numeric_guesses=None, precision=15):
    """
    Решает алгебраическое уравнение или выражение.

    Алгоритм:
    1. Если нет переменной - просто вычисляет значение выражения
    2. Пытается символьное решение через solve()
    3. Если не получилось - базовое численное решение (21 точка)
    4. Если не получилось И уравнение сложное - расширенное численное решение (200+ точек)

    :param equation: Строка с уравнением или выражением (expr = 0)
    :param variable: Переменная для решения
    :param local_dict: Локальный словарь для пользовательских функций и констант
    :param numeric_guesses: Список стартовых значений для численного решения (опционально)
    :param precision: Количество знаков после запятой (по умолчанию 15, как в Wolfram Alpha)
    :return: Список решений или значение выражения
    """
    if not equation:
        return []

    local_dict = local_dict or {}
    var = symbols(variable)

    # ========================================================================
    # ШАГ 1: Парсинг уравнения
    # ========================================================================
    if '=' in equation:
        left_side, right_side = equation.sympylit('=', 1)
    else:
        left_side, right_side = equation, '0'

    lhs = sympify(left_side, locals=local_dict)
    rhs = sympify(right_side, locals=local_dict)
    expr = lhs - rhs

    # ========================================================================
    # ШАГ 2: Проверка наличия переменной
    # ========================================================================
    if var not in expr.free_symbols:
        # Это не уравнение, а просто выражение - вычисляем его
        try:
            result = expr.evalf()
            return result
        except:
            return expr

    # ========================================================================
    # ШАГ 3: Символьное решение (для простых уравнений)
    # ========================================================================
    try:
        solutions = solve(Eq(lhs, rhs), var)
        if solutions and isinstance(solutions, list) and len(solutions) > 0:
            return solutions
    except Exception:
        pass  # Переходим к численному решению

    # ========================================================================
    # ШАГ 4: Базовое численное решение (для уравнений средней сложности)
    # ========================================================================
    if numeric_guesses is not None:
        # Если пользователь задал свои начальные точки - используем их
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
        # Стандартное базовое численное решение
        numeric_solutions = _solve_numerically_basic(expr, var, num_points=21, precision=precision)

        if numeric_solutions:
            return sorted(numeric_solutions)

    # ========================================================================
    # ШАГ 5: Расширенное численное решение (для сложных трансцендентных уравнений)
    # ========================================================================
    if _is_equation_complex(expr):
        numeric_solutions = _solve_numerically_extended(expr, var, precision=precision)

        if numeric_solutions:
            return sorted(numeric_solutions)

    # ========================================================================
    # Если ничего не помогло
    # ========================================================================
    return f"Не удалось найти решения для: {equation}"


# ============================================
# ОБНОВЛЕННАЯ ФУНКЦИЯ derivative_expression
# ============================================

@log_call
def derivative_expression(expression: str):
    """
    Обёртка для вычисления производной с предобработкой выражения.
    Извлекает переменную из "at <var>" или "по <var>" ДО парсинга.

    :param expression: Строка типа "at y 4x + 1y" или "по x x**2 + y"
    :return: Строковое представление производной
    """
    if not expression:
        return "Пустое выражение"

    # ШАГ 1: Извлекаем переменную ДО парсинга
    variable = None
    clean_expr = expression

    # Ищем "at <variable>" (английский) - должно быть отдельным словом
    match_en = re.search(r'\bat\s+([a-zA-Z])\b', expression, re.IGNORECASE)
    if match_en:
        variable = match_en.group(1)
        # Убираем "at <variable>" из строки
        clean_expr = re.sub(r'\bat\s+[a-zA-Z]\b', '', expression, flags=re.IGNORECASE).strip()

    # Ищем "по <variable>" (русский)
    match_ru = re.search(r'по\s+([a-zA-Zа-яА-Я])\b', expression, re.IGNORECASE)
    if match_ru:
        variable = match_ru.group(1)
        # Убираем "по <variable>" из строки
        clean_expr = re.sub(r'по\s+[a-zA-Zа-яА-Я]\b', '', expression, flags=re.IGNORECASE).strip()

    print(f"Original expression: {expression}")
    print(f"Cleaned expression: {clean_expr}")
    print(f"Extracted variable: {variable}")

    # ШАГ 2: Теперь парсим очищенное выражение (без "at y" или "по x")
    parsed_expr, local_dict = parse_user_input(clean_expr)

    # ШАГ 3: Вызываем основную функцию derivative
    try:
        # Если переменная была извлечена, добавляем её обратно в формате "по <var>"
        if variable:
            expr_with_var = f"{parsed_expr} по {variable}"
        else:
            expr_with_var = parsed_expr

        result = derivative(expr_with_var, local_dict=local_dict)
        return str(result)
    except Exception as e:
        return f"Ошибка: {e}"


# ============================================
# ОБНОВЛЕННАЯ ФУНКЦИЯ derivative
# ============================================

@log_call
def derivative(expression: str, local_dict=None):
    """
    Вычисляет производную заданного выражения по указанной переменной.
    Автораспознаёт переменную, если указано 'по <variable>' в строке.
    Если переменная не указана и в выражении одна переменная, берётся она автоматически.

    :param expression: Строковое представление выражения, например:
                       "4*x + 1*y по x"
                       "x**2 + 3*x + 2 по x"
    :param local_dict: Словарь доступных функций и констант для sympy.
    :return: Производная выражения.
    """
    if not expression:
        return None

    # Ищем ключевое слово "по <variable>"
    match = re.search(r'по\s+([a-zA-Zа-яА-Я])\b', expression, re.IGNORECASE)
    if match:
        variable = match.group(1)
        # Убираем 'по <variable>' из строки
        expression = re.sub(r'по\s+[a-zA-Zа-яА-Я]\b', '', expression, flags=re.IGNORECASE).strip()
    else:
        variable = None

    print(f"Parsed expression for derivative: {expression}")

    # Преобразуем строку в символьное выражение
    try:
        expr = sympify(expression, locals=local_dict)
    except SympifyError:
        raise ValueError(f"Некорректное выражение: {expression}")

    # Определяем переменную
    free_vars = list(expr.free_symbols)

    if variable is None:
        if len(free_vars) == 1:
            var = free_vars[0]
        elif len(free_vars) == 0:
            return 0
        else:
            raise ValueError(
                f"Несколько переменных в выражении: {free_vars}. "
                f"Укажите явно через 'по <variable>' или 'at <variable>'"
            )
    else:
        var = symbols(variable)

    # Вычисляем производную
    derivative_expr = diff(expr, var)

    return derivative_expr


def calculation_residue(expression: str, variable: str = 'x', local_dict=None):
    pattern = r'(.+?)\s+по\s+(\w+)\s+в\s+(.+)'
    match = re.match(pattern, expression.strip(), re.I)
    if not match:
        raise ValueError("Неправильный формат. Нужно: <функция> по <переменная> в <точка>")

    func_str = match.group(1)
    var_str = match.group(2)
    point_str = match.group(3)

    # преобразуем строки в SymPy
    f = sympify(func_str, locals=local_dict)
    var = symbols(var_str)
    pt = sympify(point_str, locals=local_dict)

    return sympy.residue(f, var, pt)


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
        # Автоопределение: берём первую переменную
        expr = sympify(expression, locals=local_dict)
        free_vars = list(expr.free_symbols)
        if len(free_vars) == 0:
            raise ValueError("Выражение не содержит переменных")
        elif len(free_vars) == 1:
            var_name = str(free_vars[0])
        else:
            raise ValueError(f"Несколько переменных: {free_vars}. Укажите через 'по x', 'at x', и т.д.")

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


@log_call
def gcd_func(expression: str, local_dict=None):
    # expression = "expr1, expr2, ..."
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
        raise ValueError("div требует ровно 2 аргумента")
    expr1 = sympify(parts[0], locals=local_dict)
    expr2 = sympify(parts[1], locals=local_dict)
    return sympy.div(expr1, expr2)


@log_call
def quo_func(expression: str, local_dict=None):
    parts = [p.strip() for p in expression.split(',')]
    if len(parts) != 2:
        raise ValueError("quo требует ровно 2 аргумента")
    expr1 = sympify(parts[0], locals=local_dict)
    expr2 = sympify(parts[1], locals=local_dict)
    return sympy.quo(expr1, expr2)


@log_call
def rem_func(expression: str, local_dict=None):
    parts = [p.strip() for p in expression.split(',')]
    if len(parts) != 2:
        raise ValueError("rem требует ровно 2 аргумента")
    expr1 = sympify(parts[0], locals=local_dict)
    expr2 = sympify(parts[1], locals=local_dict)
    return sympy.rem(expr1, expr2)


@log_call
def poly_func(expression: str, local_dict=None):
    expr = sympify(expression, locals=local_dict)
    return sympy.Poly(expr)


@log_call
def content_func(expression: str, local_dict=None):
    # Извлекаем переменную если указана
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
    # Извлекаем переменную если указана
    match = re.search(r'по\s+([a-zA-Zа-яА-Я])\b', expression)
    if match:
        var_name = match.group(1)
        expression = re.sub(r'по\s+[a-zA-Zа-яА-Я]\b', '', expression).strip()
        var = symbols(var_name)
    else:
        var = None

    expr = sympify(expression, locals=local_dict)
    poly = sympy.Poly(expr, var) if var else sympy.Poly(expr)
    return poly.primitive()  # Возвращает (content, primitive_poly)


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
    """Leading Term - старший член (LC * LM)"""
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
def collect_func(expression: str, local_dict=None):
    # Извлекаем переменную из "по <var>"
    match = re.search(r'по\s+([a-zA-Zа-яА-Я])\b', expression)
    if match:
        var_name = match.group(1)
        expression = re.sub(r'по\s+[a-zA-Zа-яА-Я]\b', '', expression).strip()
    else:
        # Автоопределение: берём первую переменную
        expr = sympify(expression, locals=local_dict)
        free_vars = list(expr.free_symbols)
        if len(free_vars) == 0:
            raise ValueError("Выражение не содержит переменных")
        elif len(free_vars) == 1:
            var_name = str(free_vars[0])
        else:
            raise ValueError(f"Несколько переменных: {free_vars}. Укажите через 'по x', 'at x', и т.д.")

    expr = sympify(expression, locals=local_dict)
    var = symbols(var_name)
    return sympy.collect(expr, var)


@log_call
def degree_func(expression: str, local_dict=None):
    # Извлекаем переменную если указана
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
            raise ValueError(
                f"Для многочленов с несколькими переменными укажите переменную через 'по x', 'at x', и т.д. ({e})")
        raise


# В sympy_solver.py:
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