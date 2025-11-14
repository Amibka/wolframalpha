import json
import os
import re
import sys

import sympy
from sympy import symbols, Integral, Limit, sympify

from core.actions import actions, actions_ru
from core.math_functions import math_functions
from core.sympy_solver import (
    derivative, solve_equation, calculation_residue, poly_func, degree_func,
    rem_func, quo_func, div_func, lcm_func, gcd_func, separatevars_func,
    expand_complex_func, factor_terms_func, sqrtdenest_func, nsimplify_func,
    logcombine_func, ratsimp_func, radsimp_func, powsimp_func, trigsimp_func,
    collect_func, apart_func, together_func, cancel_func, factor_func,
    expand_func, simplify_func, primitive_func, content_func, integrate_func
)
from logs.logger import log_call
from utils.error_handler import math_error_handler
from utils.suggest_correction import suggest_correction, suggest_correction_ru


def resource_path(relative_path):
    """Получить абсолютный путь к ресурсу, работает для dev и PyInstaller"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# Все переменные
a, b, c, d, e, f, g, h, i, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z = symbols(
    'a b c d e f g h i k l m n o p q r s t u v w x y z')


class MathParser:
    """Улучшенный математический парсер с поддержкой математических символов"""

    def __init__(self):
        self.local_dict = self._build_local_dict()

        # Таблица замены математических символов
        self.symbol_replacements = {
            '∫': 'integral',
            '∂': 'derivative',
            '→': '->',
            '∞': 'oo',
            '∑': 'Sum',
            '∏': 'Product',
            '√': 'sqrt',
            '∛': 'root3',
            '∜': 'root4',
            '±': '+-',
            '×': '*',
            '÷': '/',
            '≠': '!=',
            '≤': '<=',
            '≥': '>=',
            '≈': '~=',
            'π': 'pi',
            'α': 'alpha',
            'β': 'beta',
            'γ': 'gamma',
            'δ': 'delta',
            'θ': 'theta',
            'λ': 'lambda',
            'μ': 'mu',
            'σ': 'sigma',
            'ω': 'omega',
            'ε': 'eps',
            'ctg': 'cot',
            'tg': 'tan',
            'arcctg': 'acot',
            'arctg': 'atan'
        }

    def _build_local_dict(self):
        """Создаёт локальный словарь для sympify"""
        local = {name: getattr(sympy, name, None) for name in math_functions}
        local.update({
            "pi": sympy.pi,
            "E": sympy.E,
            "oo": sympy.oo,
            "tg": sympy.tan,
            "ctg": sympy.cot,
            "ln": sympy.ln,
            "Integral": Integral,
            "Limit": Limit,
            "root": lambda x, n: sympy.root(x, n)
        })
        # Добавляем все переменные
        letters = symbols('a b c d e f g h i k l m n o p q r s t u v w x y z')
        local.update({str(s): s for s in letters})

        return {k: v for k, v in local.items() if v is not None}

    @log_call
    @math_error_handler
    def replace_math_symbols(self, expr: str) -> str:
        """Заменяет математические символы на текстовые эквиваленты"""
        if not expr:
            return ""

        for symbol, replacement in self.symbol_replacements.items():
            # ИСПРАВЛЕНИЕ: Добавляем пробел после замены команд
            if replacement in ['integral', 'derivative']:
                # Заменяем и добавляем пробел, если следующий символ не пробел
                expr = re.sub(
                    re.escape(symbol) + r'(?=\S)',  # ∫ за которым НЕ следует пробел
                    replacement + ' ',  # Заменяем на "integral "
                    expr
                )
                # Если уже есть пробел - просто заменяем
                expr = expr.replace(symbol, replacement)
            else:
                expr = expr.replace(symbol, replacement)

        return expr

    @log_call
    @math_error_handler
    def balance_parentheses(self, expr: str) -> str:
        """Балансирует скобки"""
        if not expr:
            return ""

        open_count = expr.count('(')
        close_count = expr.count(')')

        if open_count > close_count:
            missing = open_count - close_count
            expr += ')' * missing
            print(f"⚠ Добавлено {missing} закрывающих скобок")
        elif close_count > open_count:
            extra = close_count - open_count
            for _ in range(extra):
                expr = expr.rstrip(')').rstrip()
            print(f"⚠ Удалено {extra} лишних закрывающих скобок")

        return expr

    @log_call
    @math_error_handler
    def insert_multiplication(self, expr: str) -> str:
        """Вставляет знак умножения где нужно"""
        if not expr:
            return ""

        # Защищаем специальные функции от изменения
        protected_functions = ['Limit', 'Integral', 'Sum', 'Product', 'Derivative']

        # 2x -> 2*x
        expr = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', expr)

        # xy -> x*y (но не sin, cos, Limit и т.д.)
        # НЕ применяем внутри Integral(...), так как уже обработано
        @math_error_handler
        def repl_var(match):
            first = match.group(1)
            second = match.group(2)

            # Проверяем, находимся ли мы внутри Integral/Limit
            pos = match.start()
            before = expr[:pos]

            # Подсчитываем открытые Integral/Limit контексты
            integral_depth = before.count('Integral(') - before.count(')')
            limit_depth = before.count('Limit(') - before[:before.rfind('Limit(') if 'Limit(' in before else 0].count(
                ')')

            if integral_depth > 0 or limit_depth > 0:
                # Внутри Integral/Limit - не трогаем
                return first + second

            rest = first + expr[match.end() - 1:]

            # Проверяем обычные математические функции
            if any(rest.startswith(func) for func in math_functions):
                return first + second

            # Проверяем защищенные функции
            if any(rest.startswith(func) for func in protected_functions):
                return first + second

            return f"{first}*{second}"

        pattern = r'(?<![a-zA-Z])([a-zA-Z])([a-zA-Z(])'
        expr = re.sub(pattern, repl_var, expr)

        # ^ -> **
        expr = expr.replace('^', '**')

        return expr

    @log_call
    @math_error_handler
    def replace_custom_log(self, expr: str) -> str:
        """
        Преобразует logBASE(expr) -> log(expr, BASE)
        Примеры:
            log10(x) -> log(x, 10)
            log2(x+1) -> log(x+1, 2)
        """
        if not expr:
            return ""

        def is_base_char(ch):
            return bool(re.match(r'[A-Za-z0-9_]', ch))

        s = expr
        i = 0
        out = []

        while i < len(s):
            if s[i:i + 3].lower() == 'log':
                j = i + 3
                base = ''

                # Собираем базу
                while j < len(s) and is_base_char(s[j]):
                    base += s[j]
                    j += 1

                # Если есть база и открывающая скобка
                if base and j < len(s) and s[j] == '(':
                    k = j
                    depth = 0

                    # Ищем закрывающую скобку
                    while k < len(s):
                        if s[k] == '(':
                            depth += 1
                        elif s[k] == ')':
                            depth -= 1
                            if depth == 0:
                                break
                        k += 1

                    if k >= len(s) or s[k] != ')':
                        out.append(s[i])
                        i += 1
                        continue

                    inner = s[j + 1:k]
                    inner_repl = self.replace_custom_log(inner)

                    out.append(f'log({inner_repl}, {base})')
                    i = k + 1
                    continue

            out.append(s[i])
            i += 1

        return ''.join(out)

    @log_call
    @math_error_handler
    def replace_roots(self, expr: str) -> str:
        """
        Преобразует корни n-й степени в root(x, n)

        Поддерживаемые формы:
        - root(3, 8) -> root(8, 3)  (меняем порядок аргументов)
        - root3(8) -> root(8, 3)
        - cbrt(8) -> root(8, 3)
        - ∛(8) -> root(8, 3)
        - ∜(16) -> root(16, 4)
        """
        if not expr:
            return ""

        # 1. Обрабатываем root(n, x) -> root(x, n)
        @math_error_handler
        def fix_root_args(m):
            arg1 = m.group(1).strip()
            arg2 = m.group(2).strip()

            # Проверяем, первый аргумент - это число (степень)?
            if re.match(r'^\d+$', arg1):
                print(f"🔄 Меняем порядок: root({arg1}, {arg2}) -> root({arg2}, {arg1})")
                return f'root({arg2}, {arg1})'
            return m.group(0)

        expr = re.sub(r'root\((\d+),\s*([^)]+)\)', fix_root_args, expr)

        # 2. Обрабатываем rootN(x) -> root(x, N)
        @math_error_handler
        def replace_rootn(m):
            n = m.group(1)
            x = m.group(2)
            result = f'root({x}, {n})'
            print(f"🔄 Преобразование: root{n}({x}) -> {result}")
            return result

        expr = re.sub(r'root(\d+)\(([^)]+)\)', replace_rootn, expr)

        # 3. Обрабатываем cbrt(x) -> root(x, 3)
        @math_error_handler
        def replace_cbrt(m):
            x = m.group(1)
            result = f'root({x}, 3)'
            print(f"🔄 Преобразование: cbrt({x}) -> {result}")
            return result

        expr = re.sub(r'cbrt\(([^)]+)\)', replace_cbrt, expr)

        return expr

    @log_call
    @math_error_handler
    def replace_limits(self, expr: str) -> str:
        """
        Преобразует текстовые пределы в Limit(...)

        Поддерживаемые формы:
        - lim x->3 (expr)
        - lim x->oo expr
        - limit x->0 expr
        - предел x->1 expr
        - x->3 (expr)  (без ключевого слова, если уже извлечено командой)
        - expr при x->3
        """
        if not expr:
            return ""

        def normalize_pow(s: str) -> str:
            return s.replace('^', '**')

        # Паттерны с ключевыми словами и без
        patterns = [
            # x->point (expr) - если команда уже извлечена
            (r'^([a-zA-Z])\s*->\s*([^\s()]+)\s*\((.+?)\)\s*$', 'direct_paren'),
            # x->point expr - если команда уже извлечена
            (r'^([a-zA-Z])\s*->\s*([^\s()]+)\s+(.+)$', 'direct'),
            # lim x->point (expr)
            (r'(?:lim|limit|предел)\s+([a-zA-Z])\s*->\s*([^\s()]+)\s*\((.+?)\)', 'arrow_paren'),
            # lim x->point expr
            (r'(?:lim|limit|предел)\s+([a-zA-Z])\s*->\s*([^\s()]+)\s+(.+)', 'arrow'),
            # expr при x->point
            (r'(.+)\s+при\s+([a-zA-Z])\s*->\s*([^\s()]+)', 'pri'),
            # expr as x->point
            (r'(.+)\s+as\s+([a-zA-Z])\s*->\s*([^\s()]+)', 'as'),
        ]

        @math_error_handler
        def repl_direct_paren(m):
            """x->3 (expr)"""
            var = m.group(1)
            point = m.group(2).strip()
            inner = normalize_pow(m.group(3).strip())

            # Проверяем направление (+ или -)
            direction = ''
            if point.endswith('+') or point.endswith('-'):
                direction = f", '{point[-1]}'"
                point = point[:-1].strip()

            result = f'Limit({inner}, {var}, {point}{direction})'
            print(f"🔄 Преобразование: {m.group(0)} -> {result}")
            return result

        @math_error_handler
        def repl_direct(m):
            """x->3 expr"""
            var = m.group(1)
            point = m.group(2).strip()
            inner = normalize_pow(m.group(3).strip())

            # Проверяем направление
            direction = ''
            if point.endswith('+') or point.endswith('-'):
                direction = f", '{point[-1]}'"
                point = point[:-1].strip()

            result = f'Limit({inner}, {var}, {point}{direction})'
            print(f"🔄 Преобразование: {m.group(0)} -> {result}")
            return result

        @math_error_handler
        def repl_arrow_paren(m):
            """lim x->3 (expr)"""
            var = m.group(1)
            point = m.group(2).strip()
            inner = normalize_pow(m.group(3).strip())

            direction = ''
            if point.endswith('+') or point.endswith('-'):
                direction = f", '{point[-1]}'"
                point = point[:-1].strip()

            result = f'Limit({inner}, {var}, {point}{direction})'
            print(f"🔄 Преобразование: {m.group(0)} -> {result}")
            return result

        @math_error_handler
        def repl_arrow(m):
            """lim x->3 expr"""
            var = m.group(1)
            point = m.group(2).strip()
            inner = normalize_pow(m.group(3).strip())

            direction = ''
            if point.endswith('+') or point.endswith('-'):
                direction = f", '{point[-1]}'"
                point = point[:-1].strip()

            result = f'Limit({inner}, {var}, {point}{direction})'
            print(f"🔄 Преобразование: {m.group(0)} -> {result}")
            return result

        @math_error_handler
        def repl_pri(m):
            """expr при x->3"""
            inner = normalize_pow(m.group(1).strip())
            var = m.group(2)
            point = m.group(3)
            return f'Limit({inner}, {var}, {point})'

        def repl_as(m):
            """expr as x->3"""
            inner = normalize_pow(m.group(1).strip())
            var = m.group(2)
            point = m.group(3)
            return f'Limit({inner}, {var}, {point})'

        # Маппинг обработчиков
        handlers = {
            'direct_paren': repl_direct_paren,
            'direct': repl_direct,
            'arrow_paren': repl_arrow_paren,
            'arrow': repl_arrow,
            'pri': repl_pri,
            'as': repl_as,
        }

        # Применяем замены
        prev = None
        max_iterations = 3
        iteration = 0

        while prev != expr and iteration < max_iterations:
            prev = expr
            for pattern, mode in patterns:
                match = re.search(pattern, expr, flags=re.I)
                if match:
                    handler = handlers.get(mode)
                    if handler:
                        expr = re.sub(pattern, lambda m: handler(m), expr, count=1, flags=re.I)
                        break
            iteration += 1

        return expr

    @log_call
    @math_error_handler
    def replace_integrals(self, expr: str) -> str:
        """
        Преобразует текстовые интегралы в Integral(...)

        Поддерживаемые формы:
        - integral x**2 dx
        - integrate x**2 dx
        - x**2 dx
        - x**2 from 0 to 1
        - integral x**2 from 0 to 1
        - интеграл x**2 по x
        - x**2 от 0 до 1
        """
        if not expr:
            return ""

        def looks_like_bound(tok: str) -> bool:
            tok = tok.strip()
            if not tok:
                return False
            if re.search(r'\d', tok):
                return True
            if tok.lower() in ('pi', 'e', 'oo', 'inf', 'infinity'):
                return True
            if tok.startswith('(') and tok.endswith(')'):
                return True
            if re.search(r'[()+\-/*]', tok):
                return True
            return re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', tok) is not None

        def normalize_pow(s: str) -> str:
            return s.replace('^', '**')

        def add_multiplication_to_inner(s: str) -> str:
            """Добавляет умножение внутри выражения интеграла"""
            # xlnx -> x*ln(x), но сохраняем функции
            s = re.sub(r'([a-zA-Z])ln([a-zA-Z])', r'\1*ln(\2)', s)
            return s

        # Паттерны с ключевыми словами
        patterns_kw = [
            (r'\b(?:integral|integrate|int|интеграл|интегрировать)\s+(.+?)\s+d\s*([a-zA-Z])\b', 'dx'),
            (r'\b(?:интеграл|интегрировать)\s+(.+?)\s+по\s+([a-zA-Z])\b', 'po'),
            (r'\b(?:integrate|integral|int)\s+(.+?)\s+from\s+([^\s]+)\s+to\s+([^\s]+)\b', 'from_to'),
            (r'\b(?:интеграл|интегрировать)\s+(.+?)\s+от\s+([^\s]+)\s+до\s+([^\s]+)\b', 'ot_do'),
        ]

        # Паттерны без ключевых слов
        patterns_naked = [
            (r'(.+?)\s+from\s+([^\s]+)\s+to\s+([^\s]+)', 'from_to'),
            (r'(.+?)\s+от\s+([^\s]+)\s+до\s+([^\s]+)', 'ot_do'),
            (r'(.+?)\s+d\s*([a-zA-Z])\b', 'dx'),
            (r'(.+?)\s+по\s+([a-zA-Z])\b', 'po'),
        ]

        @math_error_handler
        def repl_dx(m):
            inner = m.group(1).strip()
            var = m.group(2)

            # Проверяем на пределы внутри
            m_from = re.search(r'(.+?)\s+from\s+([^\s]+)\s+to\s+([^\s]+)\s*$', inner, re.I)
            if m_from:
                expr_part = normalize_pow(m_from.group(1).strip())
                expr_part = add_multiplication_to_inner(expr_part)
                a, b = m_from.group(2), m_from.group(3)
                return f'Integral({expr_part}, ({var}, {a}, {b}))'

            m_ot = re.search(r'(.+?)\s+от\s+([^\s]+)\s+до\s+([^\s]+)\s*$', inner, re.I)
            if m_ot:
                expr_part = normalize_pow(m_ot.group(1).strip())
                expr_part = add_multiplication_to_inner(expr_part)
                a, b = m_ot.group(2), m_ot.group(3)
                return f'Integral({expr_part}, ({var}, {a}, {b}))'

            # ИСПРАВЛЕНИЕ: Паттерн для определённого интеграла с пределами в начале
            # "0 1 sqrt(x^2+1)" -> Integral(sqrt(x^2+1), (x, 0, 1))
            bounds_pattern = r'^([^\s()]+)\s+([^\s()]+)\s+(.+)$'
            m_bounds = re.match(bounds_pattern, inner)

            if m_bounds:
                potential_a = m_bounds.group(1)
                potential_b = m_bounds.group(2)
                potential_expr = m_bounds.group(3)

                # Проверяем, что первые два токена - это границы
                if looks_like_bound(potential_a) and looks_like_bound(potential_b):
                    expr_part = normalize_pow(potential_expr.strip())
                    expr_part = add_multiplication_to_inner(expr_part)
                    return f'Integral({expr_part}, ({var}, {potential_a}, {potential_b}))'

            # Обычный неопределённый интеграл
            expr_part = normalize_pow(inner)
            expr_part = add_multiplication_to_inner(expr_part)
            return f'Integral({expr_part}, {var})'

        @math_error_handler
        def repl_from_to(m):
            expr_part = normalize_pow(m.group(1).strip())
            expr_part = add_multiplication_to_inner(expr_part)
            a, b = m.group(2), m.group(3)
            var = self._detect_variable(expr_part)
            return f'Integral({expr_part}, ({var}, {a}, {b}))'

        @math_error_handler
        def repl_handler(m, mode):
            if mode == 'dx' or mode == 'po':
                return repl_dx(m)
            elif mode == 'from_to' or mode == 'ot_do':
                return repl_from_to(m)
            return m.group(0)

        # Применяем замены итеративно
        prev = None
        max_iter = 5
        iteration = 0

        while prev != expr and iteration < max_iter:
            prev = expr

            # С ключевыми словами (один раз за итерацию)
            for pattern, mode in patterns_kw:
                if re.search(pattern, expr, flags=re.I):
                    expr = re.sub(pattern, lambda m: repl_handler(m, mode), expr, count=1, flags=re.I)
                    break  # Обрабатываем по одному за раз

            # Без ключевых слов (если не сработали keyword паттерны)
            if prev == expr:
                for pattern, mode in patterns_naked:
                    if re.search(pattern, expr, flags=re.I):
                        expr = re.sub(pattern, lambda m: repl_handler(m, mode), expr, count=1, flags=re.I)
                        break

            iteration += 1

        return expr

    @math_error_handler
    def _detect_variable(self, expr: str) -> str:
        """Автоопределение переменной в выражении"""
        var_pattern = r'(?<![a-zA-Z])([a-zA-Z])(?![a-zA-Z])'
        variables = re.findall(var_pattern, expr)
        valid_vars = [v for v in set(variables) if v not in math_functions]

        if 'x' in valid_vars:
            return 'x'
        elif 'y' in valid_vars:
            return 'y'
        elif 'z' in valid_vars:
            return 'z'
        elif valid_vars:
            return valid_vars[0]
        return 'x'

    @log_call
    @math_error_handler
    def parse(self, expr: str) -> tuple:
        """
        Главная функция парсинга
        Возвращает: (обработанное_выражение, local_dict)
        """
        if not expr:
            return "", self.local_dict

        # 0. Заменяем математические символы
        expr = self.replace_math_symbols(expr)

        # 0.5. ВАЖНО: Добавляем пробел после 'integral' если его нет
        # Исправляет: integralsqrt -> integral sqrt, integral1 -> integral 1
        expr = re.sub(r'\bintegral(?=[a-zA-Z0-9])', 'integral ', expr)

        # 1. Балансируем скобки
        expr = self.balance_parentheses(expr)

        # 2. Обрабатываем логарифмы
        if 'log' in expr or 'ln' in expr:
            expr = self.replace_custom_log(expr)

            # 2.5. Обрабатываем корни
        if 'root' in expr or 'cbrt' in expr:
            expr = self.replace_roots(expr)

        # 3. Обрабатываем пределы
        if any(kw in expr.lower() for kw in ['lim', 'limit', 'предел', 'при', ' as ', '->']):
            expr = self.replace_limits(expr)

        # 4. Обрабатываем интегралы ПЕРЕД insert_multiplication
        if any(kw in expr.lower() for kw in
               ['integral', 'integrate', 'int', 'интеграл', 'от', 'до', 'from', 'to', ' d']):
            expr = self.replace_integrals(expr)

        # 5. Вставляем умножение ПОСЛЕ обработки интегралов
        expr = self.insert_multiplication(expr)

        return expr, self.local_dict


class IntegralComputer:
    """Вычисляет все интегралы в выражении"""

    @staticmethod
    @log_call
    @math_error_handler
    def compute_all_integrals(parsed_str: str, local_dict: dict):
        """
        Находит все Integral(...) в выражении и вычисляет их

        Возвращает: (sympy.Expr, None) при успехе или (None, error_message) при ошибке
        """
        if not parsed_str:
            return None, "Пустая строка"

        try:
            expr = sympify(parsed_str, locals=local_dict)
        except Exception as e:
            # Пробрасываем исключение дальше, декоратор его поймает
            raise

        # Вычисляем интегралы итеративно (для обработки вложенных)
        try:
            max_iterations = 10
            iteration = 0

            while iteration < max_iterations:
                integrals = list(expr.atoms(Integral))
                if not integrals:
                    break

                # Вычисляем каждый интеграл
                for integral in integrals:
                    try:
                        # Используем встроенный интегратор
                        computed = integrate_func(integral)
                        expr = expr.xreplace({integral: computed})
                    except Exception as e:
                        return None, f"Ошибка при вычислении интеграла {integral}: {e}"

                iteration += 1

            # Упрощаем результат
            try:
                expr = sympy.simplify(expr)
            except:
                pass

            return expr, None

        except Exception as e:
            return None, f"Ошибка при обработке интегралов: {e}"


class LimitComputer:
    """Вычисляет все пределы в выражении"""

    @staticmethod
    @log_call
    @math_error_handler
    def compute_all_limits(parsed_str: str, local_dict: dict):
        """
        Находит все Limit(...) в выражении и вычисляет их

        Возвращает: (sympy.Expr, None) при успехе или (None, error_message) при ошибке
        """
        if not parsed_str:
            return None, "Пустая строка"

        print(f"DEBUG: Начинаем обработку '{parsed_str}'")

        try:
            expr = sympify(parsed_str, locals=local_dict)
            print(f"DEBUG: Успешно распарсили: {expr}")
        except Exception as e:
            error_msg = f"Ошибка sympify: {e}\nВыражение: {parsed_str}"
            print(f"DEBUG: Получен результат: {error_msg}")
            print(f"DEBUG: Тип результата: {type(error_msg)}")
            return None, error_msg

        # Вычисляем пределы итеративно
        try:
            max_iterations = 10
            iteration = 0

            while iteration < max_iterations:
                limits = list(expr.atoms(Limit))
                if not limits:
                    break

                print(f"DEBUG: Найдено пределов: {len(limits)}")

                # Вычисляем каждый предел
                for limit_obj in limits:
                    try:
                        print(f"DEBUG: Вычисляем предел: {limit_obj}")
                        # Используем встроенный вычислитель пределов
                        computed = limit_obj.doit()
                        print(f"DEBUG: Результат: {computed}")
                        expr = expr.xreplace({limit_obj: computed})
                    except Exception as e:
                        return None, f"Ошибка при вычислении предела {limit_obj}: {e}"

                iteration += 1

            # Упрощаем результат
            try:
                expr = sympy.simplify(expr)
            except:
                pass

            return expr, None

        except Exception as e:
            return None, f"Ошибка при обработке пределов: {e}"


class CommandRouter:
    """Маршрутизация команд"""

    def __init__(self):
        self.parser = MathParser()
        self.integral_computer = IntegralComputer()
        self.limit_computer = LimitComputer()

        # ИСПРАВЛЕНО: используем resource_path вместо прямого пути
        commands_file = resource_path("language/commands_translate.json")
        with open(commands_file, "r", encoding="utf-8") as f:
            self.command_translate = json.load(f)

    @log_call
    @math_error_handler
    def extract_command(self, user_input: str) -> tuple:
        """
        Извлекает команду и выражение из ввода
        Возвращает: (command, expression) или ("error", error_message)
        """
        if not user_input:
            return "error", "Пустой ввод"

        user_lower = user_input.lower().strip()

        # Проверяем точное совпадение с командами
        for command, synonyms in self.command_translate.items():
            sorted_synonyms = sorted(synonyms, key=len, reverse=True)

            for synonym in sorted_synonyms:
                if user_lower.startswith(synonym):
                    next_pos = len(synonym)
                    if next_pos >= len(user_lower) or user_lower[next_pos].isspace():
                        expression = user_input[len(synonym):].strip()
                        return command, expression

        # Если команда не найдена, ищем похожую
        first_word = user_lower.split()[0] if user_lower.split() else user_lower
        is_russian = any('а' <= c <= 'я' for c in first_word)

        if is_russian:
            suggestion = suggest_correction_ru(first_word, actions_ru)
        else:
            suggestion = suggest_correction(first_word, actions)

        if suggestion:
            return "error", f'Неизвестная команда: "{first_word}", возможно вы имели в виду "{suggestion}"?'

        # По умолчанию - solve
        return "solve", user_input

    @log_call
    @math_error_handler
    def extract_variable(self, expression: str, keywords=None, auto_detect=True):
        """
        Извлекает переменную из выражения
        Возвращает: (variable, clean_expression, error_message)
        """
        if keywords is None:
            keywords = ['по', 'at', 'by', 'in']

        # Ищем "keyword <variable>"
        pattern = r'\b(' + '|'.join(keywords) + r')\s+([a-zA-Zа-яА-Я])\b'
        match = re.search(pattern, expression, re.I)

        if match:
            variable = match.group(2)
            clean_expr = re.sub(pattern, '', expression, flags=re.I).strip()
            return variable, clean_expr, None

        # Убираем висячие ключевые слова
        orphan_pattern = r'\b(' + '|'.join(keywords) + r')(\s*$|\s+(?![a-zA-Zа-яА-Я]))'
        if re.search(orphan_pattern, expression, re.I):
            clean_expr = re.sub(r'\b(' + '|'.join(keywords) + r')\s*$', '', expression, flags=re.I).strip()
            return None, clean_expr, None

        # Автоопределение
        if auto_detect:
            var_pattern = r'(?<![a-zA-Z])([a-zA-Z])(?![a-zA-Z])'
            variables = re.findall(var_pattern, expression)
            valid_vars = [v for v in set(variables) if v not in math_functions]

            if 'x' in valid_vars:
                return 'x', expression, None
            elif 'y' in valid_vars:
                return 'y', expression, None
            elif 'z' in valid_vars:
                return 'z', expression, None
            elif valid_vars:
                return valid_vars[0], expression, None

        return None, expression, None

    @log_call
    @math_error_handler
    def process_command(self, command: str, expression: str):
        """Обрабатывает команду"""

        # Убираем = для всех команд кроме solve
        if '=' in expression and command not in ['solve', 'plot', 'graph', 'график']:
            expression = expression.split('=')[0].strip()

        # Маршрутизация команд
        action_map = {
            "solve": self._solve,
            "derivative": self._derivative,
            "residue": self._residue,
            "integral": self._integrate,
            "limit": self._limit,

            # Упрощение
            "simplify": lambda e: self._simple_func(e, simplify_func),
            "expand": lambda e: self._simple_func(e, expand_func),
            "factor": lambda e: self._simple_func(e, factor_func),
            "cancel": lambda e: self._simple_func(e, cancel_func),
            "together": lambda e: self._simple_func(e, together_func),
            "apart": lambda e: self._simple_func(e, apart_func),

            # Тригонометрия и степени
            "trigsimp": lambda e: self._simple_func(e, trigsimp_func),
            "powsimp": lambda e: self._simple_func(e, powsimp_func),
            "radsimp": lambda e: self._simple_func(e, radsimp_func),
            "ratsimp": lambda e: self._simple_func(e, ratsimp_func),
            "logcombine": lambda e: self._simple_func(e, logcombine_func),
            "nsimplify": lambda e: self._simple_func(e, nsimplify_func),
            "sqrtdenest": lambda e: self._simple_func(e, sqrtdenest_func),
            "factor_terms": lambda e: self._simple_func(e, factor_terms_func),
            "expand_complex": lambda e: self._simple_func(e, expand_complex_func),
            "separatevars": lambda e: self._simple_func(e, separatevars_func),

            # С переменными
            "collect": self._collect,
            "degree": self._degree,

            # Несколько аргументов
            "advanced.gcd": self._gcd,
            "advanced.lcm": self._lcm,
            "advanced.div": self._div,
            "advanced.quo": self._quo,
            "advanced.rem": self._rem,
            "advanced.Poly": self._poly,
            "advanced.content": self._content,
            "advanced.primitive": self._primitive,

            "plot": self._plot,
            "graph": self._plot,
            "график": self._plot,
        }

        if command in action_map:
            try:
                return action_map[command](expression)
            except Exception as e:
                return f"Ошибка при выполнении {command}: {e}"

        return f"Неизвестная команда: {command}"

    @math_error_handler
    def _solve(self, expression: str):
        """Решение уравнений с обработкой интегралов и пределов"""
        variable, clean_expr, error = self.extract_variable(expression, auto_detect=True)
        if error:
            return f"Ошибка: {error}"

        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Обрабатываем уравнения с '='
        # Преобразуем "expr1 = expr2" в "expr1 - (expr2)"
        has_equation = '=' in clean_expr

        if has_equation:
            parts = clean_expr.split('=')
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip()
                # Если правая часть пустая или "0", используем левую часть
                if not right or right == '0':
                    clean_expr = left
                else:
                    clean_expr = f"({left}) - ({right})"
            else:
                return "Ошибка: Некорректное уравнение (больше одного знака =)"

        parsed_expr, local_dict = self.parser.parse(clean_expr)

        # Проверяем, есть ли в исходном выражении интегралы или пределы
        has_integrals = 'Integral(' in parsed_expr
        has_limits = 'Limit(' in parsed_expr

        # Вычисляем все интегралы
        expr_computed, err = self.integral_computer.compute_all_integrals(parsed_expr, local_dict)
        if err:
            return err

        # Вычисляем все пределы
        expr_computed, err = self.limit_computer.compute_all_limits(str(expr_computed), local_dict)
        if err:
            return err

        # КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ #1:
        # Если в исходном выражении были только интегралы/пределы (без знака =),
        # возвращаем вычисленный результат, а не решаем уравнение
        if (has_integrals or has_limits) and not has_equation:
            print(f"📝 Результат вычисления: {expr_computed}")
            return expr_computed

        # КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ #2:
        # Если выражение НЕ содержит переменных И НЕ является уравнением,
        # просто возвращаем вычисленное значение (например, cbrt(27) -> 3)
        try:
            # Проверяем наличие переменных
            from sympy import symbols
            expr_sympy = sympify(str(expr_computed), locals=local_dict)
            free_vars = expr_sympy.free_symbols

            # Если нет переменных и нет знака =, это просто вычисление
            if not free_vars and not has_equation:
                print(f"📝 Простое вычисление (без переменных): {expr_computed}")
                return expr_computed
        except:
            pass

        # КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ #3:
        # Если не было знака = в исходном выражении и есть переменные,
        # это может быть выражение для упрощения, а не уравнение
        if not has_equation:
            # Проверяем, может ли это быть уравнением (содержит переменную)
            if variable is None:
                # Нет переменной -> возвращаем упрощенное выражение
                print(f"📝 Упрощённое выражение: {expr_computed}")
                return expr_computed

        # Иначе решаем уравнение
        final_str = str(expr_computed)
        print(f"📝 Выражение после вычислений: {final_str}")
        print(f"📝 Переменная: {variable}")

        result = solve_equation(final_str, variable, local_dict=local_dict)

        # ИСПРАВЛЕНИЕ #4: Если solve_equation вернул пустой список для константы,
        # возвращаем саму константу
        if result == [] and not has_equation:
            return expr_computed

        return result

    @math_error_handler
    def _derivative(self, expression: str):
        """Производная"""
        variable, clean_expr, error = self.extract_variable(expression, ['по', 'at', 'by', 'in'])
        if error:
            return f"Ошибка: {error}"

        parsed_expr, local_dict = self.parser.parse(clean_expr)

        if variable:
            expr_with_var = f"{parsed_expr} по {variable}"
        else:
            expr_with_var = parsed_expr

        return derivative(expr_with_var, local_dict=local_dict)

    @math_error_handler
    def _residue(self, expression: str):
        """Вычет"""
        parsed_expr, local_dict = self.parser.parse(expression)
        return calculation_residue(parsed_expr, local_dict=local_dict)

    @math_error_handler
    def _integrate(self, expression: str):
        """Интегрирование"""
        parsed_expr, local_dict = self.parser.parse(expression)

        expr_computed, err = self.integral_computer.compute_all_integrals(parsed_expr, local_dict)
        if err:
            return err

        return expr_computed

    @math_error_handler
    def _limit(self, expression: str):
        """Вычисление предела"""
        parsed_expr, local_dict = self.parser.parse(expression)

        expr_computed, err = self.limit_computer.compute_all_limits(parsed_expr, local_dict)
        if err:
            return err

        return expr_computed

    @math_error_handler
    def _simple_func(self, expression: str, func):
        """Простые функции без переменных"""
        parsed_expr, local_dict = self.parser.parse(expression)
        return func(parsed_expr, local_dict=local_dict)

    @math_error_handler
    def _collect(self, expression: str):
        """Группировка"""
        variable, clean_expr, error = self.extract_variable(expression, ['по', 'at', 'by', 'in'])
        if error:
            return f"Ошибка: {error}"

        parsed_expr, local_dict = self.parser.parse(clean_expr)

        if variable:
            expr_with_var = f"{parsed_expr} по {variable}"
        else:
            expr_with_var = parsed_expr

        return collect_func(expr_with_var, local_dict=local_dict)

    @math_error_handler
    def _degree(self, expression: str):
        """Степень многочлена"""
        variable, clean_expr, error = self.extract_variable(expression, ['по', 'at', 'by', 'in'])
        if error:
            return f"Ошибка: {error}"

        parsed_expr, local_dict = self.parser.parse(clean_expr)

        if variable:
            expr_with_var = f"{parsed_expr} по {variable}"
        else:
            expr_with_var = parsed_expr

        return degree_func(expr_with_var, local_dict=local_dict)

    @math_error_handler
    def _gcd(self, expression: str):
        """НОД"""
        if ',' not in expression:
            return "Ошибка: gcd требует минимум 2 аргумента через запятую"

        parts = [p.strip() for p in expression.split(',')]
        parsed_parts = []

        for part in parts:
            parsed, local_dict = self.parser.parse(part)
            parsed_parts.append(parsed)

        combined = ', '.join(parsed_parts)
        return gcd_func(combined, local_dict=local_dict)

    @math_error_handler
    def _lcm(self, expression: str):
        """НОК"""
        if ',' not in expression:
            return "Ошибка: lcm требует минимум 2 аргумента через запятую"

        parts = [p.strip() for p in expression.split(',')]
        parsed_parts = []

        for part in parts:
            parsed, local_dict = self.parser.parse(part)
            parsed_parts.append(parsed)

        combined = ', '.join(parsed_parts)
        return lcm_func(combined, local_dict=local_dict)

    @math_error_handler
    def _div(self, expression: str):
        """Деление многочленов"""
        if ',' not in expression:
            return "Ошибка: div требует 2 аргумента через запятую"

        parts = expression.split(',', 1)
        parsed1, local_dict = self.parser.parse(parts[0].strip())
        parsed2, _ = self.parser.parse(parts[1].strip())

        combined = f"{parsed1}, {parsed2}"
        return div_func(combined, local_dict=local_dict)

    @math_error_handler
    def _quo(self, expression: str):
        """Частное"""
        if ',' not in expression:
            return "Ошибка: quo требует 2 аргумента через запятую"

        parts = expression.split(',', 1)
        parsed1, local_dict = self.parser.parse(parts[0].strip())
        parsed2, _ = self.parser.parse(parts[1].strip())

        combined = f"{parsed1}, {parsed2}"
        return quo_func(combined, local_dict=local_dict)

    @math_error_handler
    def _rem(self, expression: str):
        """Остаток"""
        if ',' not in expression:
            return "Ошибка: rem требует 2 аргумента через запятую"

        parts = expression.split(',', 1)
        parsed1, local_dict = self.parser.parse(parts[0].strip())
        parsed2, _ = self.parser.parse(parts[1].strip())

        combined = f"{parsed1}, {parsed2}"
        return rem_func(combined, local_dict=local_dict)

    @math_error_handler
    def _poly(self, expression: str):
        """Многочлен"""
        parsed_expr, local_dict = self.parser.parse(expression)
        return poly_func(parsed_expr, local_dict=local_dict)

    @math_error_handler
    def _content(self, expression: str):
        """Содержимое"""
        variable, clean_expr, error = self.extract_variable(expression, ['по', 'at', 'by', 'in'])
        if error:
            return f"Ошибка: {error}"

        parsed_expr, local_dict = self.parser.parse(clean_expr)

        if variable:
            expr_with_var = f"{parsed_expr} по {variable}"
        else:
            expr_with_var = parsed_expr

        return content_func(expr_with_var, local_dict=local_dict)

    @math_error_handler
    def _primitive(self, expression: str):
        """Примитивная часть"""
        variable, clean_expr, error = self.extract_variable(expression, ['по', 'at', 'by', 'in'])
        if error:
            return f"Ошибка: {error}"

        parsed_expr, local_dict = self.parser.parse(clean_expr)

        if variable:
            expr_with_var = f"{parsed_expr} по {variable}"
        else:
            expr_with_var = parsed_expr

        return primitive_func(expr_with_var, local_dict=local_dict)

    """
    Добавьте/замените метод _plot в вашем core/router.py (класс CommandRouter)
    """

    @math_error_handler
    def _plot(self, expression: str):
        """
        Построение графика функции

        Форматы:
        - y = f(x) → 2D явная функция
        - F(x,y) = 0 → 2D неявная кривая (окружность, эллипс и т.д.)
        - z = f(x,y) → 3D явная функция (поверхность)
        - F(x,y,z) = 0 → 3D неявная поверхность
        """
        try:
            expr = expression.strip()

            is_3d = False
            is_implicit = False
            func_expr = None
            variables = []

            # 1. f(x,y) = expr  (3D функция)
            match_3d_func = re.match(r'^[fgz]\s*\(\s*([a-zA-Z])\s*,\s*([a-zA-Z])\s*\)\s*=\s*(.+)$', expr, re.I)
            if match_3d_func:
                is_3d = True
                is_implicit = False
                variables = [match_3d_func.group(1), match_3d_func.group(2)]
                func_expr = match_3d_func.group(3).strip()
                print(f"✅ 3D явная функция: f({variables[0]},{variables[1]}) = {func_expr}")

            # 2. f(x) = expr  (2D функция)
            elif re.match(r'^[fgh]\s*\(\s*([a-zA-Z])\s*\)\s*=\s*(.+)$', expr, re.I):
                match = re.match(r'^[fgh]\s*\(\s*([a-zA-Z])\s*\)\s*=\s*(.+)$', expr, re.I)
                is_3d = False
                is_implicit = False
                variables = [match.group(1)]
                func_expr = match.group(2).strip()
                print(f"✅ 2D явная функция: f({variables[0]}) = {func_expr}")

            # 3. y = expr  (2D явная функция)
            elif re.match(r'^y\s*=\s*(.+)$', expr, re.I):
                match = re.match(r'^y\s*=\s*(.+)$', expr, re.I)
                is_3d = False
                is_implicit = False
                func_expr = match.group(1).strip()

                # КРИТИЧНО: Определяем переменные в выражении
                detected = self._detect_variables_for_plot(func_expr, expected=None)

                # Если в выражении 2+ переменных - это 3D!
                if len(detected) >= 2:
                    # y = f(x,z) интерпретируем как z = f(x,y)
                    print(f"⚠️ ВНИМАНИЕ: y = {func_expr} содержит переменные {detected}")
                    print(f"   Для 3D графика используйте: z = {func_expr}")
                    return {
                        'type': 'error',
                        'message': f'y = {func_expr} содержит несколько переменных {detected}.\n'
                                   f'Для 2D графика: y = f(x)\n'
                                   f'Для 3D графика: z = f(x,y)'
                    }

                variables = detected if detected else ['x']
                print(f"✅ 2D явная функция: y = {func_expr}, переменная: {variables}")

            # 4. z = expr  (3D явная функция)
            elif re.match(r'^z\s*=\s*(.+)$', expr, re.I):
                match = re.match(r'^z\s*=\s*(.+)$', expr, re.I)
                is_implicit = False
                func_expr = match.group(1).strip()
                detected = self._detect_variables_for_plot(func_expr, expected=None)

                if len(detected) >= 2:
                    is_3d = True
                    variables = detected[:2]
                    print(f"✅ 3D явная функция: z = {func_expr}, переменные: {variables}")
                elif len(detected) == 1:
                    is_3d = False
                    variables = detected
                    print(f"✅ 2D явная функция: z = {func_expr}, переменная: {variables}")
                else:
                    return {
                        'type': 'error',
                        'message': 'Не удалось определить переменные'
                    }

            # 5. Уравнение с '=' → НЕЯВНАЯ КРИВАЯ/ПОВЕРХНОСТЬ
            elif '=' in expr:
                parts = expr.split('=', 1)
                if len(parts) == 2:
                    left = parts[0].strip()
                    right = parts[1].strip()

                    # Преобразуем в F(x,y) = 0 или F(x,y,z) = 0
                    if right and right != '0':
                        func_expr = f"({left}) - ({right})"
                    else:
                        func_expr = left

                    # Определяем размерность
                    detected = self._detect_variables_for_plot(func_expr, expected=None)

                    if len(detected) == 2:
                        # Две переменные → 2D неявная кривая!
                        is_3d = False
                        is_implicit = True
                        variables = detected[:2]
                        print(f"✅ 2D неявная кривая: {expr} → F({variables[0]},{variables[1]}) = 0")

                    elif len(detected) == 3:
                        # Три переменные → 3D неявная поверхность
                        is_3d = True
                        is_implicit = True
                        variables = detected[:3]
                        print(f"✅ 3D неявная поверхность: {expr} → F({','.join(variables)}) = 0")

                    elif len(detected) == 1:
                        # Одна переменная → ошибка, это не кривая
                        return {
                            'type': 'error',
                            'message': f'{expr} содержит только одну переменную.\n'
                                       f'Используйте формат: y = f(x) или x^2 + y^2 = 25'
                        }

                    else:
                        return {
                            'type': 'error',
                            'message': f'Не удалось определить тип уравнения: {expr}'
                        }

            # 6. Простое выражение без '='
            else:
                func_expr = expr
                detected = self._detect_variables_for_plot(func_expr, expected=None)

                # Простое выражение всегда трактуем как явную функцию
                if len(detected) >= 2:
                    is_3d = True
                    is_implicit = False
                    variables = detected[:2]
                    print(f"✅ 3D явная функция (без z=): {func_expr}, переменные: {variables}")
                elif len(detected) == 1:
                    is_3d = False
                    is_implicit = False
                    variables = detected
                    print(f"✅ 2D явная функция (без y=): {func_expr}, переменная: {variables}")
                else:
                    return {
                        'type': 'error',
                        'message': 'Не удалось определить переменные'
                    }

            # ВАЛИДАЦИЯ
            if not func_expr or func_expr.strip() == '':
                return {
                    'type': 'error',
                    'message': f'Пустое выражение: "{expression}"'
                }

            # Парсим выражение
            parsed_expr, local_dict = self.parser.parse(func_expr)

            # Определяем финальный тип графика
            if is_3d:
                plot_type = 'plot_3d_implicit' if is_implicit else 'plot_3d'
            else:
                plot_type = 'plot_2d_implicit' if is_implicit else 'plot_2d'

            print(f"📊 Финальный результат: type={plot_type}, expr={parsed_expr}, vars={variables}")

            return {
                'type': plot_type,
                'expression': parsed_expr,
                'variables': variables,
                'original': expression,
                'is_implicit': is_implicit
            }

        except Exception as e:
            import traceback
            return {
                'type': 'error',
                'message': f'Ошибка: {str(e)}\n\n{traceback.format_exc()}'
            }

    @math_error_handler
    def _detect_variables_for_plot(self, expr_str: str, expected=None):
        """
        Автоопределение переменных в выражении

        :param expr_str: Строка выражения
        :param expected: Ожидаемое количество переменных (1, 2, или None)
        :return: Список переменных
        """
        import re
        from core.math_functions import math_functions

        # Находим все переменные
        var_pattern = r'(?<![a-zA-Z])([a-zA-Z])(?![a-zA-Z])'
        variables = re.findall(var_pattern, expr_str)

        # Исключаем константы и функции
        exclude = set(math_functions) | {'e', 'E', 'i', 'I'}
        valid_vars = [v for v in variables if v not in exclude]

        # Убираем дубликаты, сохраняя порядок
        seen = set()
        unique_vars = []
        for v in valid_vars:
            if v not in seen:
                seen.add(v)
                unique_vars.append(v)

        # Сортируем по приоритету
        priority = ['x', 'y', 'z', 't', 'r', 'u', 'v', 'w']
        sorted_vars = []

        for p in priority:
            if p in unique_vars:
                sorted_vars.append(p)
                unique_vars.remove(p)

        sorted_vars.extend(sorted(unique_vars))

        # Возвращаем нужное количество
        if expected:
            return sorted_vars[:expected]
        return sorted_vars


# Главная функция для внешнего API
@log_call
@math_error_handler
def get_text(user_input: str):
    """
    Главная точка входа для обработки пользовательского ввода

    :param user_input: Ввод пользователя
    :return: Результат вычисления
    """
    if not user_input:
        return "Пожалуйста, введите математическое выражение."

    router = CommandRouter()

    # Извлекаем команду
    command, expression = router.extract_command(user_input)

    if command == "error":
        return expression

    # Обрабатываем команду
    return router.process_command(command, expression)


@math_error_handler
def smart_display_implicit_2d(result):
    """Отображение неявной 2D кривой F(x,y) = 0"""
    import numpy as np
    import matplotlib.pyplot as plt
    from sympy import symbols, sympify, lambdify, solve

    expr_str = result['expression']
    var_names = result['variables']

    x_sym, y_sym = symbols(f'{var_names[0]} {var_names[1]}')
    equation = sympify(expr_str)

    # Пытаемся найти характерный размер кривой
    try:
        # Решаем уравнение для y=0
        x_intercepts = solve(equation.subs(y_sym, 0), x_sym)
        x_vals = [float(val.evalf()) for val in x_intercepts if val.is_real]

        # Решаем для x=0
        y_intercepts = solve(equation.subs(x_sym, 0), y_sym)
        y_vals = [float(val.evalf()) for val in y_intercepts if val.is_real]

        if x_vals and y_vals:
            x_max = max(abs(v) for v in x_vals) * 1.5
            y_max = max(abs(v) for v in y_vals) * 1.5
        else:
            x_max = y_max = 10
    except:
        x_max = y_max = 10

    # Создаём сетку
    x_range = np.linspace(-x_max, x_max, 500)
    y_range = np.linspace(-y_max, y_max, 500)
    X, Y = np.meshgrid(x_range, y_range)

    # Вычисляем F(x,y)
    f = lambdify((x_sym, y_sym), equation, 'numpy')

    try:
        Z = f(X, Y)
    except Exception as e:
        print(f"Ошибка вычисления: {e}")
        return None

    # Рисуем
    fig, ax = plt.subplots(figsize=(8, 8))

    # Основная кривая F(x,y) = 0
    contour = ax.contour(X, Y, Z, levels=[0], colors='blue', linewidths=2.5)

    # Закрашиваем область F(x,y) < 0
    ax.contourf(X, Y, Z, levels=[-1e10, 0], colors=['lightblue'], alpha=0.2)

    # Координатные оси
    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='black', linewidth=0.5, alpha=0.5)

    # Сетка
    ax.grid(True, alpha=0.3, linestyle='--')

    # Подписи
    ax.set_xlabel(var_names[0], fontsize=12)
    ax.set_ylabel(var_names[1], fontsize=12)
    ax.set_title(f"Неявная кривая: {result['original']}", fontsize=14, fontweight='bold')

    # Равные масштабы (критично для окружностей!)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()

    return "График успешно построен"
