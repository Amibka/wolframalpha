import json
import re

import sympy
from sympy.abc import a, b, c, d, e, f, g, h, i, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z

from core.actions import actions, actions_ru
from core.math_functions import math_functions

from core.sympy_solver import derivative, solve_equation, calculation_residue, poly_func, degree_func, rem_func, \
    quo_func, div_func, lcm_func, gcd_func, separatevars_func, expand_complex_func, factor_terms_func, sqrtdenest_func, \
    nsimplify_func, logcombine_func, ratsimp_func, radsimp_func, powsimp_func, trigsimp_func, collect_func, apart_func, \
    together_func, cancel_func, factor_func, expand_func, simplify_func, LM_func, LC_func, primitive_func, content_func, \
    LT_func, sqf_list_func

from logs.logger import log_call
from utils.suggest_correction import suggest_correction, suggest_correction_ru


@log_call
def parse_user_input(expression: str):
    """
    Анализирует ввод пользователя и выполняет соответствующее математическое действие.
    
    :param user_input: Ввод пользователя в виде строки.
    :return: tuple (обработанное выражение, локальный словарь для sympy)
    """

    @log_call
    def insert_multiplication(expression: str, commands=None) -> str:
        if not expression:
            return ""

        commands = commands or ['derivative', 'residue', 'solve']

        # 1) проверяем, начинается ли строка с командой
        prefix = ""
        for cmd in commands:
            if expression.startswith(cmd):
                prefix = cmd
                expression = expression[len(cmd):].strip()
                break

        # 2) обычная вставка *
        expression = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', expression)

        pattern = r'(?<![a-zA-Z])([a-zA-Z])([a-zA-Z(])'

        def repl_var(match):
            first = match.group(1)
            second = match.group(2)
            rest = first + expression[match.end() - 1:]
            if any(rest.startswith(func) for func in math_functions):
                return first + second
            return f"{first}*{second}"

        expression = re.sub(pattern, repl_var, expression)

        if '^' in expression:
            expression = expression.replace('^', '**')

        # 3) возвращаем команду + обработанную строку
        if prefix:
            return f"{prefix} {expression}"
        return expression

    # Замена имен функций на sympy.имя_функции
    @log_call
    def replace_math_functions(expression: str) -> str:
        if not expression:
            return ""

        # Просто проверяем, что функции пишутся правильно, без добавления "sympy."
        sorted_funcs = sorted(math_functions, key=len, reverse=True)

        for func in sorted_funcs:
            pattern = r'\b' + re.escape(func) + r'\b'
            expression = re.sub(pattern, func, expression)

        return expression

    def replace_custom_log(expression: str) -> str:
        """
        Преобразует пользовательский синтаксис loga(x) в SymPy log(x, a)
        """
        if not expression:
            return ""

        # Паттерн: loga(x) или log12(y)
        pattern = r'log([a-zA-Z0-9_]+)\(([^()]+)\)'

        def repl(match):
            base = match.group(1)
            arg = match.group(2)
            return f'log({arg}, {base})'  # SymPy синтаксис

        expression = re.sub(pattern, repl, expression)
        return expression

    if 'log' in expression:
        expression = replace_custom_log(expression)

    expression = insert_multiplication(expression)
    expression = replace_math_functions(expression)

    # Создаём локальный словарь для parse_expr
    local_dict = {name: getattr(sympy, name, None) for name in math_functions}
    local_dict.update({
        "pi": sympy.pi,
        "E": sympy.E,
        "oo": sympy.oo,
        "tg": sympy.tan,
        "ctg": sympy.cot,
        "ln": sympy.ln
    })
    # Убираем None
    local_dict = {k: v for k, v in local_dict.items() if v is not None}

    return expression, local_dict


@log_call
def get_text(user_input: str):
    if not user_input:
        return "Пожалуйста, введите математическое выражение."

    #    for action in actions:
    #        if action in user_input:
    #            first_word = user_input.split(' ', 1)[0]
    #            expression = user_input[len(first_word):].strip()
    #        else:
    #            first_word = action
    #            expression = expression

    with open("language/commands_translate.json", "r", encoding="utf-8") as f:
        COMMAND_TRANSLATE = json.load(f)

    def get_first_word(user_input):
        user_input_original = user_input
        user_input = user_input.lower().strip()

        # Сначала проверяем точное совпадение с командами из COMMAND_TRANSLATE
        for first_word, synonyms in COMMAND_TRANSLATE.items():
            # ВАЖНО: Сортируем синонимы по длине (от длинных к коротким)
            # Чтобы "решить" проверялся раньше, чем "реши"
            sorted_synonyms = sorted(synonyms, key=len, reverse=True)

            for synonym in sorted_synonyms:
                # Проверяем, что после синонима либо конец строки, либо пробел
                if user_input.startswith(synonym):
                    # Проверяем, что это именно команда, а не часть слова
                    next_char_pos = len(synonym)
                    if next_char_pos >= len(user_input) or user_input[next_char_pos].isspace():
                        # заменяем синоним на основную команду
                        expression = user_input[len(synonym):].strip()
                        return first_word, expression

        # Если команда не найдена, пробуем найти похожую
        first_word_input = user_input.split()[0] if user_input.split() else user_input

        # Определяем язык по первому слову
        # Проверяем, содержит ли слово кириллицу
        is_russian = any('а' <= c <= 'я' or 'А' <= c <= 'Я' for c in first_word_input)

        if is_russian:
            suggestion = suggest_correction_ru(first_word_input, actions_ru)
            if suggestion:
                return "error", f'Неизвестное действие: "{first_word_input}", возможно вы имели в виду "{suggestion}"?'
        else:
            suggestion = suggest_correction(first_word_input, actions)
            if suggestion:
                return "error", f'Неизвестное действие: "{first_word_input}", возможно вы имели в виду "{suggestion}"?'

        # Если не нашли похожую команду, используем solve по умолчанию
        return "solve", user_input

    # Команда по умолчанию
    first_word, expression = get_first_word(user_input)

    if first_word == "error":
        return expression  # expression содержит сообщение об ошибке

    # проверка, есть ли явная команда в начале строки
    parts = user_input.strip().split(' ', 1)
    if parts[0] in actions and len(parts) > 1:
        first_word = parts[0]
        expression = parts[1]


    # Убираем = так как нельзя подставить в некоторые функции
    if '=' in expression and first_word != 'solve':
        expression = expression.split('=')[0].strip()

    action_map = {
        # Основные операции
        "solve": lambda expr: solve_expression(expr),
        "derivative": lambda expr: derivative_expression(expr),
        "residue": lambda expr: compute_residue(expr),

        # Упрощение
        "simplify": lambda expr: simplify_expression(expr),
        "expand": lambda expr: expand_expression(expr),
        "factor": lambda expr: factor_expression(expr),
        "cancel": lambda expr: cancel_expression(expr),
        "together": lambda expr: together_expression(expr),
        "apart": lambda expr: apart_expression(expr),
        "collect": lambda expr: collect_expression(expr),

        # Тригонометрия и степени
        "trigsimp": lambda expr: trigsimp_expression(expr),
        "powsimp": lambda expr: powsimp_expression(expr),
        "radsimp": lambda expr: radsimp_expression(expr),

        # Рациональные выражения
        "ratsimp": lambda expr: ratsimp_expression(expr),
        "logcombine": lambda expr: logcombine_expression(expr),
        "nsimplify": lambda expr: nsimplify_expression(expr),
        "sqrtdenest": lambda expr: sqrtdenest_expression(expr),
        "factor_terms": lambda expr: factor_terms_expression(expr),
        "expand_complex": lambda expr: expand_complex_expression(expr),
        "separatevars": lambda expr: separatevars_expression(expr),

        # Операции с несколькими аргументами
        "advanced.gcd": lambda expr: gcd_expression(expr),
        "advanced.lcm": lambda expr: lcm_expression(expr),
        "advanced.div": lambda expr: div_expression(expr),
        "advanced.quo": lambda expr: quo_expression(expr),
        "advanced.rem": lambda expr: rem_expression(expr),


        "advanced.Poly": lambda expr: poly_expression(expr),
        "advanced.degree": lambda expr: degree_expression(expr),
        "advanced.content": lambda expr: content_expression(expr),
        "advanced.primitive": lambda expr: primitive_expression(expr),
    }

    # Если действие есть в словаре
    if first_word in action_map:
        try:
            return action_map[first_word](expression)
        except Exception as e:
            return f"Ошибка при выполнении {first_word}: {e}"

    # Если действия нет — подсказка



# ============================================
# УНИВЕРСАЛЬНАЯ ФУНКЦИЯ ДЛЯ ИЗВЛЕЧЕНИЯ ПЕРЕМЕННОЙ
# ============================================

def extract_variable(expression: str, keywords=None, required=False, auto_detect=True):
    """
    Универсальная функция для извлечения переменной из выражения.

    :param expression: Выражение с переменной
    :param keywords: Список ключевых слов (по умолчанию ['по', 'at', 'by', 'in'])
    :param required: Если True и переменная не найдена, вернёт ошибку
    :param auto_detect: Если True, попытается определить переменную автоматически
    :return: (variable, clean_expression, error_message)
             variable - найденная переменная или None
             clean_expression - выражение без ключевых слов
             error_message - сообщение об ошибке или None
    """
    if keywords is None:
        keywords = ['по', 'at', 'by', 'in']

    # ШАГ 1: Ищем "keyword <variable>"
    pattern = r'\b(' + '|'.join(keywords) + r')\s+([a-zA-Zа-яА-Я])\b'
    match = re.search(pattern, expression, re.IGNORECASE)

    if match:
        variable = match.group(2)
        # Убираем "keyword <variable>" из строки
        clean_expr = re.sub(pattern, '', expression, flags=re.IGNORECASE).strip()
        return variable, clean_expr, None

    # ШАГ 2: Проверяем "висячие" ключевые слова (например "по" без переменной)
    orphan_pattern = r'\b(' + '|'.join(keywords) + r')(\s*$|\s+(?![a-zA-Zа-яА-Я]))'
    if re.search(orphan_pattern, expression, re.IGNORECASE):
        # Убираем "висячее" ключевое слово
        clean_expr = re.sub(r'\b(' + '|'.join(keywords) + r')\s*$', '', expression, flags=re.IGNORECASE).strip()

        if required:
            keyword_found = re.search(orphan_pattern, expression, re.IGNORECASE).group(1)
            error_msg = f"Указано '{keyword_found}', но не указана переменная. Используйте: '{keyword_found} x'"
            return None, clean_expr, error_msg

        return None, clean_expr, None

    if auto_detect:
        # Ищем все одиночные буквы в выражении (в том числе приклеенные к цифрам)
        excluded = math_functions

        # Улучшенный паттерн: находит буквы даже если они приклеены к цифрам/операторам
        # Исключаем буквы внутри длинных слов (функции типа sin, cos, log)
        var_pattern = r'(?<![a-zA-Z])([a-zA-Z])(?![a-zA-Z])'
        variables = re.findall(var_pattern, expression)

        # Фильтруем переменные
        valid_vars = [v for v in set(variables) if v not in excluded]

        if valid_vars:
            # Приоритет: x -> y -> z -> первая найденная
            if 'x' in valid_vars:
                return 'x', expression, None
            elif 'y' in valid_vars:
                return 'y', expression, None
            elif 'z' in valid_vars:
                return 'z', expression, None
            else:
                return valid_vars[0], expression, None


    # ШАГ 3: Ключевого слова нет
    if required and not auto_detect:
        keywords_str = "', '".join(keywords)
        error_msg = f"Требуется указать переменную через '{keywords_str}'"
        return None, expression, error_msg

    return None, expression, None


# ============================================
# WRAPPER ФУНКЦИИ - УПРОЩЕННЫЕ
# ============================================

def solve_expression(expression: str):
    """
    Обёртка для solve с автоматическим извлечением переменной.
    """
    # Извлекаем переменную через универсальную функцию
    variable, clean_expr, error = extract_variable(
        expression,
        keywords=['по', 'at', 'by', 'in'],
        auto_detect=True  # Включаем автоопределение
    )

    if error:
        return f"Ошибка: {error}"

    # Парсим очищенное выражение
    parsed_expression, local_dict = parse_user_input(clean_expr)

    print(f"Parsed expression for solving: {parsed_expression}")
    print(f"Variable: {variable}")

    # Передаём найденную переменную (может быть None)
    return solve_equation(parsed_expression, variable, local_dict=local_dict)


def derivative_expression(expression: str):
    """
    Обёртка для вычисления производной с предобработкой выражения.
    Поддерживает: "по x", "at x", "by x", "in x" (рус/eng)
    """
    if not expression:
        return "Пустое выражение"

    # Извлекаем переменную ДО парсинга
    variable, clean_expr, error = extract_variable(expression, ['по', 'at', 'by', 'in'])

    if error:
        return f"Ошибка: {error}"

    print(f"Original expression: {expression}")
    print(f"Cleaned expression: {clean_expr}")
    print(f"Extracted variable: {variable}")

    # Парсим очищенное выражение
    parsed_expr, local_dict = parse_user_input(clean_expr)

    # Если переменная была извлечена, добавляем её обратно
    if variable:
        expr_with_var = f"{parsed_expr} по {variable}"
    else:
        expr_with_var = parsed_expr

    print(f"Parsed expression for derivative: {expr_with_var}")
    return derivative(expr_with_var, local_dict=local_dict)


def compute_residue(expression: str):
    parsed_expression, local_dict = parse_user_input(expression)
    print(f"Parsed expression for residue: {parsed_expression}")
    return calculation_residue(parsed_expression, local_dict=local_dict)


# === Простые функции (без переменных) ===

def simplify_expression(expression: str):
    parsed_expression, local_dict = parse_user_input(expression)
    print(f"Parsed expression for simplify: {parsed_expression}")
    return factor_func(simplify_func(parsed_expression, local_dict=local_dict))


def expand_expression(expression: str):
    parsed_expression, local_dict = parse_user_input(expression)
    print(f"Parsed expression for expand: {parsed_expression}")
    return expand_func(parsed_expression, local_dict=local_dict)


def factor_expression(expression: str):
    parsed_expression, local_dict = parse_user_input(expression)
    print(f"Parsed expression for factor: {parsed_expression}")
    return factor_func(parsed_expression, local_dict=local_dict)


def cancel_expression(expression: str):
    parsed_expression, local_dict = parse_user_input(expression)
    print(f"Parsed expression for cancel: {parsed_expression}")
    return cancel_func(parsed_expression, local_dict=local_dict)


def together_expression(expression: str):
    parsed_expression, local_dict = parse_user_input(expression)
    print(f"Parsed expression for together: {parsed_expression}")
    return together_func(parsed_expression, local_dict=local_dict)


def apart_expression(expression: str):
    parsed_expression, local_dict = parse_user_input(expression)
    print(f"Parsed expression for apart: {parsed_expression}")
    return apart_func(parsed_expression, local_dict=local_dict)


def trigsimp_expression(expression: str):
    parsed_expression, local_dict = parse_user_input(expression)
    print(f"Parsed expression for trigsimp: {parsed_expression}")
    return trigsimp_func(parsed_expression, local_dict=local_dict)


def powsimp_expression(expression: str):
    parsed_expression, local_dict = parse_user_input(expression)
    print(f"Parsed expression for powsimp: {parsed_expression}")
    return powsimp_func(parsed_expression, local_dict=local_dict)


def radsimp_expression(expression: str):
    parsed_expression, local_dict = parse_user_input(expression)
    print(f"Parsed expression for radsimp: {parsed_expression}")
    return radsimp_func(parsed_expression, local_dict=local_dict)


def ratsimp_expression(expression: str):
    parsed_expression, local_dict = parse_user_input(expression)
    print(f"Parsed expression for ratsimp: {parsed_expression}")
    return ratsimp_func(parsed_expression, local_dict=local_dict)


def logcombine_expression(expression: str):
    parsed_expression, local_dict = parse_user_input(expression)
    print(f"Parsed expression for logcombine: {parsed_expression}")
    return logcombine_func(parsed_expression, local_dict=local_dict)


def nsimplify_expression(expression: str):
    parsed_expression, local_dict = parse_user_input(expression)
    print(f"Parsed expression for nsimplify: {parsed_expression}")
    return nsimplify_func(parsed_expression, local_dict=local_dict)


def sqrtdenest_expression(expression: str):
    parsed_expression, local_dict = parse_user_input(expression)
    print(f"Parsed expression for sqrtdenest: {parsed_expression}")
    return sqrtdenest_func(parsed_expression, local_dict=local_dict)


def factor_terms_expression(expression: str):
    parsed_expression, local_dict = parse_user_input(expression)
    print(f"Parsed expression for factor_terms: {parsed_expression}")
    return factor_terms_func(parsed_expression, local_dict=local_dict)


def expand_complex_expression(expression: str):
    parsed_expression, local_dict = parse_user_input(expression)
    print(f"Parsed expression for expand_complex: {parsed_expression}")
    return expand_complex_func(parsed_expression, local_dict=local_dict)


def separatevars_expression(expression: str):
    parsed_expression, local_dict = parse_user_input(expression)
    print(f"Parsed expression for separatevars: {parsed_expression}")
    return separatevars_func(parsed_expression, local_dict=local_dict)


# === Функции с переменными ===

def collect_expression(expression: str):
    """
    Collect - группировка по переменной.
    Поддерживает: "по x", "at x", "by x", "in x"
    Пример: collect x**2 + 2*x + x*y по x
    """
    # Извлекаем переменную ДО парсинга
    variable, clean_expr, error = extract_variable(expression, ['по', 'at', 'by', 'in'])

    if error:
        return f"Ошибка: {error}"

    parsed_expression, local_dict = parse_user_input(clean_expr)

    if variable:
        expr_with_var = f"{parsed_expression} по {variable}"
    else:
        # Без переменной - передаём как есть, функция сама определит или выдаст ошибку
        expr_with_var = parsed_expression

    print(f"Parsed expression for collect: {expr_with_var}")
    return collect_func(expr_with_var, local_dict=local_dict)


def degree_expression(expression: str):
    """
    Degree - степень многочлена.
    Поддерживает: "по x", "at x", "by x", "in x"
    Пример: degree x**2 + 2*x по x
    """
    variable, clean_expr, error = extract_variable(expression, ['по', 'at', 'by', 'in'])

    if error:
        return f"Ошибка: {error}"

    parsed_expression, local_dict = parse_user_input(clean_expr)

    if variable:
        expr_with_var = f"{parsed_expression} по {variable}"
    else:
        expr_with_var = parsed_expression

    print(f"Parsed expression for degree: {expr_with_var}")
    return degree_func(expr_with_var, local_dict=local_dict)


# === Функции с несколькими аргументами ===

def gcd_expression(expression: str):
    """
    GCD требует два или более аргумента через запятую.
    Пример: gcd 12, 18 или gcd x**2, x**3
    """
    if ',' not in expression:
        return "Ошибка: gcd требует минимум два аргумента через запятую"

    parts = [p.strip() for p in expression.split(',')]
    parsed_parts = []
    for part in parts:
        parsed, local_dict = parse_user_input(part)
        parsed_parts.append(parsed)

    combined_expr = ', '.join(parsed_parts)
    print(f"Parsed expression for gcd: {combined_expr}")
    return gcd_func(combined_expr, local_dict=local_dict)


def lcm_expression(expression: str):
    """
    LCM требует два или более аргумента через запятую.
    Пример: lcm 12, 18
    """
    if ',' not in expression:
        return "Ошибка: lcm требует минимум два аргумента через запятую"

    parts = [p.strip() for p in expression.split(',')]
    parsed_parts = []
    for part in parts:
        parsed, local_dict = parse_user_input(part)
        parsed_parts.append(parsed)

    combined_expr = ', '.join(parsed_parts)
    print(f"Parsed expression for lcm: {combined_expr}")
    return lcm_func(combined_expr, local_dict=local_dict)


def div_expression(expression: str):
    """
    Div требует два аргумента через запятую.
    Пример: div x**2 + 2*x, x
    """
    if ',' not in expression:
        return "Ошибка: div требует два аргумента через запятую"

    parts = expression.split(',', 1)
    expr1 = parts[0].strip()
    expr2 = parts[1].strip()

    parsed_expr1, local_dict = parse_user_input(expr1)
    parsed_expr2, _ = parse_user_input(expr2)

    combined_expr = f"{parsed_expr1}, {parsed_expr2}"
    print(f"Parsed expression for div: {combined_expr}")
    return div_func(combined_expr, local_dict=local_dict)


def quo_expression(expression: str):
    """
    Quo требует два аргумента через запятую.
    Пример: quo x**2 + 2*x, x
    """
    if ',' not in expression:
        return "Ошибка: quo требует два аргумента через запятую"

    parts = expression.split(',', 1)
    expr1 = parts[0].strip()
    expr2 = parts[1].strip()

    parsed_expr1, local_dict = parse_user_input(expr1)
    parsed_expr2, _ = parse_user_input(expr2)

    combined_expr = f"{parsed_expr1}, {parsed_expr2}"
    print(f"Parsed expression for quo: {combined_expr}")
    return quo_func(combined_expr, local_dict=local_dict)


def rem_expression(expression: str):
    """
    Rem требует два аргумента через запятую.
    Пример: rem x**2 + 2*x, x
    """
    if ',' not in expression:
        return "Ошибка: rem требует два аргумента через запятую"

    parts = expression.split(',', 1)
    expr1 = parts[0].strip()
    expr2 = parts[1].strip()

    parsed_expr1, local_dict = parse_user_input(expr1)
    parsed_expr2, _ = parse_user_input(expr2)

    combined_expr = f"{parsed_expr1}, {parsed_expr2}"
    print(f"Parsed expression for rem: {combined_expr}")
    return rem_func(combined_expr, local_dict=local_dict)


def poly_expression(expression: str):
    parsed_expression, local_dict = parse_user_input(expression)
    print(f"Parsed expression for poly: {parsed_expression}")
    return poly_func(parsed_expression, local_dict=local_dict)


def content_expression(expression: str):
    """
    Content - содержимое многочлена (общий делитель коэффициентов).
    Пример: content 6*x**2 + 9*x → 3
    """
    variable, clean_expr, error = extract_variable(expression, ['по', 'at', 'by', 'in'])

    if error:
        return f"Ошибка: {error}"

    parsed_expression, local_dict = parse_user_input(clean_expr)

    if variable:
        expr_with_var = f"{parsed_expression} по {variable}"
    else:
        expr_with_var = parsed_expression

    print(f"Parsed expression for content: {expr_with_var}")
    return content_func(expr_with_var, local_dict=local_dict)


def primitive_expression(expression: str):
    """
    Primitive - примитивная часть многочлена (многочлен / content).
    Пример: primitive 6*x**2 + 9*x → (3, 2*x**2 + 3*x)
    """
    variable, clean_expr, error = extract_variable(expression, ['по', 'at', 'by', 'in'])

    if error:
        return f"Ошибка: {error}"

    parsed_expression, local_dict = parse_user_input(clean_expr)

    if variable:
        expr_with_var = f"{parsed_expression} по {variable}"
    else:
        expr_with_var = parsed_expression

    print(f"Parsed expression for primitive: {expr_with_var}")
    return primitive_func(expr_with_var, local_dict=local_dict)
