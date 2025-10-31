import json
import re

import sympy

from core.actions import actions
from core.math_functions import math_functions
from core.sympy_solver import derivative, solve_equation
from logs.logger import log_call
from utils.suggest_correction import suggest_correction


@log_call
def parse_user_input(expression: str):
    """
    Анализирует ввод пользователя и выполняет соответствующее математическое действие.
    
    :param user_input: Ввод пользователя в виде строки.
    :return: tuple (обработанное выражение, локальный словарь для sympy)
    """

    @log_call
    def insert_multiplication(expression: str) -> str:
        if not expression:
            return ""

        # 1) число перед переменной или скобкой
        expression = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', expression)

        # 2) переменная перед переменной или скобкой
        # чтобы не ломать функции, ищем только одиночные буквы
        pattern = r'(?<![a-zA-Z])([a-zA-Z])([a-zA-Z(])'

        def repl_var(match):
            first = match.group(1)
            second = match.group(2)

            # проверяем, не начинается ли с функции
            rest = first + expression[match.end() - 1:]  # остаток строки после match
            if any(rest.startswith(func) for func in math_functions):
                return first + second  # не вставляем *
            return f"{first}*{second}"

        expression = re.sub(pattern, repl_var, expression)

        if '^' in expression:
            expression = expression.replace('^', '**')

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
        COMMAND_TRANSLATE  = json.load(f)

    def get_first_word(user_input):
        user_input = user_input.lower().strip()
        for first_word, synonyms in COMMAND_TRANSLATE.items():
            for synonym in synonyms:
                if user_input.startswith(synonym):
                    # заменяем синоним на основную команду
                    expression = user_input[len(synonym):].strip()
                    return first_word, expression
        return "solve", user_input


    #Команда по умолчанию
    first_word, expression = get_first_word(user_input)

    # проверка, есть ли явная команда в начале строки
    parts = user_input.strip().split(' ', 1)  # разделяем на слово и остальное
    if parts[0] in actions and len(parts) > 1:
        first_word = parts[0]  # команда явно указана
        expression = parts[1]

    # Убираем = так как нельзя подставить в некоторые функции
    if '=' in expression and first_word != 'solve':
        expression = expression.split('=')[0].strip()

    # Словарь действий и соответствующих функций
    action_map = {
        "solve": lambda expr: solve_expression(expr),
        "derivative": lambda expr: derivative_expression(expr),
        "simplify": lambda expr: sympy.simplify(expr),
        "expand": lambda expr: sympy.expand(expr),
        "expand_trig": lambda expr: sympy.trigsimp(expr),
        "expand_log": lambda expr: sympy.log(expr),
        "expand_power_exp": lambda expr: sympy.exp(expr),
        "factor": lambda expr: sympy.factor(expr),
        "factorint": lambda expr: sympy.factorint(expr),
        "collect": lambda expr: sympy.collect(expr),
        "cancel": lambda expr: sympy.cancel(expr),
        "together": lambda expr: sympy.together(expr),
        "apart": lambda expr: sympy.apart(expr),
        "radsimp": lambda expr: sympy.radsimp(expr),
        "powsimp": lambda expr: sympy.powsimp(expr),
        "logcombine": lambda expr: sympy.logcombine(expr),
        "nsimplify": lambda expr: sympy.nsimplify(expr),
        "sqrtdenest": lambda expr: sympy.sqrtdenest(expr),
        "residue": lambda expr: sympy.residue(expr),
        "ratsimp": lambda expr: sympy.ratsimp(expr),
        "cancel_common_factors": lambda expr: sympy.cancel_common_factors(expr),
        "factor_terms": lambda expr: sympy.factor_terms(expr),
        "simplify_rational": lambda expr: sympy.simplify(expr),
        "simplify_logic": lambda expr: sympy.simplify(expr),
        "cse": lambda expr: sympy.cse(expr),
        "separatevars": lambda expr: sympy.separatevars(expr),
        "logcombine": lambda expr: sympy.logcombine(expr),
        "expand_complex": lambda expr: sympy.expand_complex(expr),
        "simplify_fraction": lambda expr: sympy.simplify(expr),
        "denest": lambda expr: sympy.denest(expr),
        "together_cancel": lambda expr: sympy.together(expr),







        # можно добавлять новые действия сюда
    }

    # Если действие есть в словаре
    if first_word in action_map:
        try:
            return action_map[first_word](expression)
        except Exception as e:
            return f"Ошибка при выполнении {first_word}: {e}"

    # Если действия нет — подсказка
    suggestion = suggest_correction(first_word, actions)
    return f'Неизвестное действие: {first_word}, возможно вы имели в виду {suggestion}?'


# Обёртки для конкретных действий
def solve_expression(expression: str):
    parsed_expression, local_dict = parse_user_input(expression)
    print(f"Parsed expression for solving: {parsed_expression}")
    return solve_equation(parsed_expression, local_dict=local_dict)


def derivative_expression(expression: str):
    parsed_expression, local_dict = parse_user_input(expression)
    print(f"Parsed expression for derivative: {parsed_expression}")
    return derivative(parsed_expression, local_dict=local_dict)  # если понадобится, можно добавить local_dict
