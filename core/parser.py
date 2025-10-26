from core.sympy_solver import derivative, solve_equation
from core.actions import actions
from core.math_functions import math_functions
from logs.logger import log_call
from utils.suggest_correction import suggest_correction
import re
import sympy


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

    first_word = user_input.split(' ', 1)[0]
    expression = user_input[len(first_word):].strip()

    # Словарь действий и соответствующих функций
    action_map = {
        "solve": lambda expr: solve_expression(expr),
        "derivative": lambda expr: derivative_expression(expr),
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
