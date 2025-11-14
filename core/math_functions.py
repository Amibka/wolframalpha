# ==========================
# Математические функции
# ==========================
math_functions = [
    # тригонометрические (прямые и обратные)
    "sin", "cos", "tan", "cot", "sec", "csc",
    "asin", "acos", "atan", "acot", "asec", "acsc",
    "tg", "ctg",

    # гиперболические (и некоторые обратные)
    "sinh", "cosh", "tanh", "coth", "sech", "csch",
    "asinh", "acosh", "atanh",

    # экспоненты / логарифмы / степени / корни
    "exp", "log", "ln",  # ln — алиас для log в SymPy
    "sqrt", "root",  # sqrt(x) и root(x, n)
    "pow",  # степень (обычно x**y, но есть вспомогательные функции)

    # базовые числовые функции
    "Abs", "sign", "floor", "ceiling", "factorial",

    # специальные функции (часто используются как sympy.<name>)
    "gamma", "loggamma", "digamma", "polygamma",
    "beta", "zeta", "erf", "erfc",
    "besselj", "bessely", "besseli", "besselk",

    # комплексные/утилитарные математические
    "re", "im", "arg", "conjugate",

    'e', 'E', 'i', 'I', 'pi', 'sin', 'cos', 'tan', 'log', 'ln', 'exp', 'oo', 'integral'
]
