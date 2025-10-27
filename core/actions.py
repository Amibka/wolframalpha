import csv

# Список примеров: человек -> SymPy
examples = [
    ("4x + 1 = 10", "4*x + 1 - 10"),
    ("x^2 - 5x + 6 = 0", "x**2 - 5*x + 6"),
    ("2^3 + 5 = 13", "2**3 + 5 - 13"),
    ("7x - 3 = 11", "7*x - 3 - 11"),

    # Логарифмы
    ("ln(x) = 2", "ln(x) - 2"),
    ("log(x) = 3", "ln(x) - 3"),
    ("log2(x) = 5", "log(x, 2) - 5"),
    ("log_2(x) = 5", "log(x, 2) - 5"),
    ("log10(x) = 3", "log(x, 10) - 3"),
    ("log_10(x) = 3", "log(x, 10) - 3"),

    # Тригонометрия
    ("sin(x) = 0", "sin(x) - 0"),
    ("cos(x) = 0", "cos(x) - 0"),
    ("tan(x) = 0", "tan(x) - 0"),
    ("tg(x) = 0", "tan(x) - 0"),
    ("cot(x) = 0", "cot(x) - 0"),
    ("ctg(x) = 0", "cot(x) - 0"),

    # Экспоненты
    ("e^x = 1", "exp(x) - 1"),
    ("2^x = 8", "2**x - 8"),

    # Степени и корни
    ("sqrt(x) = 4", "sqrt(x) - 4"),
    ("x^(1/2) = 4", "x**(1/2) - 4"),
    ("x^3 = 8", "x**3 - 8"),

    # Производные
    ("d/dx(x^3 + 2x^2 + x)", "diff(x**3 + 2*x**2 + x, x)"),

    # Интегралы
    ("∫(x^2 + 3x + 1) dx", "integrate(x**2 + 3*x + 1, x)"),

    # Константы
    ("pix = 3", "pi*x - 3"),
    ("ex = 2", "E*x - 2"),

    # Комбинированные выражения
    ("sin(x) + ln(x) - x^2 = 0", "sin(x) + ln(x) - x**2 - 0"),
    ("tg(x) + ctg(x) = 2", "tan(x) + cot(x) - 2"),
]

# Запись в CSV
with open('sympy_all_functions.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Human", "SymPy"])
    writer.writerows(examples)

print("CSV файл создан: sympy_all_functions.csv")