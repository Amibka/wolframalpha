"""
Генератор датасета для ВСЕХ функций SymPy из документа
Создаёт примеры: естественный язык (RU/EN) → SymPy код
"""
import json
import random
from typing import List, Dict


class SymPyDatasetGenerator:
    def __init__(self):
        self.vars = ['x', 'y', 'z', 't', 'a', 'b', 'n', 'm']
        self.nums = [0, 1, 2, 3, 4, 5, 10, -1, -2, -5]

        # ПОЛНАЯ БАЗА ВСЕХ SYMPY ФУНКЦИЙ С ШАБЛОНАМИ
        self.templates = {
            # ============ 1. ALGEBRA & SIMPLIFICATION ============
            "simplify": [
                ("упрости {expr}", "en", "simplify {expr}", "simplify({expr})"),
                ("упростить {expr}", "en", "simplify {expr}", "simplify({expr})"),
                ("сократи {expr}", "en", "reduce {expr}", "simplify({expr})"),
            ],
            "expand": [
                ("раскрой {expr}", "en", "expand {expr}", "expand({expr})"),
                ("раскрой скобки {expr}", "en", "expand brackets {expr}", "expand({expr})"),
                ("раскрыть {expr}", "en", "open {expr}", "expand({expr})"),
            ],
            "expand_trig": [
                ("раскрой тригонометрию {expr}", "en", "expand trig {expr}", "expand_trig({expr})"),
            ],
            "factor": [
                ("разложи на множители {expr}", "en", "factor {expr}", "factor({expr})"),
                ("факторизуй {expr}", "en", "factorize {expr}", "factor({expr})"),
                ("разложи {expr}", "en", "factor out {expr}", "factor({expr})"),
            ],
            "collect": [
                ("собери по {var} в {expr}", "en", "collect {expr} by {var}", "collect({expr}, {var})"),
            ],
            "cancel": [
                ("сократи дробь {expr}", "en", "cancel {expr}", "cancel({expr})"),
            ],
            "apart": [
                ("разложи на простые дроби {expr}", "en", "partial fractions {expr}", "apart({expr})"),
            ],
            "together": [
                ("приведи к общему знаменателю {expr}", "en", "combine fractions {expr}", "together({expr})"),
            ],

            # ============ 2. POLYNOMIALS ============
            "degree": [
                ("степень многочлена {expr}", "en", "degree of {expr}", "degree({expr})"),
            ],
            "gcd": [
                ("нод {expr1} и {expr2}", "en", "gcd of {expr1} and {expr2}", "gcd({expr1}, {expr2})"),
            ],
            "lcm": [
                ("нок {expr1} и {expr2}", "en", "lcm of {expr1} and {expr2}", "lcm({expr1}, {expr2})"),
            ],
            "discriminant": [
                ("дискриминант {expr}", "en", "discriminant of {expr}", "discriminant({expr}, {var})"),
            ],

            # ============ 3. CALCULUS ============
            "diff": [
                ("производная {expr}", "en", "derivative of {expr}", "diff({expr}, {var})"),
                ("производная {expr} по {var}", "en", "derivative of {expr} wrt {var}", "diff({expr}, {var})"),
                ("дифференцируй {expr}", "en", "differentiate {expr}", "diff({expr}, {var})"),
                ("найди производную {expr}", "en", "find derivative of {expr}", "diff({expr}, {var})"),
            ],
            "diff_nth": [
                ("производная {n} порядка {expr}", "en", "{n}th derivative of {expr}", "diff({expr}, {var}, {n})"),
                ("{n}-я производная {expr}", "en", "{n}th derivative of {expr}", "diff({expr}, {var}, {n})"),
            ],
            "integrate": [
                ("интеграл {expr}", "en", "integral of {expr}", "integrate({expr}, {var})"),
                ("проинтегрируй {expr}", "en", "integrate {expr}", "integrate({expr}, {var})"),
                ("найди интеграл {expr}", "en", "find integral of {expr}", "integrate({expr}, {var})"),
            ],
            "integrate_definite": [
                ("интеграл {expr} от {a} до {b}", "en", "integral of {expr} from {a} to {b}",
                 "integrate({expr}, ({var}, {a}, {b}))"),
                ("определённый интеграл {expr} от {a} до {b}", "en", "definite integral of {expr} from {a} to {b}",
                 "integrate({expr}, ({var}, {a}, {b}))"),
            ],
            "limit": [
                ("предел {expr} при {var} стремится к {point}", "en", "limit of {expr} as {var} approaches {point}",
                 "limit({expr}, {var}, {point})"),
                ("предел {expr} при {var} -> {point}", "en", "limit of {expr} as {var} -> {point}",
                 "limit({expr}, {var}, {point})"),
                ("lim {expr} при {var} -> {point}", "en", "lim {expr} as {var} -> {point}",
                 "limit({expr}, {var}, {point})"),
            ],
            "series": [
                ("разложи в ряд {expr} около {point}", "en", "series expansion of {expr} at {point}",
                 "series({expr}, {var}, {point})"),
                ("ряд тейлора {expr}", "en", "taylor series of {expr}", "series({expr}, {var})"),
            ],

            # ============ 4. SOLVERS ============
            "solve": [
                ("реши {expr}", "en", "solve {expr}", "solve({expr}, {var})"),
                ("реши уравнение {expr}", "en", "solve equation {expr}", "solve({expr}, {var})"),
                ("найди корни {expr}", "en", "find roots of {expr}", "solve({expr}, {var})"),
                ("решить {expr}", "en", "solve {expr}", "solve({expr}, {var})"),
            ],
            "solve_equation": [
                ("реши уравнение {lhs} = {rhs}", "en", "solve {lhs} = {rhs}", "solve(Eq({lhs}, {rhs}), {var})"),
                ("решить {lhs} = {rhs}", "en", "solve {lhs} = {rhs}", "solve(Eq({lhs}, {rhs}), {var})"),
            ],
            "dsolve": [
                ("реши дифференциальное уравнение {expr}", "en", "solve differential equation {expr}",
                 "dsolve({expr}, {var})"),
            ],

            # ============ 5. TRANSFORMS ============
            "laplace_transform": [
                ("преобразование лапласа {expr}", "en", "laplace transform of {expr}",
                 "laplace_transform({expr}, {var}, s)"),
            ],
            "fourier_transform": [
                ("преобразование фурье {expr}", "en", "fourier transform of {expr}",
                 "fourier_transform({expr}, {var}, k)"),
            ],

            # ============ 6. NUMBER THEORY ============
            "isprime": [
                ("простое ли {n}", "en", "is {n} prime", "isprime({n})"),
                ("{n} простое число", "en", "is {n} prime", "isprime({n})"),
            ],
            "factorint": [
                ("разложи на простые множители {n}", "en", "prime factorization of {n}", "factorint({n})"),
                ("факторизация {n}", "en", "factorize {n}", "factorint({n})"),
            ],
            "fibonacci": [
                ("число фибоначчи {n}", "en", "fibonacci number {n}", "fibonacci({n})"),
                ("{n}-е число фибоначчи", "en", "{n}th fibonacci", "fibonacci({n})"),
            ],
            "factorial": [
                ("факториал {n}", "en", "factorial of {n}", "factorial({n})"),
                ("{n} факториал", "en", "{n} factorial", "factorial({n})"),
            ],
            "binomial": [
                ("биномиальный коэффициент C({n}, {k})", "en", "binomial coefficient C({n}, {k})",
                 "binomial({n}, {k})"),
            ],

            # ============ 7. MATRICES ============
            "Matrix": [
                ("создай матрицу {rows}", "en", "create matrix {rows}", "Matrix({rows})"),
            ],
            "det": [
                ("определитель матрицы {M}", "en", "determinant of {M}", "{M}.det()"),
                ("детерминант {M}", "en", "det of {M}", "{M}.det()"),
            ],
            "inv": [
                ("обратная матрица {M}", "en", "inverse of {M}", "{M}.inv()"),
            ],
            "eigenvals": [
                ("собственные значения {M}", "en", "eigenvalues of {M}", "{M}.eigenvals()"),
            ],

            # ============ 8. TRIGONOMETRY ============
            "sin": [
                ("синус {expr}", "en", "sine of {expr}", "sin({expr})"),
            ],
            "cos": [
                ("косинус {expr}", "en", "cosine of {expr}", "cos({expr})"),
            ],
            "tan": [
                ("тангенс {expr}", "en", "tangent of {expr}", "tan({expr})"),
            ],

            # ============ 9. SPECIAL FUNCTIONS ============
            "exp": [
                ("экспонента {expr}", "en", "exponential of {expr}", "exp({expr})"),
                ("e в степени {expr}", "en", "e to the power {expr}", "exp({expr})"),
            ],
            "log": [
                ("логарифм {expr}", "en", "logarithm of {expr}", "log({expr})"),
                ("натуральный логарифм {expr}", "en", "natural log of {expr}", "log({expr})"),
            ],
            "sqrt": [
                ("корень из {expr}", "en", "square root of {expr}", "sqrt({expr})"),
                ("квадратный корень {expr}", "en", "sqrt of {expr}", "sqrt({expr})"),
            ],

            # ============ 10. LOGIC & SETS ============
            "FiniteSet": [
                ("множество {elements}", "en", "set {elements}", "FiniteSet({elements})"),
            ],
            "Interval": [
                ("интервал от {a} до {b}", "en", "interval from {a} to {b}", "Interval({a}, {b})"),
            ],

            # ============ 11. UTILITIES ============
            "evalf": [
                ("вычисли численно {expr}", "en", "evaluate numerically {expr}", "({expr}).evalf()"),
                ("численное значение {expr}", "en", "numerical value of {expr}", "({expr}).evalf()"),
            ],
            "subs": [
                ("подставь {var} = {value} в {expr}", "en", "substitute {var} = {value} in {expr}",
                 "({expr}).subs({var}, {value})"),
            ],
        }

        # Математические выражения для генерации
        self.expressions = [
            "{v}**2 + 2*{v} + 1",
            "{v}**2 - 4",
            "{v}**3 - 1",
            "sin({v})",
            "cos({v}**2)",
            "exp({v})",
            "log({v})",
            "sqrt({v})",
            "{v}**2 + {v}",
            "({v} + 1)*({v} - 1)",
            "sin({v})/cos({v})",
            "{v}**2/{v}",
        ]

    def generate_expression(self, var='x'):
        """Генерирует случайное математическое выражение"""
        expr_template = random.choice(self.expressions)
        return expr_template.format(v=var)

    def generate_sample(self, func_name, template):
        """Генерирует один пример для функции"""
        ru_template, _, en_template, sympy_template = template
        var = random.choice(self.vars)

        # Подставляем реальные значения
        expr = self.generate_expression(var)
        num1 = random.choice(self.nums)
        num2 = random.choice(self.nums)

        ru_text = ru_template.format(
            expr=expr, var=var, n=num1, k=num2,
            a=num1, b=num2, point=num1,
            lhs=f"{num1}*{var} + 1", rhs="10",
            expr1=f"{var}**2", expr2=f"{var} + 1",
            M="M", rows="[[1, 2], [3, 4]]",
            elements="1, 2, 3", value=num1
        )

        en_text = en_template.format(
            expr=expr, var=var, n=num1, k=num2,
            a=num1, b=num2, point=num1,
            lhs=f"{num1}*{var} + 1", rhs="10",
            expr1=f"{var}**2", expr2=f"{var} + 1",
            M="M", rows="[[1, 2], [3, 4]]",
            elements="1, 2, 3", value=num1
        )

        sympy_code = sympy_template.format(
            expr=expr, var=var, n=num1, k=num2,
            a=num1, b=num2, point=num1,
            lhs=f"{num1}*{var} + 1", rhs="10",
            expr1=f"{var}**2", expr2=f"{var} + 1",
            M="M", rows="[[1, 2], [3, 4]]",
            elements="1, 2, 3", value=num1
        )

        return {
            "input_ru": ru_text,
            "input_en": en_text,
            "output": sympy_code
        }

    def generate_dataset(self, num_samples=10000):
        """Генерирует полный датасет"""
        dataset = []

        for _ in range(num_samples):
            func_name = random.choice(list(self.templates.keys()))
            template = random.choice(self.templates[func_name])
            sample = self.generate_sample(func_name, template)

            # Добавляем русский и английский варианты
            dataset.append({"input": sample["input_ru"], "output": sample["output"]})
            dataset.append({"input": sample["input_en"], "output": sample["output"]})

        return dataset


def generate_and_save(output_path, num_samples=100000):
    """Генерирует и сохраняет датасет"""
    generator = SymPyDatasetGenerator()
    dataset = generator.generate_dataset(num_samples)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Датасет сохранён: {len(dataset)} примеров → {output_path}")


if __name__ == "__main__":
    from pathlib import Path

    output = Path(__file__).parent.parent / "data" / "training_data.json"
    generate_and_save(output, 50000)