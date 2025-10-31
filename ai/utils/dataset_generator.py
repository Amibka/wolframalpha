#!/usr/bin/env python3
"""
sympy_full_dataset_generator.py

Генератор датасета NL -> SymPy-snippet в формате JSON:
[
  {"input": "<натуральный язык (ru/en)>", "output": "<sympy code snippet string>"},
  ...
]

- Поменяй EXAMPLES_PER_FUNCTION = 100 чтобы сгенерировать 100 примеров на каждую функцию.
- DEMO_MODE True генерирует небольшой примерный файл.
- Скрипт не выполняет SymPy-код — только генерирует строки.
"""

import json
import random
import math
from typing import Tuple

random.seed(0)

# -------------------- Настройки --------------------
EXAMPLES_PER_FUNCTION = 100   # <-- поставь 100 для полного набора
OUTPUT_FILE = "sympy_full_dataset.json"
DEMO_MODE = True            # True -> генерирует меньше (useful for quick preview)
SAMPLE_OUTPUT = "sympy_full_dataset_sample.json"

# -------------------- Базовые наборы --------------------
VARS = ['x','y','z','t','a','b','c','u','v','w','p','q']
CONSTS = ['pi', 'E']
UNARY_FUNCS = ['sin','cos','tan','asin','acos','atan','sinh','cosh','exp','log','sqrt','abs']
BINARY_OPS = [' + ', ' - ', ' * ', ' / ', ' ** ']

# -------------------- Утилиты создания выражений --------------------
def choose_var():
    return random.choice(VARS)

def choose_const_or_num():
    r = random.random()
    if r < 0.12:
        return random.choice(CONSTS)
    if r < 0.28:
        # rational
        return f"{random.randint(-9, 9)}/{random.randint(1,9)}"
    return str(random.randint(-10, 50))

def wrap_symbol_decl(vars_list):
    # returns e.g. "x, y = symbols('x y')"
    if not vars_list:
        return ""
    unique = []
    for v in vars_list:
        if v not in unique:
            unique.append(v)
    return f"{', '.join(unique)} = symbols('{ ' '.join(unique) }')"

def rand_unary_expr(depth=0):
    if depth > 2 or random.random() < 0.3:
        choice = random.random()
        if choice < 0.5:
            return choose_var()
        elif choice < 0.8:
            return choose_const_or_num()
        else:
            return f"{random.choice(UNARY_FUNCS)}({choose_var()})"
    left = rand_unary_expr(depth+1)
    op = random.choice(BINARY_OPS)
    right = rand_unary_expr(depth+1)
    # sometimes power with small exponent
    if op.strip() == '**' and random.random() < 0.6:
        right = str(random.randint(0,5))
    s = f"({left}{op}{right})"
    return s

def random_polynomial(var=None, degree=3):
    var = var or choose_var()
    terms = []
    for p in range(degree+1):
        coeff = random.randint(-5,6)
        if coeff == 0:
            continue
        if p == 0:
            terms.append(str(coeff))
        elif p == 1:
            terms.append(f"{coeff}*{var}")
        else:
            terms.append(f"{coeff}*{var}**{p}")
    if not terms:
        terms = ["0"]
    return " + ".join(terms)

def random_matrix(rows=2, cols=2, symbolic_prob=0.3):
    rows_list = []
    for r in range(rows):
        row = []
        for c in range(cols):
            if random.random() < symbolic_prob:
                row.append(choose_var())
            else:
                row.append(str(random.randint(-5, 10)))
        rows_list.append("[" + ", ".join(row) + "]")
    return "[" + ", ".join(rows_list) + "]"

def mk_input_ru(template: str) -> str:
    # small chance to make english instead
    if random.random() < 0.15:
        return "EN: " + template
    return "RU: " + template

# -------------------- Генераторы для функций --------------------
# Каждый такой генератор возвращает (input_text, output_code)
def gen_simplify(i):
    expr = rand_unary_expr()
    code = f"{wrap_symbol_decl([v for v in VARS if v in expr])}; simplify({expr})"
    return (mk_input_ru(f"Упростить выражение {expr}"), code)

def gen_expand(i):
    expr = f"({rand_unary_expr()})"
    code = f"{wrap_symbol_decl([v for v in VARS if v in expr])}; expand({expr}**{random.randint(2,4)})"
    return (mk_input_ru(f"Развернуть {expr} в степень"), code)

def gen_expand_trig(i):
    expr = f"sin({choose_var()} + {choose_var()})"
    code = f"{wrap_symbol_decl([choose_var(), choose_var()])}; expand_trig({expr})"
    return (mk_input_ru(f"Разложить тригонометрическое выражение {expr}"), code)

def gen_expand_log(i):
    expr = f"log({choose_var()}**2)"
    code = f"{wrap_symbol_decl([choose_var()])}; expand_log({expr})"
    return (mk_input_ru(f"Применить expand_log к {expr}"), code)

def gen_expand_power_exp(i):
    expr = f"exp({choose_var()}*log({choose_var()}))"
    code = f"{wrap_symbol_decl([choose_var(), choose_var()])}; expand_power_exp({expr})"
    return (mk_input_ru(f"expand_power_exp для {expr}"), code)

def gen_factor(i):
    poly = random_polynomial(choose_var(), degree=random.randint(1,4))
    v = poly.split('*')[-1] if '*' in poly else choose_var()
    code = f"{wrap_symbol_decl([v])}; factor({poly})"
    return (mk_input_ru(f"Разложить на множители {poly}"), code)

def gen_factorint(i):
    n = random.randint(10, 2000)
    return (mk_input_ru(f"Факторизовать целое {n}"), f"factorint({n})")

def gen_collect(i):
    v = choose_var()
    expr = f"{random.randint(1,5)}*{v} + {random.randint(1,5)}*{v}**2 + {random.randint(1,5)}*{v}**2"
    return (mk_input_ru(f"Собрать подобные в {expr}"), f"{wrap_symbol_decl([v])}; collect({expr}, {v})")

def gen_cancel(i):
    v = choose_var()
    expr = f"({v}**2 - 1)/({v} - 1)"
    return (mk_input_ru(f"Сократить {expr}"), f"{wrap_symbol_decl([v])}; cancel({expr})")

def gen_together(i):
    v = choose_var()
    expr = f"1/{v} + 1/({v}+1)"
    return (mk_input_ru(f"Объединить дроби {expr}"), f"{wrap_symbol_decl([v])}; together({expr})")

def gen_apart(i):
    v = choose_var()
    expr = f"1/({v}*({v}+1))"
    return (mk_input_ru(f"Разложить в простые дроби {expr}"), f"{wrap_symbol_decl([v])}; apart({expr})")

def gen_radsimp(i):
    expr = f"sqrt({choose_var()}+1) + sqrt({choose_var()}+1)"
    vlist = list({choose_var()})
    return (mk_input_ru(f"Radsimp для {expr}"), f"{wrap_symbol_decl([choose_var()])}; radsimp({expr})")

def gen_powsimp(i):
    expr = f"({choose_var()}**2)*({choose_var()}**3)"
    return (mk_input_ru(f"Powsimp {expr}"), f"{wrap_symbol_decl([choose_var()])}; powsimp({expr})")

def gen_logcombine(i):
    expr = f"log({choose_var()}) + log({choose_var()})"
    return (mk_input_ru(f"Logcombine {expr}"), f"{wrap_symbol_decl([choose_var()])}; logcombine({expr})")

def gen_nsimplify(i):
    val = f"{random.randint(1,50)}/{random.randint(1,10)}"
    return (mk_input_ru(f"Преобразовать {val} в рациональное приближение"), f"nsimplify({val})")

def gen_sqrtdenest(i):
    expr = "sqrt(1 + 2*sqrt(3))"
    return (mk_input_ru(f"sqrtdenest {expr}"), f"sqrtdenest({expr})")

def gen_residue(i):
    x = choose_var()
    expr = f"1/({x}*(1 + {x}))"
    return (mk_input_ru(f"Найти residue {expr} wrt {x} at 0"), f"{wrap_symbol_decl([x])}; residue({expr}, {x}, 0)")

def gen_ratsimp(i):
    expr = f"({random.randint(1,5)}*{choose_var()} + {random.randint(1,5)})/({choose_var()})"
    return (mk_input_ru(f"Rational simplify {expr}"), f"{wrap_symbol_decl([choose_var()])}; ratsimp({expr})")

def gen_cse(i):
    expr1 = random_polynomial(choose_var(), degree=3)
    expr2 = random_polynomial(choose_var(), degree=3)
    return (mk_input_ru(f"CSE for {expr1} and {expr2}"), f"{wrap_symbol_decl([])}; cse([{expr1}, {expr2}])")

def gen_separatevars(i):
    expr = f"{choose_var()}*{choose_var()} + {choose_var()}"
    return (mk_input_ru(f"Separatevars {expr}"), f"{wrap_symbol_decl([choose_var()])}; separatevars({expr})")

def gen_expand_complex(i):
    expr = f"(sin({choose_var()}) + I*cos({choose_var()}))"
    return (mk_input_ru(f"Expand complex {expr}"), f"{wrap_symbol_decl([choose_var()])}; expand_complex({expr})")

def gen_denest(i):
    expr = "sqrt(2 + 2*sqrt(2))"
    return (mk_input_ru(f"Denest {expr}"), f"denest({expr})")

def gen_together_cancel(i):
    v = choose_var()
    expr = f"( {v}**2 - 1 )/({v} - 1)"
    return (mk_input_ru(f"Together/cancel {expr}"), f"{wrap_symbol_decl([v])}; together({expr})")

# -------------------- Polynomials & rational --------------------
def gen_poly_degree(i):
    v = choose_var()
    poly = random_polynomial(v, degree=random.randint(2,4))
    return (mk_input_ru(f"Степень многочлена {poly}"), f"{wrap_symbol_decl([v])}; Poly({poly}, {v}).degree()")

def gen_LC_LM_LT_coeffs(i):
    v = choose_var()
    poly = random_polynomial(v, degree=random.randint(2,4))
    return (mk_input_ru(f"LC/coeffs для {poly}"), f"{wrap_symbol_decl([v])}; Poly({poly}, {v}).LC(), Poly({poly}, {v}).coeffs()")

def gen_div_quo_rem(i):
    x = choose_var(); y = choose_var()
    f = random_polynomial(x, degree=3)
    g = random_polynomial(x, degree=2)
    return (mk_input_ru(f"Div/quo/rem {f} by {g}"), f"{wrap_symbol_decl([x])}; div({f}, {g}, {x})")

def gen_gcd_lcm(i):
    a = random.randint(12,180); b = random.randint(2,120)
    return (mk_input_ru(f"gcd/lcm of {a} and {b}"), f"gcd({a}, {b}); lcm({a}, {b})")

def gen_resultant_discriminant(i):
    x = choose_var()
    f = random_polynomial(x, degree=3)
    g = random_polynomial(x, degree=2)
    return (mk_input_ru(f"resultant/discriminant for {f} and {g}"), f"{wrap_symbol_decl([x])}; resultant({f}, {g}, {x}); discriminant({f}, {x})")

def gen_groebner(i):
    x,y = random.sample(VARS,2)
    f1 = f"{x}**2 + {y} - 1"
    f2 = f"{x} + {y}**2 - 1"
    return (mk_input_ru(f"Groebner basis for {f1} and {f2}"), f"{wrap_symbol_decl([x,y])}; groebner([{f1}, {f2}], {x}, {y})")

# -------------------- Calculus & Analysis --------------------
def gen_diff(i):
    v = choose_var()
    expr = rand_unary_expr()
    return (mk_input_ru(f"Найти производную {expr} по {v}"), f"{wrap_symbol_decl([v])}; diff({expr}, {v})")

def gen_derivative_obj(i):
    v = choose_var()
    expr = rand_unary_expr()
    return (mk_input_ru(f"Derivative({expr},{v}) object"), f"{wrap_symbol_decl([v])}; Derivative({expr}, {v})")

def gen_total_derivative(i):
    expr = f"f({choose_var()},{choose_var()})"
    return (mk_input_ru(f"Total derivative of {expr}"), f"t = symbols('t'); total_derivative({expr}, t)")

def gen_gradient(i):
    x,y = random.sample(VARS,2)
    expr = f"{x}**2 + {y}**2"
    return (mk_input_ru(f"Gradient of {expr}"), f"{wrap_symbol_decl([x,y])}; gradient({expr}, ({x},{y}))")

def gen_divergence_curl(i):
    x,y,z = random.sample(VARS,3)
    vec = f"Matrix([{x}**2, sin({y}), {z}])"
    return (mk_input_ru(f"Divergence/curl of {vec}"), f"{wrap_symbol_decl([x,y,z])}; divergence({vec}, ({x},{y},{z})); curl({vec}, ({x},{y},{z}))")

def gen_laplacian_hessian_jacobian(i):
    x,y = random.sample(VARS,2)
    expr = f"{x}**3 + y**2"
    return (mk_input_ru(f"Laplacian/hessian/jacobian of {expr}"), f"{wrap_symbol_decl([x,y])}; laplacian({expr}, ({x,y})); hessian({expr}, ({x,y})); jacobian(Matrix([{expr}]), ({x,y}))")

def gen_integrate(i):
    v = choose_var()
    expr = rand_unary_expr()
    if random.random() < 0.4:
        a = random.randint(-3,0); b = random.randint(1,5)
        return (mk_input_ru(f"Определённый интеграл {expr} от {a} до {b}"), f"{wrap_symbol_decl([v])}; integrate({expr}, ({v}, {a}, {b}))")
    return (mk_input_ru(f"Неопределённый интеграл {expr}"), f"{wrap_symbol_decl([v])}; integrate({expr}, {v})")

def gen_integral_object(i):
    v = choose_var()
    expr = rand_unary_expr()
    return (mk_input_ru(f"Создать объект Integral для {expr}"), f"{wrap_symbol_decl([v])}; Integral({expr}, {v})")

def gen_meijerg(i):
    # a placeholder-style example
    x = choose_var()
    return (mk_input_ru("Пример meijerg"), f"{wrap_symbol_decl([x])}; meijerg([], [], [], [], {x})")

def gen_limit(i):
    v = choose_var()
    expr = f"sin({v})/{v}"
    return (mk_input_ru(f"Предел {expr} при {v}->0"), f"{wrap_symbol_decl([v])}; limit({expr}, {v}, 0)")

def gen_series(i):
    v = choose_var()
    expr = f"exp({v})"
    return (mk_input_ru(f"Ряд Тейлора {expr} в 0 до 5"), f"{wrap_symbol_decl([v])}; series({expr}, {v}, 0, 5)")

# -------------------- Solvers (ODE/PDE etc) --------------------
def gen_solve(i):
    v = choose_var()
    A = random.randint(1,9)
    B = random.randint(-5,15)
    return (mk_input_ru(f"Реши линейное уравнение {A}{v} + 1 = {B}"), f"{wrap_symbol_decl([v])}; solve(({A}*{v} + 1) - ({B}), {v})")

def gen_solveset(i):
    v = choose_var()
    expr = f"{v}**2 - {random.randint(1,10)}"
    return (mk_input_ru(f"Solveset для {expr}"), f"{wrap_symbol_decl([v])}; solveset({expr}, {v})")

def gen_linsolve(i):
    x,y = random.sample(VARS,2)
    return (mk_input_ru(f"linsolve простой системы {x}+{y}=2, {x}-{y}=0"), f"{wrap_symbol_decl([x,y])}; linsolve([{x}+{y}-2, {x}-{y}], [{x},{y}])")

def gen_nonlinsolve(i):
    x,y = random.sample(VARS,2)
    return (mk_input_ru(f"nonlinsolve системы {x}**2 + {y} - 1, {x} + {y}**2 -1"), f"{wrap_symbol_decl([x,y])}; nonlinsolve([{x}**2 + {y} - 1, {x} + {y}**2 - 1], [{x},{y}])")

def gen_nsolve(i):
    v = choose_var()
    poly = random_polynomial(v, degree=3)
    return (mk_input_ru(f"nsolve numeric solve {poly} = 0"), f"{wrap_symbol_decl([v])}; nsolve({poly}, 1)")

def gen_dsolve(i):
    t = choose_var()
    y = "y"
    # simple ODE example: y'(t) - y(t) = 0
    return (mk_input_ru("Решить ОДУ y'(t) - y(t) = 0 с помощью dsolve"), "t = symbols('t'); y = Function('y'); dsolve(Eq(Derivative(y(t), t) - y(t), 0), y(t))")

def gen_solve_univariate_inequality(i):
    v = choose_var()
    expr = f"{v}**2 - {random.randint(1,4)} > 0"
    return (mk_input_ru(f"Solve univariate inequality {expr}"), f"{wrap_symbol_decl([v])}; solve_univariate_inequality({expr}, {v})")

def gen_reduce_inequalities(i):
    v = choose_var()
    expr = f"{v} > 0"
    return (mk_input_ru(f"Reduce inequalities for {expr}"), f"{wrap_symbol_decl([v])}; reduce_inequalities([{expr}], {v})")

# -------------------- Transforms & Fourier --------------------
def gen_laplace_transform(i):
    return (mk_input_ru("Laplace transform of exp(a*t)"), "t,a,s = symbols('t a s'); laplace_transform(exp(a*t), t, s)")

def gen_inverse_laplace_transform(i):
    return (mk_input_ru("Inverse Laplace transform of 1/(s-1)"), "t,s = symbols('t s'); inverse_laplace_transform(1/(s-1), s, t)")

def gen_fourier_transform(i):
    return (mk_input_ru("Fourier transform of sin(t)"), "t,w = symbols('t w'); fourier_transform(sin(t), t, w)")

def gen_cosine_transform(i):
    return (mk_input_ru("Cosine transform of exp(-t)"), "t,w = symbols('t w'); cosine_transform(exp(-t), t, w)")

def gen_mellin_transform(i):
    x = choose_var()
    return (mk_input_ru("Mellin transform example"), f"{wrap_symbol_decl([x])}; mellin_transform({x}**2, {x}, 1)")

# -------------------- Number Theory --------------------
def gen_isprime(i):
    n = random.randint(2,200)
    return (mk_input_ru(f"Проверить простоту числа {n}"), f"isprime({n})")

def gen_nextprime(i):
    n = random.randint(2,200)
    return (mk_input_ru(f"Следующее простое после {n}"), f"nextprime({n})")

def gen_factorint(i):
    n = random.randint(2,500)
    return (mk_input_ru(f"Факторизовать {n}"), f"factorint({n})")

def gen_primepi(i):
    n = random.randint(2,200)
    return (mk_input_ru(f"primepi({n})"), f"primepi({n})")

def gen_totient(i):
    n = random.randint(2,200)
    return (mk_input_ru(f"phi({n})"), f"totient({n})")

def gen_gcd_lcm(i):
    a = random.randint(2,200); b = random.randint(2,200)
    return (mk_input_ru(f"gcd/lcm {a},{b}"), f"gcd({a},{b}); lcm({a},{b})")

def gen_fibonacci(i):
    n = random.randint(1,20)
    return (mk_input_ru(f"Fibonacci({n})"), f"fibonacci({n})")

def gen_bernoulli(i):
    n = random.randint(0,10)
    return (mk_input_ru(f"Bernoulli({n})"), f"bernoulli({n})")

# -------------------- Combinatorics --------------------
def gen_factorial(i):
    n = random.randint(0,10)
    return (mk_input_ru(f"{n}!"), f"factorial({n})")

def gen_binomial(i):
    n = random.randint(1,10); k = random.randint(0,n)
    return (mk_input_ru(f"Binomial({n},{k})"), f"binomial({n}, {k})")

def gen_permutations(i):
    n = random.randint(3,7)
    return (mk_input_ru(f"permutations of {n}"), f"list(permutations(range({n})))")

# -------------------- Matrices & Linear Algebra --------------------
def gen_matrix_det(i):
    mat = random_matrix(2,2)
    return (mk_input_ru(f"Determinant of Matrix{mat}"), f"M = Matrix({mat}); M.det()")

def gen_matrix_inv(i):
    mat = "[[1,2],[3,4]]"
    return (mk_input_ru(f"Inverse of Matrix{mat}"), f"M = Matrix({mat}); M.inv()")

def gen_matrix_eigen(i):
    mat = "[[2,0],[0,3]]"
    return (mk_input_ru(f"Eigenvalues of Matrix{mat}"), f"M = Matrix({mat}); M.eigenvals()")

def gen_matrix_nullspace(i):
    mat = "[[1,2],[2,4]]"
    return (mk_input_ru(f"Nullspace of Matrix{mat}"), f"M = Matrix({mat}); M.nullspace()")

# -------------------- Geometry --------------------
def gen_point_line_circle(i):
    return (mk_input_ru("Circle center (0,0) r=3"), "from sympy import Circle, Point; Circle(Point(0,0), 3)")

def gen_polygon_area(i):
    pts = "Point(0,0), Point(1,0), Point(0,1)"
    return (mk_input_ru("Triangle area example"), f"from sympy import Polygon, Point; Polygon({pts}).area")

# -------------------- Physics --------------------
def gen_rigid_body(i):
    return (mk_input_ru("Пример Dynamics: LagrangesMethod skeleton"), "from sympy.physics.mechanics import ReferenceFrame, Point; N = ReferenceFrame('N')")

def gen_units(i):
    return (mk_input_ru("Конвертация единиц: 1 meter to centimeters"), "from sympy.physics.units import meter; convert_to(1*meter, centimeter)")

# -------------------- Statistics --------------------
def gen_distribution_pdf(i):
    mu = random.randint(-2,2); sigma = random.randint(1,3)
    return (mk_input_ru(f"PDF Normal({mu},{sigma})"), f"X = Normal('X', {mu}, {sigma}); density(X)")

def gen_cdf_example(i):
    mu = 0; sigma = 1
    return (mk_input_ru("CDF Normal(0,1) at 1"), "X = Normal('X', 0, 1); cdf(X)(1)")

# -------------------- Logic & Sets --------------------
def gen_logic_simplify(i):
    return (mk_input_ru("Simplify logical (A & B) | (A & ~B)"), "A,B = symbols('A B'); simplify_logic((A & B) | (A & ~B))")

def gen_sets(i):
    return (mk_input_ru("FiniteSet example"), "FiniteSet(1,2,3)")

# -------------------- Codegen / printing --------------------
def gen_latex(i):
    expr = random_polynomial(choose_var(), degree=2)
    v = choose_var()
    return (mk_input_ru(f"Latex for {expr}"), f"{wrap_symbol_decl([v])}; latex({expr})")

def gen_pycode(i):
    expr = f"{choose_var()}**2 + sin({choose_var()})"
    v1, v2 = choose_var(), choose_var()
    return (mk_input_ru(f"Pycode for {expr}"), f"{wrap_symbol_decl([v1,v2])}; pycode({expr})")

# -------------------- Plotting (examples only) --------------------
def gen_plot(i):
    expr = f"sin({choose_var()})"
    return (mk_input_ru(f"Plot {expr}"), f"plot({expr})")

# -------------------- Utilities / Misc --------------------
def gen_evalf(i):
    val = f"pi"
    return (mk_input_ru(f"Evaluate pi numerically"), f"N({val}, 50)")

def gen_subs(i):
    v = choose_var()
    expr = f"{v}**2 + 3"
    return (mk_input_ru(f"Substitute {v}=2 in {expr}"), f"{wrap_symbol_decl([v])}; ({expr}).subs({v}, 2)")

def gen_lambdify(i):
    v = choose_var()
    expr = f"sin({v})"
    return (mk_input_ru(f"Lambdify {expr}"), f"{wrap_symbol_decl([v])}; f = lambdify({v}, {expr}); f(1.0)")

# -------------------- Сбор шаблонов --------------------
TEMPLATES = {
    # Algebra & simplification
    'simplify': gen_simplify, 'expand': gen_expand, 'expand_trig': gen_expand_trig,
    'expand_log': gen_expand_log, 'expand_power_exp': gen_expand_power_exp, 'factor': gen_factor,
    'factorint': gen_factorint, 'collect': gen_collect, 'cancel': gen_cancel, 'together': gen_together,
    'apart': gen_apart, 'radsimp': gen_radsimp, 'powsimp': gen_powsimp, 'logcombine': gen_logcombine,
    'nsimplify': gen_nsimplify, 'sqrtdenest': gen_sqrtdenest, 'residue': gen_residue, 'ratsimp': gen_ratsimp,
    'cse': gen_cse, 'separatevars': gen_separatevars, 'expand_complex': gen_expand_complex, 'denest': gen_denest,
    'together_cancel': gen_together_cancel,

    # Polynomials & rational
    'poly_degree': gen_poly_degree, 'LC_LM': gen_LC_LM_LT_coeffs, 'div_quo_rem': gen_div_quo_rem,
    'gcd_lcm': gen_gcd_lcm, 'resultant_disc': gen_resultant_discriminant, 'groebner': gen_groebner,

    # Calculus & Analysis
    'diff': gen_diff, 'Derivative': gen_derivative_obj, 'total_derivative': gen_total_derivative,
    'gradient': gen_gradient, 'divergence_curl': gen_divergence_curl,
    'laplacian_hessian_jacobian': gen_laplacian_hessian_jacobian, 'integrate': gen_integrate,
    'Integral': gen_integral_object, 'meijerg': gen_meijerg, 'limit': gen_limit, 'series': gen_series,

    # Solvers
    'solve': gen_solve, 'solveset': gen_solveset, 'linsolve': gen_linsolve, 'nonlinsolve': gen_nonlinsolve,
    'nsolve': gen_nsolve, 'dsolve': gen_dsolve, 'solve_univariate_inequality': gen_solve_univariate_inequality,
    'reduce_inequalities': gen_reduce_inequalities,

    # Transforms
    'laplace_transform': gen_laplace_transform, 'inverse_laplace': gen_inverse_laplace_transform,
    'fourier_transform': gen_fourier_transform, 'cosine_transform': gen_cosine_transform,
    'mellin_transform': gen_mellin_transform,

    # Number theory & combinatorics
    'isprime': gen_isprime, 'nextprime': gen_nextprime, 'factorint2': gen_factorint,
    'primepi': gen_primepi, 'totient': gen_totient, 'gcd_lcm2': gen_gcd_lcm, 'fibonacci': gen_fibonacci,
    'bernoulli': gen_bernoulli, 'factorial': gen_factorial, 'binomial': gen_binomial, 'permutations': gen_permutations,

    # Matrices & linear algebra
    'matrix_det': gen_matrix_det, 'matrix_inv': gen_matrix_inv, 'matrix_eigen': gen_matrix_eigen,
    'matrix_nullspace': gen_matrix_nullspace,

    # Geometry
    'circle': gen_point_line_circle, 'polygon_area': gen_polygon_area,

    # Physics
    'rigid_body': gen_rigid_body, 'units': gen_units,

    # Stats
    'dist_pdf': gen_distribution_pdf, 'dist_cdf': gen_cdf_example,

    # Logic & sets
    'logic_simplify': gen_logic_simplify, 'sets': gen_sets,

    # Codegen / printing
    'latex': gen_latex, 'pycode': gen_pycode,

    # Plotting
    'plot': gen_plot,

    # Utils
    'evalf': gen_evalf, 'subs': gen_subs, 'lambdify': gen_lambdify,
}

# -------------------- Генерация --------------------
def generate_dataset(examples_per_func: int):
    dataset = []
    for fname, gen in TEMPLATES.items():
        for i in range(examples_per_func):
            try:
                inp, out = gen(i)
                # ensure output is a single-line string
                out_single = " ".join(out.split())
                dataset.append({"input": inp, "output": out_single})
            except Exception as e:
                # best-effort: если генератор упал — пропускаем
                continue
    return dataset

def main():
    examples = EXAMPLES_PER_FUNCTION if not DEMO_MODE else 3
    print(f"Generating ~{examples * len(TEMPLATES)} samples ({examples} per template)...")
    data = generate_dataset(examples)
    # Save full dataset
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} samples to {OUTPUT_FILE}")
    # also save a small sample preview
    if DEMO_MODE:
        with open(SAMPLE_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(data[:min(200, len(data))], f, ensure_ascii=False, indent=2)
        print(f"Saved preview to {SAMPLE_OUTPUT}")

if __name__ == "__main__":
    main()
