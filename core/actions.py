actions = ['simplify', 'expand', 'expand_trig', 'expand_log', 'expand_power_exp', 'factor', 'factorint', 'collect',
           'cancel', 'together',
           'apart', 'radsimp', 'powsimp', 'logcombine', 'nsimplify', 'sqrtdenest', 'residue', 'ratsimp',
           'cancel_common_factors',
           'factor_terms', 'simplify_rational', 'simplify_logic', 'cse', 'separatevars', 'logcombine', 'expand_complex',
           'simplify_fraction', 'denest', 'together_cancel',

           'Poly', 'polys', 'degree', 'LC', 'LM', 'LT', 'coeffs', 'monoms', 'div', 'quo', 'rem', 'gcd', 'lcm',
           'content', 'primitive', 'sqf_list',
           'factor_list', 'resultant', 'discriminant', 'interpolate',
           'hermite_interpolate', 'lagrange_interpolate', 'symmetrize',
           'symmetric_poly', 'groebner', 'sturm', 'sturm_sequence', 'real_roots', 'complex_roots', 'nroots', 'cancel',
           'apart', 'together',
           'ratsimp', 'radsimp',

           'diff', 'Derivative', 'total_derivative', 'partial_derivative', 'gradient', 'divergence', 'curl',
           'laplacian', 'hessian', 'jacobian',

           'integrate', 'Integral', 'manualintegrate', 'meijerg', 'integrate_by_parts', 'integrate_definite',
           'heurisch', 'trigsimp', 'rational_simplify', 'repeated_integrate',

           'limit', 'limit_seq', 'Limit', 'residue', 'singularities', 'finite_diff_weights',

           'series', 'Series', 'series_expansion', 'series_inverse', 'limit_seq', 'asymptotic_series', 'laurent_series',
           'fourier_series', 'power_series', 'removeO', 'O', 'series_coeff',

           'solve', 'solveset', 'linsolve', 'nonlinsolve', 'rsolve', 'dsolve', 'pdsolve', 'nsolve',
           'solve_univariate_inequality',
           'reduce_inequalities', 'diophantine', 'posify', 'solve_linear_system', 'solve_linear_system_LU',
           'checkodesol',
           'checkpdesol', 'ode_order',

           'laplace_transform', 'inverse_laplace_transform', 'fourier_transform', 'inverse_fourier_transform',
           'cosine_transform',
           'inverse_cosine_transform', 'sine_transform', 'inverse_sine_transform', 'mellin_transform',
           'inverse_mellin_transform',
           'z_transform', 'inverse_z_transform', 'hilbert_transform', 'hankel_transform',

           'isprime', 'nextprime', 'prevprime', 'primerange', 'primepi', 'factorint', 'totient', 'mobius',
           'jacobi_symbol',
           'legendre_symbol', 'kronecker_symbol', 'divisors', 'divisor_count', 'divisor_sigma', 'gcd', 'lcm', 'igcdex',
           'gcdex',
           'is_square', 'is_squarefree', 'perfect_power', 'multiplicative_order', 'nthroot_mod', 'n_order',
           'primitive_root',
           'is_primitive_root', 'binomial', 'harmonic', 'bell', 'stirling', 'catalan', 'partition', 'fibonacci',
           'lucas', 'tribonacci',
           'pell', 'bernoulli', 'euler', 'sophiegermain',

           'factorial', 'binomial', 'multinomial', 'stirling', 'bell', 'fibonacci', 'tribonacci', 'catalan',
           'partition', 'permute',
           'combinations', 'permutations', 'subset', 'powerset', 'cartes', 'combinations_with_replacement',
           'Permutation',
           'CombinatorialClass', 'PermutationGroup', 'cycle_index', 'polya', 'burnside', 'graycode', 'pigeonhole',
           'binomial_simplify',

           'Matrix', 'eye', 'zeros', 'ones', 'diag', 'BlockMatrix', 'sparse_matrix', 'banded_matrix', 'M.inv', 'M.det',
           'M.rank', 'M.trace',
           'M''.LUdecomposition', 'M.QRdecomposition', 'M.singular_value_decomposition', 'M.eigenvals', 'M.eigenvects',
           'M.charpoly',
           'M.transpose', 'M.T', 'M.adjugate', 'M.permuteBkwd', 'M.nullspace', 'M.columnspace', 'M.rowspace',
           'kronecker_product',
           'hadamard_product', 'DirectSum', 'solve_linear_system', 'MatrixExpr', 'Identity', 'ZeroMatrix',

           'gradient', 'divergence', 'curl', 'jacobian', 'laplacian', 'hessian', 'tensorcontraction', 'tensorproduct',
           'permutedims',
           'tensor_simplify', 'tensordiagonal', 'tensor_indices', 'tensorhead', 'tensor_rank', 'metric_to_Christoffel',
           'RicciScalar',
           'WeylTensor', 'RiemannTensor', 'EinsteinTensor', 'LeviCivita', 'CovariantDerivative',

           'Point', 'Line', 'Ray', 'Segment', 'Triangle', 'Polygon',
           'Circle', 'Ellipse', 'Parabola', 'Plane', 'Sphere',
           'Cylinder', 'Cone', 'Line3D', 'Point3D',
           'intersection', 'distance', 'area', 'perimeter', 'circumcircle', 'incircle',
           'encloses_point', 'contains', 'is_collinear', 'is_concyclic',
           'angle_between', 'bisector', 'tangent_lines', 'projection',

           'ReferenceFrame', 'Point', 'Body', 'RigidBody',
           'Particle', 'LagrangesMethod', 'KanesMethod',
           'Hamiltonian', 'inertia', 'angular_velocity',
           'center_of_mass', 'potential_energy', 'kinetic_energy',
           'linear_momentum', 'angular_momentum', 'dynamicsymbols',
           'gravity', 'forces', 'torque', ''

                                          'convert_to', 'Quantity', 'UnitSystem',
           'joule', 'newton', 'ampere', 'volt', 'second', 'meter', 'kilogram',

           'Operator', 'Commutator', 'AntiCommutator',
           'Dagger', 'Ket', 'Bra', 'InnerProduct',
           'TensorProduct', 'Pauli', 'SpinOp', 'HamiltonianOperator',

           'GaussBeam', 'RayTransferMatrix', 'ThinLens',
           'Mirror', 'Refraction', 'HuygensWavefront',

           'TransferFunction', 'Feedback', 'Gain', 'BodePlot',

           'Normal', 'Uniform', 'Exponential', 'Bernoulli', 'Binomial', 'Poisson',
           'Cauchy', 'Beta', 'Gamma', 'ChiSquared', 'StudentT', 'FDistribution',
           'Weibull', 'Laplace', 'Geometric', 'NegativeBinomial', 'Hypergeometric',
           'LogNormal', 'Erlang', 'Pareto', 'Rayleigh', 'Triangular',
           'PDF', 'CDF', 'E', 'variance', 'std', 'covariance',
           'correlation', 'skewness', 'kurtosis', 'sample', 'P', 'where',

           'And', 'Or', 'Not', 'Implies', 'Equivalent',
           'simplify_logic', 'to_cnf', 'to_dnf',
           'satisfiable', 'is_tautology', 'simplify_rational_inequality',
           'FiniteSet', 'Interval', 'Union', 'Intersection', 'Complement',
           'ImageSet', 'ConditionSet', 'EmptySet', 'UniversalSet',
           'Subset', 'SymmetricDifference',

           'ccode', 'fcode', 'pycode', 'jscode', 'llvmcode',
           'mathematica_code', 'rustcode', 'latex', 'pretty', 'mathml',
           'repr', 'srepr', 'codegen', 'print_ccode', 'print_latex',
           'python', 'subs', 'lambdify', 'sympify', 'parse_expr',

           'plot', 'plot3d', 'plot_parametric', 'plot_implicit',
           'plot_vector_field', 'plot_contour',
           'plot3d_parametric_line', 'plot3d_parametric_surface',
           'contour_plot', 'polar_plot', 'interactive_plot',

           'N', 'evalf', 'subs', 'replace', 'xreplace',
           'simplify', 'diff', 'integrate', 'limit',
           'expand_func', 'expand_trig', 'rewrite', 'refine',
           'assumptions', 'Q.real', 'Q.integer', 'ask',
           'with assuming', 'piecewise_fold', 'Min', 'Max',
           'clamp', 'rootof', 'RootOf', 'RootSum',
           'Symbol', 'Function', 'Lambda', 'Wild', 'Dummy'
           ]
