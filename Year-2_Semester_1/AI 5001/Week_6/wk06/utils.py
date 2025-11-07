"""Utility functions and classes.

This module provides a comprehensive collection of utilities used throughout,including:

- Sequence and iterable manipulation functions
- Statistical and mathematical functions
- Neural network activation functions and their derivatives
- Machine learning kernel functions
- Distance metrics (Euclidean, Manhattan, Hamming)
- Grid world navigation utilities
- Symbolic expression system for logical reasoning
- Priority queue implementation
- Memoization and caching utilities
- Data access helpers

These utilities support various AI algorithms including search, logic,
learning, planning, and probabilistic reasoning.
"""

import bisect
import collections
import collections.abc
import functools
import heapq
import operator
import os.path
import random
from itertools import chain, combinations
from statistics import mean

import numpy as np


# ______________________________________________________________________________
# Functions on Sequences and Iterables
#
# Utility functions for manipulating Python sequences, lists, and iterables.
# These provide common operations needed throughout AI algorithms.


def sequence(iterable):
    """Converts iterable to sequence, if it is not already one."""
    return (
        iterable
        if isinstance(iterable, collections.abc.Sequence)
        else tuple([iterable])
    )


def remove_all(item, seq):
    """Return a copy of seq (or string) with all occurrences of item removed."""
    if isinstance(seq, str):
        return seq.replace(item, "")
    elif isinstance(seq, set):
        rest = seq.copy()
        rest.remove(item)
        return rest
    else:
        return [x for x in seq if x != item]


def unique(seq):
    """Remove duplicate elements from seq. Assumes hashable elements."""
    return list(set(seq))


def count(seq):
    """Count the number of items in sequence that are interpreted as true."""
    return sum(map(bool, seq))


def multimap(items):
    """Given (key, val) pairs, return {key: [val, ....], ...}."""
    result = collections.defaultdict(list)
    for key, val in items:
        result[key].append(val)
    return dict(result)


def multimap_items(mmap):
    """Yield all (key, val) pairs stored in the multimap."""
    for key, vals in mmap.items():
        for val in vals:
            yield key, val


def product(numbers):
    """Return the product of the numbers, e.g. product([2, 3, 10]) == 60"""
    result = 1
    for x in numbers:
        result *= x
    return result


def first(iterable, default=None):
    """Return the first element of an iterable; or default."""
    return next(iter(iterable), default)


def is_in(elt, seq):
    """Similar to (elt in seq), but compares with 'is', not '=='."""
    return any(x is elt for x in seq)


def mode(data):
    """Return the most common data item. If there are ties, return any one of them."""
    [(item, count)] = collections.Counter(data).most_common(1)
    return item


def power_set(iterable):
    """power_set([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))[1:]


def extend(s, var, val):
    """Copy dict s and extend it by setting var to val; return copy."""
    return {**s, var: val}


def flatten(seqs):
    """Flatten a sequence of sequences into a single list.

    Concatenates all subsequences in seqs into one flat list.
    Example: flatten([[1, 2], [3, 4, 5]]) returns [1, 2, 3, 4, 5]
    """
    return sum(seqs, [])


# ______________________________________________________________________________
# Optimization Functions: argmin and argmax
#
# Functions for finding minimum and maximum elements with random tie-breaking.
# Essential for AI algorithms where deterministic tie-breaking could introduce bias.

identity = lambda x: x


def argmin_random_tie(seq, key=identity):
    """Return a minimum element of seq; break ties at random."""
    return min(shuffled(seq), key=key)


def argmax_random_tie(seq, key=identity):
    """Return an element with highest fn(seq[i]) score; break ties at random."""
    return max(shuffled(seq), key=key)


def shuffled(iterable):
    """Randomly shuffle a copy of iterable."""
    items = list(iterable)
    random.shuffle(items)
    return items


# ______________________________________________________________________________
# Statistical and Mathematical Functions
#
# Core mathematical operations including vector arithmetic, probability functions,
# neural network activation functions, kernel functions, distance metrics, and
# loss functions used throughout machine learning algorithms.


def histogram(values, mode=0, bin_function=None):
    """Return a list of (value, count) pairs, summarizing the input values.
    Sorted by increasing value, or if mode=1, by decreasing count.
    If bin_function is given, map it over values first."""
    if bin_function:
        values = map(bin_function, values)

    bins = {}
    for val in values:
        bins[val] = bins.get(val, 0) + 1

    if mode:
        return sorted(list(bins.items()), key=lambda x: (x[1], x[0]), reverse=True)
    else:
        return sorted(bins.items())


def dot_product(x, y):
    """Return the sum of the element-wise product of vectors x and y."""
    return sum(_x * _y for _x, _y in zip(x, y))


def element_wise_product(x, y):
    """Return vector as an element-wise product of vectors x and y."""
    assert len(x) == len(y)
    return np.multiply(x, y)


def matrix_multiplication(x, *y):
    """Return a matrix as a matrix-multiplication of x and arbitrary number of matrices *y."""

    result = x
    for _y in y:
        result = np.matmul(result, _y)

    return result


def vector_add(a, b):
    """Component-wise addition of two vectors."""
    return tuple(map(operator.add, a, b))


def scalar_vector_product(x, y):
    """Return vector as a product of a scalar and a vector"""
    return np.multiply(x, y)


def probability(p):
    """Return true with probability p."""
    return p > random.uniform(0.0, 1.0)


def weighted_sample_with_replacement(n, seq, weights):
    """Pick n samples from seq at random, with replacement, with the
    probability of each element in proportion to its corresponding
    weight.

    Args:
        n: Number of samples to draw
        seq: Sequence to sample from
        weights: List of weights corresponding to each element in seq

    Returns:
        List of n sampled elements (with possible duplicates)
    """
    sample = weighted_sampler(seq, weights)
    return [sample() for _ in range(n)]


def weighted_sampler(seq, weights):
    """Return a random-sample function that picks from seq weighted by weights.

    Creates a cumulative distribution and uses binary search for efficient
    weighted sampling. The returned function can be called repeatedly.

    Args:
        seq: Sequence to sample from
        weights: List of weights for each element in seq

    Returns:
        Function that returns a random element from seq according to weights
    """
    totals = []
    for w in weights:
        totals.append(w + totals[-1] if totals else w)
    return lambda: seq[bisect.bisect(totals, random.uniform(0, totals[-1]))]


def weighted_choice(choices):
    """A weighted version of random.choice"""
    # NOTE: should be replaced by random.choices if we port to Python 3.6

    total = sum(w for _, w in choices)
    r = random.uniform(0, total)
    upto = 0
    for c, w in choices:
        if upto + w >= r:
            return c, w
        upto += w


def rounder(numbers, d=4):
    """Round a single number, or sequence of numbers, to d decimal places."""
    if isinstance(numbers, (int, float)):
        return round(numbers, d)
    else:
        constructor = type(numbers)  # Can be list, set, tuple, etc.
        return constructor(rounder(n, d) for n in numbers)


def num_or_str(x):  # TODO: rename as `atom`
    """The argument is a string; convert to a number if possible, or strip it."""
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return str(x).strip()


def euclidean_distance(x, y):
    """Calculate the Euclidean distance between two points.

    Returns the straight-line distance between vectors x and y.
    Formula: sqrt(sum((xi - yi)²))
    """
    return np.sqrt(sum((_x - _y) ** 2 for _x, _y in zip(x, y)))


def manhattan_distance(x, y):
    """Calculate the Manhattan (L1) distance between two points.

    Returns the sum of absolute differences between coordinates.
    Also known as taxicab or city block distance.
    """
    return sum(abs(_x - _y) for _x, _y in zip(x, y))


def hamming_distance(x, y):
    """Calculate the Hamming distance between two sequences.

    Returns the number of positions where the sequences differ.
    Commonly used for comparing strings or binary sequences.
    """
    return sum(_x != _y for _x, _y in zip(x, y))


def cross_entropy_loss(x, y):
    """Calculate cross-entropy loss between true labels x and predictions y.

    Used for binary classification problems. Assumes y contains probabilities
    and x contains true binary labels (0 or 1).

    Args:
        x: True binary labels
        y: Predicted probabilities

    Returns:
        Cross-entropy loss value
    """
    return (-1.0 / len(x)) * sum(
        _x * np.log(_y) + (1 - _x) * np.log(1 - _y) for _x, _y in zip(x, y)
    )


def mean_squared_error_loss(x, y):
    """Calculate mean squared error loss between true values x and predictions y.

    Commonly used for regression problems.

    Args:
        x: True values
        y: Predicted values

    Returns:
        Mean squared error loss value
    """
    return (1.0 / len(x)) * sum((_x - _y) ** 2 for _x, _y in zip(x, y))


def rms_error(x, y):
    """Calculate root mean squared error between true values x and predictions y.

    Square root of the mean squared error, giving error in original units.
    """
    return np.sqrt(ms_error(x, y))


def ms_error(x, y):
    """Calculate mean squared error between true values x and predictions y.

    Same as mean_squared_error_loss but uses statistics.mean function.
    """
    return mean((_x - _y) ** 2 for _x, _y in zip(x, y))


def mean_error(x, y):
    """Calculate mean absolute error between true values x and predictions y.

    Also known as L1 loss or Manhattan distance in error space.
    """
    return mean(abs(_x - _y) for _x, _y in zip(x, y))


def mean_boolean_error(x, y):
    """Calculate mean boolean error rate between true values x and predictions y.

    Returns the fraction of positions where x and y differ.
    Useful for classification accuracy evaluation.
    """
    return mean(_x != _y for _x, _y in zip(x, y))


def normalize(dist):
    """Multiply each number by a constant such that the sum is 1.0.

    Normalizes a probability distribution so all values sum to 1.
    Works with both dictionaries and lists/tuples.

    Args:
        dist: Dictionary or sequence of numbers to normalize

    Returns:
        Normalized distribution (same type as input)

    Note:
        For dictionaries, modifies the input in-place and returns it.
        For sequences, returns a new list.
    """
    if isinstance(dist, dict):
        total = sum(dist.values())
        for key in dist:
            dist[key] = dist[key] / total
            assert 0 <= dist[key] <= 1  # probabilities must be between 0 and 1
        return dist
    total = sum(dist)
    return [(n / total) for n in dist]


def random_weights(min_value, max_value, num_weights):
    """Generate a list of random weights within the specified range.

    Args:
        min_value: Minimum value for weights
        max_value: Maximum value for weights
        num_weights: Number of weights to generate

    Returns:
        List of random floating-point weights
    """
    return [random.uniform(min_value, max_value) for _ in range(num_weights)]


def sigmoid(x):
    """Return activation value of x with sigmoid function."""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(value):
    """Return the derivative of the sigmoid function at the given value.

    Used in backpropagation for neural networks. Assumes value is the
    output of the sigmoid function, not the original input.
    """
    return value * (1 - value)


def elu(x, alpha=0.01):
    """Exponential Linear Unit (ELU) activation function.

    Returns x if x > 0, otherwise alpha * (exp(x) - 1).
    Provides smooth gradient for negative values.
    """
    return x if x > 0 else alpha * (np.exp(x) - 1)


def elu_derivative(value, alpha=0.01):
    """Return the derivative of the ELU activation function.

    Used in backpropagation. Assumes value is the original input,
    not the ELU output.
    """
    return 1 if value > 0 else alpha * np.exp(value)


def tanh(x):
    """Hyperbolic tangent activation function.

    Returns tanh(x), mapping inputs to range (-1, 1).
    """
    return np.tanh(x)


def tanh_derivative(value):
    """Return the derivative of the tanh activation function.

    Assumes value is the output of tanh, not the original input.
    Formula: 1 - tanh²(x)
    """
    return 1 - (value**2)


def leaky_relu(x, alpha=0.01):
    """Leaky Rectified Linear Unit activation function.

    Returns x if x > 0, otherwise alpha * x.
    Prevents dying ReLU problem by allowing small negative gradients.
    """
    return x if x > 0 else alpha * x


def leaky_relu_derivative(value, alpha=0.01):
    """Return the derivative of the Leaky ReLU activation function.

    Returns 1 for positive values, alpha for negative values.
    """
    return 1 if value > 0 else alpha


def relu(x):
    """Rectified Linear Unit (ReLU) activation function.

    Returns max(0, x). Most commonly used activation function
    in deep learning due to computational efficiency.
    """
    return max(0, x)


def relu_derivative(value):
    """Return the derivative of the ReLU activation function.

    Returns 1 for positive values, 0 for negative values.
    """
    return 1 if value > 0 else 0


def step(x):
    """Return activation value of x with sign function"""
    return 1 if x >= 0 else 0


def gaussian(mean, st_dev, x):
    """Given the mean and standard deviation of a distribution, it returns the probability of x."""
    return (
        1
        / (np.sqrt(2 * np.pi) * st_dev)
        * np.e ** (-0.5 * (float(x - mean) / st_dev) ** 2)
    )


def linear_kernel(x, y=None):
    """Linear kernel function for machine learning algorithms.

    Computes dot product between vectors x and y.
    If y is None, computes x.dot(x.T) for self-similarity.
    """
    if y is None:
        y = x
    return np.dot(x, y.T)


def polynomial_kernel(x, y=None, degree=2.0):
    """Polynomial kernel function for machine learning algorithms.

    Computes (1 + x.dot(y.T))^degree.
    Used in SVMs for non-linear classification.
    """
    if y is None:
        y = x
    return (1.0 + np.dot(x, y.T)) ** degree


def rbf_kernel(x, y=None, gamma=None):
    """Radial-basis function kernel (aka squared-exponential kernel)."""
    if y is None:
        y = x
    if gamma is None:
        gamma = 1.0 / x.shape[1]  # 1.0 / n_features
    return np.exp(
        -gamma
        * (
            -2.0 * np.dot(x, y.T)
            + np.sum(x * x, axis=1).reshape((-1, 1))
            + np.sum(y * y, axis=1).reshape((1, -1))
        )
    )


# ______________________________________________________________________________
# Grid Functions
#
# Utilities for grid-based AI problems including pathfinding, robotics,
# and spatial reasoning. Provides direction handling and distance calculations.


# Standard grid directions as (dx, dy) tuples
# EAST: right, NORTH: up, WEST: left, SOUTH: down
orientations = EAST, NORTH, WEST, SOUTH = [(1, 0), (0, 1), (-1, 0), (0, -1)]
# Turn directions: LEFT = counter-clockwise, RIGHT = clockwise
turns = LEFT, RIGHT = (+1, -1)


def turn_heading(heading, inc, headings=orientations):
    """Turn a heading by the specified increment.

    Args:
        heading: Current direction tuple (dx, dy)
        inc: Turn increment (+1 for right, -1 for left)
        headings: List of valid directions (default: EAST, NORTH, WEST, SOUTH)

    Returns:
        New heading after turning
    """
    return headings[(headings.index(heading) + inc) % len(headings)]


def turn_right(heading):
    """Turn a heading 90 degrees to the right (clockwise)."""
    return turn_heading(heading, RIGHT)


def turn_left(heading):
    """Turn a heading 90 degrees to the left (counter-clockwise)."""
    return turn_heading(heading, LEFT)


def distance(a, b):
    """The distance between two (x, y) points."""
    xA, yA = a
    xB, yB = b
    return np.hypot((xA - xB), (yA - yB))


def distance_squared(a, b):
    """The square of the distance between two (x, y) points."""
    xA, yA = a
    xB, yB = b
    return (xA - xB) ** 2 + (yA - yB) ** 2


# ______________________________________________________________________________
# Miscellaneous Functions
#
# General-purpose utilities including dependency injection, memoization,
# type checking, table printing, and algorithm testing framework.


class injection:
    """Dependency injection of temporary values for global functions/classes/etc.
    E.g., `with injection(DataBase=MockDataBase): ...`"""

    def __init__(self, **kwds):
        self.new = kwds

    def __enter__(self):
        self.old = {v: globals()[v] for v in self.new}
        globals().update(self.new)

    def __exit__(self, type, value, traceback):
        globals().update(self.old)


def memoize(fn, slot=None, maxsize=32):
    """Memoize fn: make it remember the computed value for any argument list.
    If slot is specified, store result in that slot of first argument.
    If slot is false, use lru_cache for caching the values.

    Args:
        fn: Function to memoize
        slot: Attribute name to store result in first argument (for object methods)
        maxsize: Maximum cache size when using lru_cache

    Returns:
        Memoized version of the function
    """
    if slot:

        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val
    else:

        @functools.lru_cache(maxsize=maxsize)
        def memoized_fn(*args):
            return fn(*args)

    return memoized_fn


def name(obj):
    """Try to find some reasonable name for the object."""
    return (
        getattr(obj, "name", 0)
        or getattr(obj, "__name__", 0)
        or getattr(getattr(obj, "__class__", 0), "__name__", 0)
        or str(obj)
    )


def isnumber(x):
    """Is x a number?"""
    return hasattr(x, "__int__")


def issequence(x):
    """Is x a sequence?"""
    return isinstance(x, collections.abc.Sequence)


def print_table(table, header=None, sep="   ", numfmt="{}"):
    """Print a list of lists as a table, so that columns line up nicely.
    header, if specified, will be printed as the first row.
    numfmt is the format for all numbers; you might want e.g. '{:.2f}'.
    (If you want different formats in different columns,
    don't use print_table.) sep is the separator between columns."""
    justs = ["rjust" if isnumber(x) else "ljust" for x in table[0]]

    if header:
        table.insert(0, header)

    table = [[numfmt.format(x) if isnumber(x) else x for x in row] for row in table]

    sizes = list(
        map(
            lambda seq: max(map(len, seq)), list(zip(*[map(str, row) for row in table]))
        )
    )

    for row in table:
        print(
            sep.join(
                getattr(str(x), j)(size) for (j, size, x) in zip(justs, sizes, row)
            )
        )


def open_data(name, mode="r"):
    """Open a data file from the aima-data directory.

    Args:
        name: Filename within the aima-data directory
        mode: File opening mode (default: 'r' for read)

    Returns:
        File object for the requested data file
    """
    aima_root = os.path.dirname(__file__)
    aima_file = os.path.join(aima_root, *["aima-data", name])

    return open(aima_file, mode=mode)


def failure_test(algorithm, tests):
    """Grades the given algorithm based on how many tests it passes.
    Most algorithms have arbitrary output on correct execution, which is difficult
    to check for correctness. On the other hand, a lot of algorithms output something
    particular on fail (for example, False, or None).
    tests is a list with each element in the form: (values, failure_output)."""
    return mean(int(algorithm(x) != y) for x, y in tests)


# ______________________________________________________________________________
# Symbolic Expression System
#
# A complete symbolic mathematics system for representing and manipulating
# logical and mathematical expressions. Supports operator overloading for
# natural mathematical notation and custom infix operators.

# See https://docs.python.org/3/reference/expressions.html#operator-precedence
# See https://docs.python.org/3/reference/datamodel.html#special-method-names


class Expr:
    """A mathematical expression with an operator and 0 or more arguments.
    op is a str like '+' or 'sin'; args are Expressions.
    Expr('x') or Symbol('x') creates a symbol (a nullary Expr).
    Expr('-', x) creates a unary; Expr('+', x, 1) creates a binary."""

    def __init__(self, op, *args):
        self.op = str(op)
        self.args = args

    # Operator overloads
    def __neg__(self):
        return Expr("-", self)

    def __pos__(self):
        return Expr("+", self)

    def __invert__(self):
        return Expr("~", self)

    def __add__(self, rhs):
        return Expr("+", self, rhs)

    def __sub__(self, rhs):
        return Expr("-", self, rhs)

    def __mul__(self, rhs):
        return Expr("*", self, rhs)

    def __pow__(self, rhs):
        return Expr("**", self, rhs)

    def __mod__(self, rhs):
        return Expr("%", self, rhs)

    def __and__(self, rhs):
        return Expr("&", self, rhs)

    def __xor__(self, rhs):
        return Expr("^", self, rhs)

    def __rshift__(self, rhs):
        return Expr(">>", self, rhs)

    def __lshift__(self, rhs):
        return Expr("<<", self, rhs)

    def __truediv__(self, rhs):
        return Expr("/", self, rhs)

    def __floordiv__(self, rhs):
        return Expr("//", self, rhs)

    def __matmul__(self, rhs):
        return Expr("@", self, rhs)

    def __or__(self, rhs):
        """Allow both P | Q, and P |'==>'| Q."""
        if isinstance(rhs, Expression):
            return Expr("|", self, rhs)
        else:
            return PartialExpr(rhs, self)

    # Reverse operator overloads
    def __radd__(self, lhs):
        return Expr("+", lhs, self)

    def __rsub__(self, lhs):
        return Expr("-", lhs, self)

    def __rmul__(self, lhs):
        return Expr("*", lhs, self)

    def __rdiv__(self, lhs):
        return Expr("/", lhs, self)

    def __rpow__(self, lhs):
        return Expr("**", lhs, self)

    def __rmod__(self, lhs):
        return Expr("%", lhs, self)

    def __rand__(self, lhs):
        return Expr("&", lhs, self)

    def __rxor__(self, lhs):
        return Expr("^", lhs, self)

    def __ror__(self, lhs):
        return Expr("|", lhs, self)

    def __rrshift__(self, lhs):
        return Expr(">>", lhs, self)

    def __rlshift__(self, lhs):
        return Expr("<<", lhs, self)

    def __rtruediv__(self, lhs):
        return Expr("/", lhs, self)

    def __rfloordiv__(self, lhs):
        return Expr("//", lhs, self)

    def __rmatmul__(self, lhs):
        return Expr("@", lhs, self)

    def __call__(self, *args):
        """Call: if 'f' is a Symbol, then f(0) == Expr('f', 0)."""
        if self.args:
            raise ValueError("Can only do a call for a Symbol, not an Expr")
        else:
            return Expr(self.op, *args)

    # Equality and repr
    def __eq__(self, other):
        """x == y' evaluates to True or False; does not build an Expr."""
        return (
            isinstance(other, Expr) and self.op == other.op and self.args == other.args
        )

    def __lt__(self, other):
        return isinstance(other, Expr) and str(self) < str(other)

    def __hash__(self):
        return hash(self.op) ^ hash(self.args)

    def __repr__(self):
        op = self.op
        args = [str(arg) for arg in self.args]
        if op.isidentifier():  # f(x) or f(x, y)
            return "{}({})".format(op, ", ".join(args)) if args else op
        elif len(args) == 1:  # -x or -(x + 1)
            return op + args[0]
        else:  # (x - y)
            opp = " " + op + " "
            return "(" + opp.join(args) + ")"


# Type definitions for the expression system
# An 'Expression' is either an Expr or a Number.
# Symbol is not an explicit type; it is any Expr with 0 args.

Number = (int, float, complex)
Expression = (Expr, Number)


def Symbol(name):
    """A Symbol is just an Expr with no args."""
    return Expr(name)


def symbols(names):
    """Return a tuple of Symbols; names is a comma/whitespace delimited str."""
    return tuple(Symbol(name) for name in names.replace(",", " ").split())


def subexpressions(x):
    """Yield the subexpressions of an Expression (including x itself)."""
    yield x
    if isinstance(x, Expr):
        for arg in x.args:
            yield from subexpressions(arg)


def arity(expression):
    """The number of sub-expressions in this expression."""
    if isinstance(expression, Expr):
        return len(expression.args)
    else:  # expression is a number
        return 0


# For operators that are not defined in Python, we allow new InfixOps:


class PartialExpr:
    """Given 'P |'==>'| Q, first form PartialExpr('==>', P), then combine with Q."""

    def __init__(self, op, lhs):
        self.op, self.lhs = op, lhs

    def __or__(self, rhs):
        return Expr(self.op, self.lhs, rhs)

    def __repr__(self):
        return "PartialExpr('{}', {})".format(self.op, self.lhs)


def expr(x):
    """Shortcut to create an Expression. x is a str in which:
    - identifiers are automatically defined as Symbols.
    - ==> is treated as an infix |'==>'|, as are <== and <=>.
    If x is already an Expression, it is returned unchanged. Example:
    >>> expr('P & Q ==> Q')
    ((P & Q) ==> Q)
    """
    return (
        eval(expr_handle_infix_ops(x), defaultkeydict(Symbol))
        if isinstance(x, str)
        else x
    )


# Custom infix operators for logical expressions
# ==> : implication, <== : reverse implication, <=> : biconditional
infix_ops = "==> <== <=>".split()


def expr_handle_infix_ops(x):
    """Given a str, return a new str with ==> replaced by |'==>'|, etc.
    >>> expr_handle_infix_ops('P ==> Q')
    "P |'==>'| Q"
    """
    for op in infix_ops:
        x = x.replace(op, "|" + repr(op) + "|")
    return x


class defaultkeydict(collections.defaultdict):
    """Like defaultdict, but the default_factory is a function of the key.
    >>> d = defaultkeydict(len); d['four']
    4
    """

    def __missing__(self, key):
        self[key] = result = self.default_factory(key)
        return result


class hashabledict(dict):
    """Allows hashing by representing a dictionary as tuple of key:value pairs.
    May cause problems as the hash value may change during runtime."""

    def __hash__(self):
        return 1


# ______________________________________________________________________________
# Data Structures: Queues and Priority Queues
#
# Implementation of priority queue with support for both min and max ordering.
# Stack and FIFOQueue are implemented as list and collections.deque elsewhere.
# PriorityQueue is implemented here with heap-based efficient operations.


class PriorityQueue:
    """A Queue in which the minimum (or maximum) element (as determined by f and
    order) is returned first.
    If order is 'min', the item with minimum f(x) is
    returned first; if order is 'max', then it is the item with maximum f(x).
    Also supports dict-like lookup."""

    def __init__(self, order="min", f=lambda x: x):
        self.heap = []
        if order == "min":
            self.f = f
        elif order == "max":  # now item with max f(x)
            self.f = lambda x: -f(x)  # will be popped first
        else:
            raise ValueError("Order must be either 'min' or 'max'.")

    def append(self, item):
        """Insert item at its correct position."""
        heapq.heappush(self.heap, (self.f(item), item))

    def extend(self, items):
        """Insert each item in items at its correct position."""
        for item in items:
            self.append(item)

    def pop(self):
        """Pop and return the item (with min or max f(x) value)
        depending on the order.

        Returns:
            Item with highest priority (lowest f(x) for min queue,
            highest f(x) for max queue)

        Raises:
            Exception: If the queue is empty
        """
        if self.heap:
            return heapq.heappop(self.heap)[1]
        else:
            raise Exception("Trying to pop from empty PriorityQueue.")

    def __len__(self):
        """Return current capacity of PriorityQueue."""
        return len(self.heap)

    def __contains__(self, key):
        """Return True if the key is in PriorityQueue."""
        return any([item == key for _, item in self.heap])

    def __getitem__(self, key):
        """Returns the first value associated with key in PriorityQueue.
        Raises KeyError if key is not present.

        Args:
            key: Item to look up in the queue

        Returns:
            Priority value (f(key)) for the item

        Raises:
            KeyError: If key is not found in the queue
        """
        for value, item in self.heap:
            if item == key:
                return value
        raise KeyError(str(key) + " is not in the priority queue")

    def __delitem__(self, key):
        """Delete the first occurrence of key.

        Args:
            key: Item to remove from the queue

        Raises:
            KeyError: If key is not found in the queue
        """
        try:
            del self.heap[[item == key for _, item in self.heap].index(True)]
        except ValueError:
            raise KeyError(str(key) + " is not in the priority queue")
        heapq.heapify(self.heap)


# ______________________________________________________________________________
# Useful Shorthands
#
# Convenient abbreviations and custom types for cleaner code representation.


class Bool(int):
    """Just like `bool`, except values display as 'T' and 'F' instead of 'True' and 'False'."""

    __str__ = __repr__ = lambda self: "T" if self else "F"


T = Bool(True)
F = Bool(False)
