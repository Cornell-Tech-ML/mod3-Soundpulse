"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable
#
# Implementation of a prelude of elementary functions.
# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$

# TODO: Implement for Task 0.1.

DELTA = 1e-6


def mul(x: float, y: float) -> float:
    """Multiplication of two numbers.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: The product of x and y.

    """
    return x * y


def id(x: float) -> float:
    """Identity function.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: The same input number.

    """
    return x


def add(x: float, y: float) -> float:
    """Addition of two numbers.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: The sum of x and y.

    """
    return x + y


def neg(x: float) -> float:
    """Negation of a number.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: The negation of x.

    """
    return -x


def lt(x: float, y: float) -> float:
    """Less than comparison of two numbers.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        bool: True if x is less than y, False otherwise.

    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Equality comparison of two numbers.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        bool: True if x is equal to y, False otherwise.

    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Maximum of two numbers.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: The maximum of x and y.

    """
    return x if x > y else y


def sigmoid(x: float) -> float:
    """Compute the sigmoid function of x.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: The sigmoid of x.

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def is_close(x: float, y: float) -> bool:
    """Check if two numbers are close.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        bool: True if x and y are close, False otherwise.

    """
    return (x - y < 1e-2) and (y - x < 1e-2)


def relu(x: float) -> float:
    """Compute the ReLU function of x.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: The ReLU of x.

    """
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Compute the natural logarithm of x.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: The natural logarithm of x.

    """
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Compute the exponential of x.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: The exponential of x.

    """
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """Compute the derivative of the natural logarithm function and multiply it by a second argument.

    Args:
    ----
        x (float): The input value for the logarithm function (must be greater than 0).
        d (float): The second argument to multiply with the derivative.

    Returns:
    -------
        float: The result of the derivative of log(x) multiplied by d.

    """
    return d / (x + EPS)


def inv(x: float) -> float:
    """Compute the reciprocal of x.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: The reciprocal of x.

    """
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """Compute the derivative of the reciprocal function and multiply it by a second argument.

    Args:
    ----
        x (float): The input value for the reciprocal function.
        d (float): The second argument to multiply with the derivative.

    Returns:
    -------
        float: The result of the derivative of 1/x multiplied by d, which is -d / x**2.

    """
    return -(1.0 / x**2) * d


def relu_back(x: float, d: float) -> float:
    """Compute the derivative of the ReLU function and multiply it by a second argument.

    Args:
    ----
        x (float): The input value for the ReLU function.
        d (float): The second argument to multiply with the derivative.

    Returns:
    -------
        float: The result of the derivative of ReLU(x) multiplied by d.

    """
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher order map.

    Args:
    ----
        fn: Function from one value to one value

    Returns:
    -------
        A function that takes a list, applies 'fn' to each element, and returns a
        new list

    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order zipwidth (or map2).

    Args:
    ----
        fn: combine two values

    Returns:
    -------
        Function that takes two equally sized lists 'ls1' and 'ls2', produces a new list
        by applying fn(x, y) on each pair of elements.

    """

    def _zipWidth(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret

    return _zipWidth


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""Higher order reduce.

    Args:
    ----
        fn: combine two values
        start: start value $x_0$

    Returns:
    -------
        Function that takes a list 'ls' of elements
        $x_1 \ldots x_n$ and computes the reduction :math: 'fn(x_3, fn(x_2
        , fn(x_1, x_0)))'

    """

    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return _reduce


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Use 'map' and 'neg' to negate each element in 'ls'"""
    return map(neg)(ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add the elements of 'ls1' and 'ls2' using 'zipWidth' and 'add'."""
    return zipWith(add)(ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Sum up a list using 'reduce' and 'add'."""
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Product of a list using 'reduce' and 'mul'."""
    return reduce(mul, 1.0)(ls)
