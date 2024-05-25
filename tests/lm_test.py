from timeit import default_timer

from scipy.optimize import least_squares
import numpy as np


def model(x, t):
    return x[0] + x[1] * np.exp(x[2] * t)


def fun(x, t, y):
    return model(x, t) - y


t = np.linspace(0, 10, 10)
y = model([0.5, 2.0, -1], t) + 0.1 * np.random.randn(t.size)


# SciPy Benchmark
start_time = default_timer()
res = least_squares(fun, x0=[1, 1, 0], args=(t, y), method="lm")
scipy_time = default_timer() - start_time


print(f"SciPy LM Optimization Time: {scipy_time} seconds")
