import numpy as np
import matplotlib.pyplot as plt
import math


def Himmelblau_2d(X):
    return (X[0] ** 2 + X[1] - 11) ** 2 + (X[1] ** 2 + X[0] - 7) ** 2 - 5.


def griewank(X):
    dim = len(X)
    s = 0
    for i in range(dim):
        s += X[i] ** 2
    p = 1
    for i in range(dim):
        p *= math.cos(float(X[i]) / math.sqrt(i + 1))
    return 1 + float(s) / 4000.0 - float(p)


def styblinski_tang(X):
    dim = len(X)
    s = 0
    for i in range(dim):
        s += X[i] ** 4 - 16 * X[i] ** 2 + 5 * X[i]
    return 1/2 * float(s)


def michalewicz(X):
    dim = len(X)
    s = 0
    for i in range(dim):
        s += math.sin(X[i]) * (math.sin(((i + 1) * X[i]) / math.pi)) ** (2*10)
    return -float(s)


# def f1(X): # Griewank
#     k = len(X)
#     d = X.shape[1]
#     y = np.zeros((k, 1))
#     for i in range(k):
#         s = 0
#         prod = 1
#         for j in range(d):
#             s += X[i, j] ** 2
#             prod *= np.cos(X[i, j] / np.sqrt(j + 1))
#             y[i, 0] = s / 4000 - prod + 1
#     return y


def f4(X):
    dim = len(X)
    s = 0
    for i in range(dim):
        s += X[i] ** 2
    return float(s)


def f5(X):
    dim = len(X)
    s = 0
    for i in range(1, dim):
        s += X[i] ** 2
    return float(s) + X[0] ** 3

def f6(X):
    y = X[0]/2 + 3*X[1] + X[2]/8
    return y


def f7(t, b):
    delta = 2.1952/(t**3 * b)
    return delta


def f8(t, b):
    sigma = 504000/(t**2 * b)
    return sigma


def f9(h, l, t):
    tau1 = 6000/(np.sqrt(2) * h * l)
    tau2 = 6000 * (14 + 0.5 * l) * np.sqrt(0.25 * (l**2 + (h + t)**2)) / (2 * 0.707 * h * l * (l**2 / 12 + 0.25 * (h + t)**2))
    tau = np.sqrt((tau1**2 + tau2**2 + l*tau1*tau2) / np.sqrt(0.25 * (l**2 + (h + t)**2)))
    return tau


def f14(b, c):
    return np.sqrt((-b - np.sqrt(b**2 - 4*c))/2) / (2 * np.pi)


