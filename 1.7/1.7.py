import numpy as np
import matplotlib.pyplot as plt
import math


def funkcja_a(x):
    a, b, c = 7, 4, 0
    return (a - 4) * x ** 2 + (b - 5) * x + c - 6


def funkcja_b(x):
    e_x = math.exp(x)
    return e_x / (e_x + 1)


def zadanie_1_7():
    data = np.arange(1, 10)
    results1 = np.array([funkcja_a(x) for x in data])
    results2 = np.array([funkcja_b(x) for x in data])

    # pokaż wykres
    plt.plot(data, results1, label = "A")
    plt.plot(data, results2, label = "B")

    plt.xlabel('x')
    plt.ylabel('value')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    zadanie_1_7()
