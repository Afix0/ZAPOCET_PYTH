import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import scipy.integrate
import random
import math


def f_1(x):
    return np.sin(x) * np.exp(np.cos(x))


def f_2(x):
    return np.sin(x) ** 6


def f_3(x):
    return x**2

import matplotlib.pyplot as plt
import numpy as np  # For numerical operations

def plot_funkce(f, start, end, num_points=200, title=None):
  """Vykresli funkci f na intervalu [start, end].

  Args:
      f: funkce
      start: spodni hranice intervalu
      end: horni hranice intervalu
      num_points: pocet bodu k vykresleni funkce (default: 200)
      label: nadpis (default: None)

  Returns:
      vykres funkce f
  """

  x = np.linspace(start, end, num_points)

  y = f(x)
  plt.plot(x, y)
  plt.xlabel("X")
  plt.ylabel("Y")
  plt.title(title)
  plt.grid(True)
  plt.show()



def monte_carlo(f, n, start, end):
    """Funkce pocita odhad hodnoty integralu funkce f pomoci metody Monte Carlo na intervalu [start, end].

    Args:
        f: funkce
        n: pocet kapek/strel
        start: spodni hranice intervalu
        end: horni hranice intervalu

    Returns:
        float: odhad hodnoty integralu
    """

    x_points = np.linspace(start, end, 1000)
    counter = 0
    max_value = max(f(x) for x in x_points)
    for _ in range(n):
        x_rand = random.uniform(start, end)
        y_rand = random.uniform(f(start), max_value)
        if y_rand <= f(x_rand):
            counter += 1
    return counter / n * (end - start) * abs(max_value - f(start))


def divide_interval(start, end, N):
    """Deli interval [start, end] na N podintervalu.

    Args:
        start: spodni hranice intervalu
        end: horni hranice intervalu
        N: pocet subintervalu

    Returns:
        list: list listu(podintervalu)
    """

    return np.split(np.linspace(start, end, 10000), N)


def argmax_interval(f, start, end):
    """Vraci hodnotu x pri ktere funkce f na intervalu [start, end] nabyva maxima.

    Args:
        f: funkce
        start: spodni hranice intervalu
        end: horni hranice intervalu

    Returns:
        float: x maximalizujici funkci f
    """

    result = minimize_scalar(lambda x: -f(x), bounds=(start, end), method='bounded')
    if result.success:
        return result.x
    else:
        raise ValueError("Minimizatce selhala")


def horni_odhad(f, N, start, end):
    """Funkce odhadujici zadanou funkci f se zhora.

    Args:
        f: funkce
        N: pocet podintervalu
        start: spodni hranice intervalu
        end: horni hranice intervalu

    Returns:
        list: list bodu x maximalizujicich funkci f na jednotlivych podintervalech
    """

    maximums = []
    for arr in divide_interval(start, end, N):
        maximums.append(argmax_interval(f, arr[0], arr[-1]))
    return maximums


def monte_carlo_interval(f, n, start, end):
    """Pocita odhad hodnoty integralu funkce f na [start, end] pomoci metody Monte Carlo.

    Args:
        f: funkce
        n: pocet kapek/strel
        start: spodni hranice intervalu
        end: horni hranice intervalu

    Returns:
        float: odhadovana hodnota integralu funkce f na [start, end]
    """

    counter = 0
    max_value = max(f(x) for x in np.linspace(start, end, 1000))
    for _ in range(n):
        x_rand = random.uniform(start, end)
        y_rand = random.uniform(0, max_value)
        if y_rand <= f(x_rand):
            counter += 1
    return counter / n * (end - start) * max_value

def plot_horni_odhad(f, N, start, end):
    """Vykresli horni obdelnikovy odhad funkce f na intervalu [start, end].

    Args:
        f: funkce
        N: pocet obdelniku
        start: spodni hranice intervalu
        end: horni hranice intervalu
    """

    X = np.linspace(start, end)
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 3, 2)

    x_max = horni_odhad(f, N, start, end)
    x_discrete = []

    for x in divide_interval(start, end, N):
        x_discrete.append(x[0])

    y_max = np.array([f(x) for x in horni_odhad(f, N, start, end)])

    plt.bar(x_discrete, y_max, width=(end - start) / N, alpha=0.2, edgecolor='b', align='edge')
    plt.plot(X, f(X))
    # plt.title(f"Horni obdelnikovy odhad funkce {f}")
    plt.grid(True)
    plt.show()


def integrace(f, N, n, start, end):
    """Funkce pocita odhad hodnoty integralu funkce f pomoci vylepsene metody Monte Carlo na intervalu [start, end].

    Args:
        f: funkce
        N: pocet podintervalu/obdelniku
        n: pocet kapek/strel
        start: spodni hranice intervalu
        end: horni hranice intervalu

    Returns:
        float: odhadovana hodnota integralu
    """

    intervals = divide_interval(start, end, N)
    suma = 0.0
    for arr in intervals:
        suma += monte_carlo_interval(f, n, arr[0], arr[-1])
    return suma


def plot_porovnani(f, N, start, end, num_points=100):
    """Vykresli absolutni chybu standardni a vylepsene Monte Carlo metody.

    Args:
        f: funkce
        N: pocet podintervalu/obdelniku
        n: pocet kapek/strel
        start: spodni hranice intervalu
        end: horni hranice intervalu
        num_points: pocet iteraci (default: 100).
    """

    iterace = np.linspace(1, num_points, num_points)
    result, error = scipy.integrate.quad(f, start, end)
    chyba_monte_carlo = [monte_carlo(f, i * 100, start, end) for i in range(1, num_points + 1)]
    chyba_vylepsene_monte_carlo = [
        integrace(f, N, int((i * num_points) / N), start, end) for i in range(1, num_points + 1)
    ]

    plt.plot(iterace, chyba_monte_carlo, label='Chyba standardni metody Monte Carlo')
    plt.plot(iterace, chyba_vylepsene_monte_carlo, label='Chyba obdelnikove Monte Carlo metody')
    plt.axhline(y=result, color='red', linestyle='--', label='Numericka hodnota')
    plt.title("Porovnani konvergence jednotlivych metod")
    plt.xlabel("Cislo iterace")
    plt.ylabel("Absolutni chyba")
    plt.xlim(0, num_points)
    plt.grid(True)
    plt.legend()
    plt.show()