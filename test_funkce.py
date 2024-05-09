import scipy.integrate
import numpy as np
import funkce

def test_f_1():
    assert funkce.f_1(0) == 0

def test_monte_carlo():
    assert funkce.monte_carlo(funkce.f_1, 1000, 0, np.pi) > 0

def test_horni_odhad():
    assert abs(max(funkce.f_1(x) for x in funkce.horni_odhad(funkce.f_1, 10, 0, np.pi)) - max(funkce.f_1(x) for x in np.linspace(0, np.pi))) < 1e-3

def test_integrace():
    result, error = scipy.integrate.quad(funkce.f_1, 0, np.pi)
    assert abs(funkce.integrace(funkce.f_1, 10, 10000, 0, np.pi) - result) < 1e-2