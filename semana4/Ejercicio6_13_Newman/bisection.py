def binary_search(f, a, b, eps=1e-6, Nmax=1000):
    """Bisección en [a,b] hasta |f(x)| < eps."""
    assert f(a) * f(b) < 0, "f(a) y f(b) deben tener signos opuestos"
    for _ in range(Nmax):
        mid = (a + b) / 2
        if abs(f(mid)) < eps:
            return mid
        if f(a) * f(mid) < 0:
            b = mid
        else:
            a = mid
    return (a + b) / 2
