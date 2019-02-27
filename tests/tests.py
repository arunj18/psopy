import numpy as np
from psopy.minimize import minimize_qpso,minimize
from psopy import init_feasible
from scipy.optimize import rosen
import time

def test_rosendisc():
    """Test against the Rosenbrock function constrained to a disk."""

    cons = ({'type': 'ineq', 'fun': lambda x: -x[0]**2 - x[1]**2 + 2},)
    x0 = init_feasible(cons, low=-1.5, high=1.5, shape=(1000, 2))
    sol = np.array([1., 1.])

    def rosen(x):
        return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    t0 = time.time()
    res = minimize_qpso(rosen, x0)
    converged = res.success
    print("Actual solution: ",sol)

    assert converged, res.message
    np.testing.assert_array_almost_equal(sol, res.x, 3)
    t1= time.time()
    x0 = init_feasible(cons, low=-1.5, high=1.5, shape=(1000, 2))
    sol = np.array([1., 1.])
    t2= time.time()
    res = minimize(rosen, x0)
    converged = res.success
    print("Actual solution: ",sol)

    assert converged, res.message
    np.testing.assert_array_almost_equal(sol, res.x, 3)
    t3 = time.time()
    print(t1-t0)
    print(t3-t2)
    input()
def test_mishra():
    """Test against the Mishra Bird function."""

    cons = (
        {'type': 'ineq', 'fun': lambda x: 25 - np.sum((x + 5) ** 2)},)
    x0 = init_feasible(cons, low=-10, high=0, shape=(1000, 2))
    sol = np.array([-3.130, -1.582])

    def mishra(x):
        cos = np.cos(x[0])
        sin = np.sin(x[1])
        return sin*np.e**((1 - cos)**2) + cos*np.e**((1 - sin)**2) + \
            (x[0] - x[1])**2

    res = minimize_qpso(mishra, x0)
    converged = res.success
    print("Actual solution: ",sol)
    assert converged, res.message
    np.testing.assert_array_almost_equal(sol, res.x, 3)


def test_unconstrained():
    """Test against the Rosenbrock function."""

    x0 = np.random.uniform(0, 2, (1000, 5))
    sol = np.array([1., 1., 1., 1., 1.])
    res = minimize_qpso(rosen, x0)
    converged = res.success
    assert converged, res.message
    np.testing.assert_array_almost_equal(sol, res.x, 3)

def test_constrained():
    """Test against the following function::

        y = (x0 - 1)^2 + (x1 - 2.5)^2

    under the constraints::

            x0 - 2.x1 + 2 >= 0
        -x0 - 2.x1 + 6 >= 0
        -x0 + 2.x1 + 2 >= 0
                x0, x1 >= 0

    """
    cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
            {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
            {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2},
            {'type': 'ineq', 'fun': lambda x: x[0]},
            {'type': 'ineq', 'fun': lambda x: x[1]})
    x0 = init_feasible(cons, low=0, high=2, shape=(1000, 2))
    options = {'stable_iter': 50}
    sol = np.array([1.4, 1.7])
    res = minimize_qpso(lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2, x0,
                    constraints=cons, options=options)
                    
    print("Actual solution: ",sol)
    converged = res.success
    assert converged, res.message
    np.testing.assert_array_almost_equal(sol, res.x, 3)


def test_ackley():
    """Test against the Ackley function."""

    x0 = np.random.uniform(-5, 5, (1000, 2))
    sol = np.array([0., 0.])

    def ackley(x):
        return -20 * np.exp(-.2 * np.sqrt(0.5 * (x[0]**2 + x[1]**2))) - \
            np.exp(.5 * (np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))) + \
            np.e + 20

    res = minimize_qpso(ackley, x0)
    
    print("Actual solution: ",sol)
    converged = res.success

    assert converged, res.message
    np.testing.assert_array_almost_equal(sol, res.x, 3)

def test_levi():
    """Test against the Levi function."""

    x0 = np.random.uniform(-10, 10, (1000, 2))
    sol = np.array([1., 1.])

    def levi(x):
        sin3x = np.sin(3*np.pi*x[0]) ** 2
        sin2y = np.sin(2*np.pi*x[1]) ** 2
        return sin3x + (x[0] - 1)**2 * (1 + sin3x) + \
            (x[1] - 1)**2 * (1 + sin2y)

    res = minimize_qpso(levi, x0)
    converged = res.success

    assert converged, res.message
    np.testing.assert_array_almost_equal(sol, res.x, 3)



test_rosendisc()
print("Rosenbrock function constrained to a disk test done")
print("-------------------------------------------------")
input()
test_mishra()
print("Mishra Bird function test done")
print("-------------------------------------------------")
input()
test_unconstrained()
print("Rosenbrock function test done")
print("-------------------------------------------------")
input()
test_constrained()
print("Function y = (x0 - 1)^2 + (x1 - 2.5)^2 test done")
print("-------------------------------------------------")
input()

test_ackley()
print("Ackley function test done")
print("-------------------------------------------------")
input()

test_levi()
print("Levi function test done")
print("-------------------------------------------------")
input()
