import numpy as np
from psopy import init_feasible
from psopy import minimize

def test_ackley():
    """Test against the Ackley function."""

    x0 = np.random.uniform(-5, 5, (1000, 2))
    sol = np.array([0., 0.])

    def ackley(x):
        return -20 * np.exp(-.2 * np.sqrt(0.5 * (x[0]**2 + x[1]**2))) - \
            np.exp(.5 * (np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))) + \
            np.e + 20

    res = minimize(ackley, x0)
    
    print("Actual solution: ",sol)
    print("Current solution: ", res.x)
    converged = res.success

    assert converged, res.message
    np.testing.assert_array_almost_equal(sol, res.x, 3)

test_ackley()