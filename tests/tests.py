import numpy as np
from psopy.minimize import minimize_qpso,minimize
from psopy import init_feasible
from scipy.optimize import rosen
import time
import pandas as pd
from statistics import mean,stdev
from tqdm import tqdm
import multiprocessing as mp
import itertools as it
import dill

def rosen(x):
    '''
    cons = ({'type': 'ineq', 'fun': lambda x: -x[0]**2 - x[1]**2 + 2},)
    low = -1.5
    high=1.5
    shape = (1000,2)
    sol = np.array([1.,1.])
    '''
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
def mishra(x):
    '''
    cons = ({'type': 'ineq', 'fun': lambda x: 25 - np.sum((x + 5) ** 2)},)
    low = -10
    high = 0
    shape = (1000,2)
    sol = np.array([-3.130, -1.582])

    '''
    cos = np.cos(x[0])
    sin = np.sin(x[1])
    return sin*np.e**((1 - cos)**2) + cos*np.e**((1 - sin)**2) + \
        (x[0] - x[1])**2

def unconstrained(x):
    '''
    x0 = np.random.uniform(0, 2, (1000, 5))
    sol = np.array([1., 1., 1., 1., 1.])
    '''
    return rosen(x)


def constrained(x):
    '''
    cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
            {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
            {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2},
            {'type': 'ineq', 'fun': lambda x: x[0]},
            {'type': 'ineq', 'fun': lambda x: x[1]})
    low = 0
    high = 2
    shape(1000,2)
    stable_iter = 50
    sol = np.array([1.4, 1.7])

    '''
    return (x[0] - 1)**2 + (x[1] - 2.5)**2

def ackley(x):
    '''
    x0 = np.random.uniform(-5, 5, (1000, 2))
    sol = np.array([0., 0.])

    
    '''
    return -20 * np.exp(-.2 * np.sqrt(0.5 * (x[0]**2 + x[1]**2))) - \
            np.exp(.5 * (np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))) + \
            np.e + 20
def levi(x):
    '''
    x0 = np.random.uniform(-10, 10, (1000, 2))
    sol = np.array([1., 1.])

    '''

    sin3x = np.sin(3*np.pi*x[0]) ** 2
    sin2y = np.sin(2*np.pi*x[1]) ** 2
    return sin3x + (x[0] - 1)**2 * (1 + sin3x) + \
        (x[1] - 1)**2 * (1 + sin2y)


def test_generic(args):
    """Test against the General function"""
    (tol,cons,sol,test_func,low,high,shape) = args
    if low is None:
        x0 = np.copy(low)
    else:    
        x0 = init_feasible(cons, low=low, high=high, shape=shape)
    t0 = time.time()
    res = minimize_qpso(test_func, x0, tol=tol)
    t1= time.time()
    converged = res.success
    qpso_converged = 0
    qpso_nit = res.nit
    try:
        np.testing.assert_array_almost_equal(sol, res.x, 3)
    except:
        qpso_converged = 1
    if low is None:
        x0 = np.copy(low)
    else:
        x0 = init_feasible(cons, low=low, high=high, shape=shape)
    t2= time.time()
    res = minimize(test_func,x0, tol=tol)
    t3 = time.time()
    converged = res.success
    pso_converged = 0
    pso_nit = res.nit
    assert converged, res.message
    try:
        np.testing.assert_array_almost_equal(sol, res.x, 3)
    except:
        pso_converged = 1
    
    return qpso_converged, qpso_nit ,t1-t0, pso_converged , pso_nit , t3-t2



def stats_func(test_func,file_name,cons,high,low,shape,sol):
    tol = [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12,1e-13,1e-14,1e-15,1e-16,1e-17,1e-18,1e-19,1e-20]
    qpso_fail = []
    pso_fail = []
    mean_qpso_nit = []
    stddev_qpso_nit = []
    mean_qpso_time = []
    stddev_qpso_time = []
    mean_pso_time = []
    stddev_pso_time = []
    mean_pso_nit = []
    stddev_pso_nit = []
    p = mp.Pool(4)
    for to in tqdm(tol):
        stats_list = [[],[],[],[]]
        qpso_conv = 0
        pso_conv = 0
        results = []
        r = p.map_async(test_generic,it.repeat([to,cons,sol,test_func,low,high,shape],100),callback=results.append)
        r.get()
        #test_generic([to,cons,sol,test_func,low,high,shape])
        #print(results[0])
        for i in results[0]:
        #for it in tqdm(range(10)):
        #    qpso_converged, qpso_nit ,time_qpso,pso_converged,pso_nit,time_pso = test_generic(to,cons,sol,test_func,low,high,shape)
            qpso_conv+=i[0]
            pso_conv+=i[3]
            stats_list[0].append(i[1])
            stats_list[1].append(i[2])
            stats_list[2].append(i[4])
            stats_list[3].append(i[5])
        qpso_fail.append(qpso_conv)
        pso_fail.append(pso_conv)
        mean_qpso_nit.append(mean(stats_list[0]))
        stddev_qpso_nit.append(stdev(stats_list[0]))
        mean_qpso_time.append(mean(stats_list[1]))
        stddev_qpso_time.append(stdev(stats_list[1]))
        mean_pso_nit.append(mean(stats_list[2]))
        stddev_pso_nit.append(stdev(stats_list[2]))
        mean_pso_time.append(mean(stats_list[3]))
        stddev_pso_time.append(stdev(stats_list[3]))

    stats = {
        'tol' : tol,
        'qpso_fail' : qpso_fail,
        'mean_qpso_nit' : mean_qpso_nit,
        'stddev_qpso_nit' : stddev_qpso_nit,
        'mean_qpso_time' : mean_qpso_time,
        'stddev_qpso_time' : stddev_qpso_time,
        'pso_fail' : pso_fail,
        'mean_pso_nit' : mean_pso_nit,
        'stddev_pso_nit' : stddev_pso_nit,
        'mean_pso_time' : mean_pso_time,
        'stddev_pso_time' : stddev_pso_time
    }
    df = pd.DataFrame(data=stats)
    df.to_csv(file_name)

def constraint_1(x):
    return 25 - np.sum((x + 5) ** 2)
if __name__ == '__main__':
    mp.freeze_support()
    cons = ({'type': 'ineq', 'fun': constraint_1},)      
    low=-10
    high=0
    shape=(1000, 2)    
    sol = np.array([-3.130, -1.582])

    stats_func(mishra,"mishra.csv",cons,high,low,shape,sol)
    print("Done!")
    print("-------------------------------------------------")
    #test_generic(to,cons,sol,test_func,low,high,shape)