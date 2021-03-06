import numpy as np
from psopy.minimize import minimize_qpso,minimize
from psopy import init_feasible
from psopy.constraints import init_lorenz_chaos
from scipy.optimize import rosen
import time
import pandas as pd
from statistics import mean,stdev
from tqdm import tqdm
import multiprocessing as mp
import itertools as it
import dill
import csv

def rosen_def(x):
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
    #if shape == 0:
    #x0 = np.random.uniform(0, 2, (1000, 5))
        #print('here')
    x0 = init_lorenz_chaos(shape=shape,low=low,high=high) 
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
  #  if high is None:
    #x0 = np.random.uniform(0, 2, (1000, 5))
   # else:
    x0 = init_lorenz_chaos(shape=shape,low=low,high=high) 
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
    #tol = [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10]
    tol = [1e-1]
    qpso_fai1l = []
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
        test_generic([to,cons,sol,test_func,low,high,shape])
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
    return -x[0]**2 - x[1]**2 + 2

if __name__ == '__main__':
    mp.freeze_support()
    #d = open('qpso_levy.csv','a')
    low = -1.5
    high = 1.5
    shape = (1000,2)
    sol = np.array([1.,1.])

    
    print("Testing qpso with levy")
    print("rosen")
    fd = open('qpso_levy_rosen.csv','a')  
    wr = csv.writer(fd, dialect='excel')
    for i in range(30):
        #print(i)
        cons = ({'type': 'ineq', 'fun': rosen},)
        #x0 = init_lorenz_chaos(shape=shape,low=low,high=high) 
        x0 = init_lorenz_chaos(shape=shape,low=low,high=high) 
        res = minimize_qpso(rosen, x0, options={'levy_rate':1})
        res_p = [res.fun, res.nit, res.nsit, res.status, res.success, res.x[0], res.x[1]]
        wr.writerow(res_p)
    fd.close()
    print("rosen_def")
    fd = open('qpso_levy_rosen_def.csv','a')  
    wr = csv.writer(fd, dialect='excel')
    for i in range(30):
        print(i)
        cons = ({'type': 'ineq', 'fun': rosen_def},)
        x0 = init_lorenz_chaos(shape=shape,low=low,high=high) 
        res = minimize_qpso(rosen_def, x0, options={'levy_rate':1})
        res_p = [res.fun, res.nit, res.nsit, res.status, res.success, res.x[0], res.x[1]]
        wr.writerow(res_p)
    fd.close()
    print("mishra")
    fd = open('qpso_levy_mishra.csv','a')  
    wr = csv.writer(fd, dialect='excel')
    for i in range(30):
        print(i)
        cons = ({'type': 'ineq', 'fun': mishra},)
        x0 = init_lorenz_chaos(shape=shape,low=low,high=high) 
        res = minimize_qpso(mishra, x0, options={'levy_rate':1})
        res_p = [res.fun, res.nit, res.nsit, res.status, res.success, res.x[0], res.x[1]]         
        wr.writerow(res_p)
    fd.close()
    print("ackley")
    fd = open('qpso_levy_ackley.csv','a')  
    wr = csv.writer(fd, dialect='excel')
    for i in range(30):
        print(i)
        cons = ({'type': 'ineq', 'fun': ackley},)
        x0 = init_lorenz_chaos(shape=shape,low=low,high=high) 
        res = minimize_qpso(ackley, x0, options={'levy_rate':1})
        res_p = [res.fun, res.nit, res.nsit, res.status, res.success, res.x[0], res.x[1]]         
        wr.writerow(res_p)
    fd.close()
    print("levi")
    fd = open('qpso_levy_levi.csv','a')  
    wr = csv.writer(fd, dialect='excel')
    for i in range(30):
        print(i)
        cons = ({'type': 'ineq', 'fun': levi},)
        x0 = init_lorenz_chaos(shape=shape,low=low,high=high) 
        res = minimize_qpso(levi, x0, options={'levy_rate':1})
        res_p = [res.fun, res.nit, res.nsit, res.status, res.success, res.x[0], res.x[1]]         
        wr.writerow(res_p)
    fd.close()
    

    #----------------------------------------------
    print("Testing qpso")
    print("rosen")
    fd = open('qpso_rosen.csv','a')  
    wr = csv.writer(fd, dialect='excel')
    for i in range(30):
        print(i)
        cons = ({'type': 'ineq', 'fun': rosen},)
        x0 = init_lorenz_chaos(shape=shape,low=low,high=high) 
        res = minimize_qpso(rosen, x0)
        res_p = [res.fun, res.nit, res.nsit, res.status, res.success, res.x[0], res.x[1]]         
        wr.writerow(res_p)
    fd.close()
    print("rosen_def")
    fd = open('qpso_rosen_def.csv','a')  
    wr = csv.writer(fd, dialect='excel')
    for i in range(30):
        print(i)
        cons = ({'type': 'ineq', 'fun': rosen_def},)
        x0 = init_lorenz_chaos(shape=shape,low=low,high=high) 
        res = minimize_qpso(rosen_def, x0)
        res_p = [res.fun, res.nit, res.nsit, res.status, res.success, res.x[0], res.x[1]]         
        wr.writerow(res_p)
    fd.close()
    print("mishra")
    fd = open('qpso_mishra.csv','a')  
    wr = csv.writer(fd, dialect='excel')
    for i in range(30):
        print(i)
        cons = ({'type': 'ineq', 'fun': mishra},)
        x0 = init_lorenz_chaos(shape=shape,low=low,high=high) 
        res = minimize_qpso(mishra, x0)
        res_p = [res.fun, res.nit, res.nsit, res.status, res.success, res.x[0], res.x[1]]         
        wr.writerow(res_p)
    fd.close()
    print("ackley")
    fd = open('qpso_ackley.csv','a')  
    wr = csv.writer(fd, dialect='excel')
    for i in range(30):
        print(i)
        cons = ({'type': 'ineq', 'fun': ackley},)
        x0 = init_lorenz_chaos(shape=shape,low=low,high=high) 
        res = minimize_qpso(ackley, x0)
        res_p = [res.fun, res.nit, res.nsit, res.status, res.success, res.x[0], res.x[1]]         
        wr.writerow(res_p)
    fd.close()
    print("levi")
    fd = open('qpso_levi.csv','a')  
    wr = csv.writer(fd, dialect='excel')
    for i in range(30):
        print(i)
        cons = ({'type': 'ineq', 'fun': levi},)
        x0 = init_lorenz_chaos(shape=shape,low=low,high=high) 
        res = minimize_qpso(levi, x0)
        res_p = [res.fun, res.nit, res.nsit, res.status, res.success, res.x[0], res.x[1]]         
        wr.writerow(res_p)
    fd.close()

    #-----------------------------------------------------
    
    print("Testing qpso with levy and decay")
    print("rosen")
    fd = open('qpso_levy_decay_rosen.csv','a')  
    wr = csv.writer(fd, dialect='excel')
    for i in range(30):
        print(i)
        cons = ({'type': 'ineq', 'fun': rosen},)
        x0 = init_lorenz_chaos(shape=shape,low=low,high=high) 
        res = minimize_qpso(rosen, x0, options={'levy_rate':1, 'decay_rate':1})
        res_p = [res.fun, res.nit, res.nsit, res.status, res.success, res.x[0], res.x[1]]         
        wr.writerow(res_p)
    fd.close()
    print("rosen_def")
    fd = open('qpso_levy_decay_rosen_def.csv','a')  
    wr = csv.writer(fd, dialect='excel')
    for i in range(30):
        print(i)
        cons = ({'type': 'ineq', 'fun': rosen_def},)
        x0 = init_lorenz_chaos(shape=shape,low=low,high=high) 
        res = minimize_qpso(rosen_def, x0, options={'levy_rate':1, 'decay_rate':1})
        res_p = [res.fun, res.nit, res.nsit, res.status, res.success, res.x[0], res.x[1]]         
        wr.writerow(res_p)
    fd.close()
    print("mishra")
    fd = open('qpso_levy_decay_mishra.csv','a') 
    wr = csv.writer(fd, dialect='excel')
    for i in range(30):
        print(i)
        cons = ({'type': 'ineq', 'fun': mishra},)
        x0 = init_lorenz_chaos(shape=shape,low=low,high=high) 
        res = minimize_qpso(mishra, x0, options={'levy_rate':1, 'decay_rate':1})
        res_p = [res.fun, res.nit, res.nsit, res.status, res.success, res.x[0], res.x[1]]         
        wr.writerow(res_p)
    fd.close()
    print("ackley")
    fd = open('qpso_levy_decay_ackley.csv','a')  
    wr = csv.writer(fd, dialect='excel')
    for i in range(30):
        print(i)
        cons = ({'type': 'ineq', 'fun': ackley},)
        x0 = init_lorenz_chaos(shape=shape,low=low,high=high) 
        res = minimize_qpso(ackley, x0, options={'levy_rate':1, 'decay_rate':1})
        res_p = [res.fun, res.nit, res.nsit, res.status, res.success, res.x[0], res.x[1]]         
        wr.writerow(res_p)
    fd.close()
    print("levi")
    fd = open('qpso_levy_decay_levi.csv','a')  
    wr = csv.writer(fd, dialect='excel')
    for i in range(30):
        print(i)
        cons = ({'type': 'ineq', 'fun': levi},)
        x0 = init_lorenz_chaos(shape=shape,low=low,high=high) 
        res = minimize_qpso(levi, x0, options={'levy_rate':1, 'decay_rate':1})
        res_p = [res.fun, res.nit, res.nsit, res.status, res.success, res.x[0], res.x[1]]         
        wr.writerow(res_p)
    fd.close()

    #-------------------------------------------------------------------------

    print("Testing pso")
    print("rosen")
    fd = open('pso_rosen.csv','a')  
    wr = csv.writer(fd, dialect='excel')
    for i in range(30):
        print(i)
        cons = ({'type': 'ineq', 'fun': rosen},)
        x0 = init_lorenz_chaos(shape=shape,low=low,high=high) 
        res = minimize(rosen, x0)
        res_p = [res.fun, res.nit, res.nsit, res.status, res.success, res.x[0], res.x[1]]         
        wr.writerow(res_p)
    fd.close()
    print("rosen_def")
    fd = open('pso_rosen_def.csv','a')  
    wr = csv.writer(fd, dialect='excel')
    for i in range(30):
        print(i)
        cons = ({'type': 'ineq', 'fun': rosen_def},)
        x0 = init_lorenz_chaos(shape=shape,low=low,high=high) 
        res = minimize(rosen_def, x0)
        res_p = [res.fun, res.nit, res.nsit, res.status, res.success, res.x[0], res.x[1]]         
        wr.writerow(res_p)
    fd.close()
    print("mishra")
    fd = open('pso_mishra.csv','a')  
    wr = csv.writer(fd, dialect='excel')
    for i in range(30):
        print(i)
        cons = ({'type': 'ineq', 'fun': mishra},)
        x0 = init_lorenz_chaos(shape=shape,low=low,high=high) 
        res = minimize(mishra, x0)
        res_p = [res.fun, res.nit, res.nsit, res.status, res.success, res.x[0], res.x[1]]         
        wr.writerow(res_p)
    fd.close()
    print("ackley")
    fd = open('pso_ackley.csv','a')  
    wr = csv.writer(fd, dialect='excel')
    for i in range(30):
        print(i)
        cons = ({'type': 'ineq', 'fun': ackley},)
        x0 = init_lorenz_chaos(shape=shape,low=low,high=high) 
        res = minimize(ackley, x0)
        res_p = [res.fun, res.nit, res.nsit, res.status, res.success, res.x[0], res.x[1]]         
        wr.writerow(res_p)
    fd.close()
    print("levi")
    fd = open('pso_levi.csv','a')  
    wr = csv.writer(fd, dialect='excel')
    for i in range(30):
        print(i)
        cons = ({'type': 'ineq', 'fun': levi},)
        x0 = init_lorenz_chaos(shape=shape,low=low,high=high) 
        res = minimize(levi, x0)
        res_p = [res.fun, res.nit, res.nsit, res.status, res.success, res.x[0], res.x[1]]         
        wr.writerow(res_p)
    fd.close()


    #x0 = np.random.uniform(-5, 5, (1000, 2))
    #x0 = np.random.uniform(0, 2, (1000, 5))
    #res = minimize(rosen, x0)
    #converged = res.success
    #stats_func(rosen,"test_levy.csv",cons,low,high,shape,sol)
    print("Done!")
    print("-------------------------------------------------")
    #test_generic(to,cons,sol,test_func,low,high,shape)
