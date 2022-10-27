
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.optimize import curve_fit
import time
import math
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

"""
Code to reproduce Figure 4 Left of arxiv.org/abs/2210.14891
"""

def bnsl_with_1_break(_x, a, b, c0, c1, d1, f1):
    y = a + b * _x**(-c0) * (1 + (_x/d1)**(1/f1))**(-c1 * f1)
    return y

def bnsl_with_1_break__log(_x, a, b, c0, c1, d1, f1):
    y = a + b * _x**(-c0) * (1 + (_x/d1)**(1/f1))**(-c1 * f1)
    return np.log(y+1)

def bnsl_with_1_break__msle_optim(p, _x, _y):
    a, b, c0, c1, d1, f1 = p
    b = 1.25**b - .9
    d1 = 1.25**d1  - .9
    y = a + b * _x**(-c0) * (1 + (_x/d1)**(1/f1))**(-c1 * f1)
    return np.mean((np.log(y+1)-np.log(_y+1))**2)

def bnsl_with_1_break__sle(p, _x, _y):
    a, b, c0, c1, d1, f1 = p
    y = a + b * _x**(-c0) * (1 + (_x/d1)**(1/f1))**(-c1 * f1)
    return (np.log(y)-np.log(_y))**2


x = np.array([160,  192,  256,  320,  384,  
              448,  480,  512,  544,  576,  
              608,  640,  672,  736,  800,
              864,  928])

y = np.array([2.13809046, 2.11813418, 2.08955508, 2.06988398, 2.05404987, 
              2.03837089, 2.02814281, 2.00496872, 1.95576149, 1.86313841, 
              1.70891537, 1.50637664, 1.29754721, 0.96559684, 0.75856477, 
              0.64768338, 0.55695445])

if __name__ == '__main__':

    print("x ground_truth: ", x)
    print("y ground_truth: ", y)

    split_point = 14

    x1 = x[:split_point]
    y1 = y[:split_point]

    x2 = x[split_point:]
    y2 = y[split_point:]

    plt.plot(x2, y2, 'o', color=[0.0, 0.925, 0.0])
    plt.plot(x1, y1, 'o', color='black')

    params = (x1, y1)

    p_grid = (slice(0.0, 1., .1), slice(0, 40, 2.5), slice(0, 1, 0.25), slice(0, 1, 0.25), slice(0, 40, 2.5), slice(0, 1, 0.25))

    start = time.time()
    res = scipy.optimize.brute(bnsl_with_1_break__msle_optim, p_grid, args=params, full_output=False, finish=None, workers=-1)
    a, b, c0, c1, d1, f1 = res
    b = 1.25**b - .9
    d1 = 1.25**d1  - .9
    y_log = np.log(y1+1)
    popt, _ = curve_fit(bnsl_with_1_break__log, x1, y_log, p0=[a, b, c0, c1, d1, f1], maxfev=100000000)
    a, b, c0, c1, d1, f1 = popt
    total_time = time.time() - start
    print("time: ", total_time)

    points = 4096
    x_tile = np.array([1.01**i * 10**0 for i in range(points)]).astype(float)

    print("a: ", a)
    print("b: ", b)
    print("c0: ", c0)
    print("c1: ", c1)
    print("d1: ", d1)
    print("f1: ", f1)

    pred = bnsl_with_1_break(x_tile.astype(float), a, b, c0, c1, d1, f1)
    plt.plot(x_tile, pred, color=[1.0, 0.125, 0.125], linewidth=2.5)

    sle = bnsl_with_1_break__sle((a, b, c0, c1, d1, f1), x, y)

    print("rmsle train: ", np.sqrt(np.mean(sle[:split_point])))
    print("rmsle extrapolate: ", np.sqrt(np.mean(sle[split_point:])))

    plt.title("4 Digit Addition")
    plt.xlabel("Training Dataset Size")
    plt.ylabel("Test Cross-Entropy")

    plt.xlim(140,983)
    plt.ylim(0,2.5)
    plt.savefig('plot__bnsl__fit_and_extrapolate__4_digit_addition__dataset_size_x-axis.png', bbox_inches='tight')
    plt.show()

    plt.close()
    plt.cla()
    plt.clf()
