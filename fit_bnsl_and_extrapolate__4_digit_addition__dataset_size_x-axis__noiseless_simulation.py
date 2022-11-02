
import matplotlib.pyplot as plt
import scipy.optimize
import time
import math
plt.style.use('seaborn-whitegrid')
import numpy as np

"""
Code to reproduce Figure 4 Left of arxiv.org/abs/2210.14891
"""

def bnsl_with_1_break(_x, a, b, c0, c1, d1, f1):
    y = a + b * _x**(-c0) * (1 + (_x/d1)**(1/f1))**(-c1 * f1)
    return y

def bnsl_with_1_break__log(_x, a, b, c0, c1, d1, f1):
    y = bnsl_with_1_break(_x, a, b, c0, c1, d1, f1)
    return np.log(y+1)

def bnsl_with_1_break__msle_optim(p, _x, _y):
    a, b, c0, c1, d1, f1 = p
    b = 1.25**b - 1 + 1e-8
    d1 = 1.25**d1  - 1 + 1e-8
    y = bnsl_with_1_break(_x, a, b, c0, c1, d1, f1)
    return np.mean((np.log(y+1)-np.log(_y+1))**2)

def bnsl_with_1_break__sle(p, _x, _y):
    a, b, c0, c1, d1, f1 = p
    y = bnsl_with_1_break(_x, a, b, c0, c1, d1, f1)
    return (np.log(y)-np.log(_y))**2


# ground_truth
a_gt = 0.41388453071629455
b_gt = 2.2772722897737556
c0_gt = 0.055077348404973955
c1_gt = 5.662903816010331
d1_gt = 612.5836172918001
f1_gt = 0.059193036393742314

x_points = 4096
x = np.array([i for i in range(1, x_points)]).astype(float)
y = bnsl_with_1_break(x, a_gt, b_gt, c0_gt, c1_gt, d1_gt, f1_gt)

if __name__ == '__main__':

    print("x ground_truth: ", x)
    print("y ground_truth: ", y)

    split_point = 405

    # set split_point to 390 to see what failure looks like
    # split_point = 390

    x1 = x[:split_point]
    y1 = y[:split_point]

    x2 = x[split_point:]
    y2 = y[split_point:]

    plt.plot(x2, y2, 'o', color=[0.0, 0.925, 0.0])
    plt.plot(x1, y1, 'o', color='black')

    # this range can be made as wide as you want and extrapolation will be the same, but grid search will run slower on a laptop if made wider
    p_grid = (slice(0.0, 2.5, .1), slice(0, 5, .25), slice(0, .2, 0.05), slice(0, 8, 0.5), slice(0, 35, 2.5), slice(0, .2, 0.05))

    start = time.time()
    res = scipy.optimize.brute(bnsl_with_1_break__msle_optim, p_grid, args=(x1, y1), full_output=False, finish=None, workers=-1)
    a, b, c0, c1, d1, f1 = res
    b = 1.25**b - 1 + 1e-8
    d1 = 1.25**d1  - 1 + 1e-8
    y_log = np.log(y1+1)
    popt, _ = scipy.optimize.curve_fit(bnsl_with_1_break__log, x1, y_log, p0=[a, b, c0, c1, d1, f1], maxfev=100000000)
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

    """
    plt.xscale('log')
    plt.yscale('log')
    #"""

    plt.xlim(140,983)
    plt.ylim(0,2.5)
    plt.savefig('plot__bnsl__fit_and_extrapolate__4_digit_addition__dataset_size_x-axis__noiseless_simulation.png', bbox_inches='tight')
    plt.show()

    plt.close()
    plt.cla()
    plt.clf()
