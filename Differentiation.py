# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 19:30:08 2023

@author: Sinurat
"""
import numpy as np
import matplotlib.pyplot as plt

def forward_diff(f, x, dx):
    """
    Calculate a numerical approximation of f'(x) using first-order accurate
    forward difference scheme

    Parameters
    ----------
    f : callable object, function with unknown gradient
    x : float, point to evaluate the gradient
    dx : float, resolution of scheme.

    Returns
    -------
    TYPE
        Approximation gradient value of f at point x.

    """
    return (f(x + dx) - f(x)) / dx

def backward_diff(f, x, dx):
    """
    Calculate a numerical approximation of f'(x) using first-order accurate
    backward difference scheme

    Parameters
    ----------
    f : callable object, function with unknown gradient
    x : float, point to evaluate the gradient
    dx : float, resolution of scheme.

    Returns
    -------
    TYPE
        Approximation gradient value of f at point x.

    """
    return (f(x) - f(x - dx)) / dx

def centered_diff(f, x, dx):
    """
    Calculate a numerical approximation of f'(x) using second-order accurate
    centered difference scheme

    Parameters
    ----------
    f : callable object, function with unknown gradient
    x : float, point to evaluate the gradient
    dx : float, resolution of scheme.

    Returns
    -------
    TYPE
        Approximation gradient value of f at point x.

    """
    return (f(x + dx) - f(x - dx)) / (2 * dx)

def f(x):
    """
    Calculate the analytic solution using numpy exponential

    Parameters
    ----------
    x : float, point to evaluate the gradient.

    Returns
    -------
    TYPE
        exponential value at point x.

    """
    return np.exp(x)

def analytic_sol(N):
    """
    Calculate the derivative of numpy exponential and number of array

    Parameters
    ----------
    N : Integer, number of evenly spaced points.

    Returns
    -------
    x : integer, interval spaced points.
    dydx : derivative of numpy exponential.

    """
    x = np.linspace(0, 1, N+1)
    dydx = np.exp(x)
    return x, dydx

#%% Question 2

def grad_approx(N):
    """
    Calculate centered second order and compare it with exponential gradient
    Then, it will be shown in graph.

    Parameters
    ----------
    N : Integer, number of evenly spaced points.

    Returns
    -------
    Graph shows the difference result between approximation and exponential
    gradient.

    """
    dx = 1 / N
    x = np.linspace(0, 1, N+1)
    y = f(x)
    dydx = np.zeros(N+1)
    
    #calling forward and backward approximation function
    dydx[0] = backward_diff(f, x[0], dx)
    dydx[N] = forward_diff(f, x[N], dx)
    #calling centered approximation function for looping
    for j in range(1, N):
        dydx[j] = centered_diff(f, x[j], dx) 
    return x, dydx, y
    
def plot_grad_approx(N):
    """
    Plotting centered difference approximation and exponential gradient 

    Parameters
    ----------
    N : Integer, number of evenly spaced points..

    Returns
    -------
    plotting result

    """

    x, dydx, y = grad_approx(N)
    #Plotting graph, label, grid, and legend
    plt.figure(figsize=(7, 7))
    plt.plot(x, dydx, 'r--', label = 'centered difference', ms = 1)
    plt.plot(x, y, 'b', label = 'exponential gradient', ms = 1) 
    plt.xlabel('N')
    plt.ylabel('y = exp(x)')
    plt.grid(which='both')
    plt.legend()
    plt.show()
    return

plot_grad_approx(10) 


#%% Question 3
def order_accuracy(delta_x_mat, error_sch):
    """
    Calculate how to find order accuracy of finite difference

    Parameters
    ----------
    delta_x_mat : float, array, resolution of matrix.
    error_sch : float, array, error number of numerical solution.

    Returns
    -------
    estimate the order of convergence.

    """
    return (np.log(error_sch[1]) - np.log(error_sch[0])) /\
        (np.log(delta_x_mat[1]) - np.log(delta_x_mat[0]))

def order_diff():
    """
    Calculate the order accuracy of forward, centered, and backward difference

    Returns
    -------
    delta_x_mat : float, array, resolution of matrix.
    error_matrix : float, array, error number of numerical solution.

    """
    N_values = [2, 5, 10, 20, 50, 100]
    error = []
    delta_x = []

    for i, Ni in enumerate(N_values): 
        y_exact = analytic_sol(Ni)[1]
        y_approx = grad_approx(Ni)[1]
    
        forward_error = abs(y_exact[0] - y_approx[0])
        centered_error = abs(y_exact[1] - y_approx[1])
        backward_error = abs(y_exact[-1] - y_approx[-1])
        error.append([forward_error, centered_error, backward_error])
        delta_x.append(1 / Ni)
    
    error_matrix = np.asarray(error)
    delta_x_mat = np.asarray(delta_x)

    n_forw = order_accuracy(delta_x_mat, error_matrix[:,0])
    n_cent = order_accuracy(delta_x_mat, error_matrix[:,1])  
    n_back = order_accuracy(delta_x_mat, error_matrix[:,2])
    print(f"order of accuracy of forward difference: {n_forw}")
    print(f"order of accuracy of forward difference: {n_cent}")
    print(f"order of accuracy of forward difference: {n_back}")
    return delta_x_mat, error_matrix
    
def plot_order_diff():
    """
    Plotting the error of the forward, centered, and backward difference

    Returns
    -------
    plotting result

    """
    delta_x, error_matrix = order_diff()
    
    #Plotting graph, label, grid, and legend
    plt.figure(figsize=(7, 7))
    plt.plot(delta_x, error_matrix[:, 0], 'r-', label = 'Forward Difference')
    plt.plot(delta_x, error_matrix[:, 1], 'y-', label = 'Centered Difference')
    plt.plot(delta_x, error_matrix[:, 2], 'b-', label = 'Backward Difference')
    plt.plot(delta_x, error_matrix, 'g^')
    plt.xlabel(r'$\Delta$x')
    plt.ylabel(r'$\epsilon$x')
    plt.grid(which='both')
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.show()
    abc = 5 * 5 / 5
    print (abc)
    return

plot_order_diff()


#%% Question 4

def centered_diff_4th(f, x, dx):
    """
    Calculate a numerical approximation of a function using forth-order accuracy
    centered difference scheme.

    Parameters
    ----------
    f : callable object, function with unknown gradient
    x : float, point to evaluate the gradient
    dx : float, resolution of scheme.

    Returns
    -------
    Approximation gradient value of f at point x using 4th order accuracy.

    """
    return (f(x - 2 * dx) - 8 * f(x - dx) + 8 * f(x + dx) - f(x + 2 * dx)) / (12 * dx)

def forward_diff_2nd(f, x, dx):
    """
    Calculate a numerical approximation of a function using second-order accuracy
    forward difference scheme.

    Parameters
    ----------
    f : callable object, function with unknown gradient
    x : float, point to evaluate the gradient
    dx : float, resolution of scheme.

    Returns
    -------
    Approximation gradient value of f at point x using 3th order accuracy.

    """
    return (-f(x + 2 * dx) + 4 * f(x + dx) - 3 * f(x)) / (2 * dx)

def backward_diff_2nd(f, x, dx):
    """
    Calculate a numerical approximation of a function using second-order accuracy
    backward difference scheme.

    Parameters
    ----------
    f : callable object, function with unknown gradient
    x : float, point to evaluate the gradient
    dx : float, resolution of scheme.

    Returns
    -------
    Approximation gradient value of f at point x using 3th order accuracy..

    """
    return (3 * f(x) - 4 * f(x - dx) + f(x - 2 * dx)) / (2 * dx)

def grad_approx_high(N):
    """
    Calculate centered second order and compare it with exponential gradient
    Then, it will be shown in graph.

    Parameters
    ----------
    N : Integer, number of evenly spaced points.

    Returns
    -------
    Graph shows the difference result between approximation and exponential
    gradient.

    """
    dx = 1 / N
    x = np.linspace(0, 1, N+1)
    y = f(x)
    dydx = np.zeros(N+1)
    
    #calling forward and backward approximation function
    dydx[0] = backward_diff_2nd(f, x[0], dx)
    dydx[N] = forward_diff_2nd(f, x[N], dx)
    #calling centered approximation function for looping
    for j in range(1, N):
        dydx[j] = centered_diff_4th(f, x[j], dx) 
    return x, dydx, y
    
def order_diff_high():
    """
    Calculate the order accuracy of forward, centered, and backward difference

    Returns
    -------
    delta_x_mat : float, array, resolution of matrix.
    error_matrix : float, array, error number of numerical solution.

    """
    N_values = [2, 5, 10, 20, 50, 100]
    error = []
    delta_x = []

    for i, Ni in enumerate(N_values): 
        y_exact = analytic_sol(Ni)[1]
        y_approx = grad_approx_high(Ni)[1]
    
        forward_error = abs(y_exact[0] - y_approx[0])
        centered_error = abs(y_exact[1] - y_approx[1])
        backward_error = abs(y_exact[-1] - y_approx[-1])
        error.append([forward_error, centered_error, backward_error])
        delta_x.append(1 / Ni)
    
    error_matrix = np.asarray(error)
     
    plt.figure(figsize=(7, 7))
    plt.plot(delta_x, error_matrix[:, 0], 'c-', label = '2nd order Forward Difference')
    plt.plot(delta_x, error_matrix[:, 1], 'y-', label = '4th order Centered Difference')
    plt.plot(delta_x, error_matrix[:, 2], 'b-', label = '2nd order Backward Difference')
    plt.plot(delta_x, error_matrix, 'g^')
    plt.xlabel(r'$\Delta$x')
    plt.ylabel(r'$\epsilon$x')
    plt.grid(which='both')
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.show()
    return

order_diff_high()



