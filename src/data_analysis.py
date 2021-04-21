# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:56:03 2021

@author: Jonah Post
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# def generate_data(reduced_temp, a, b):
#     data= powerlaw(reduced_temp, a, b)
#     noise = 0.2 * np.random.normal(size=data.size)
#     return(data+noise)

# Tc = 2.2
# T = np.arange(1,2,.2)
# reduced_temp = np.abs(T-Tc)
# print(reduced_temp)

def powerlaw(t,a,b):
    return a * t**b

def func_linear(x,a,b):
    return a + b*x

def find_critical_exponents(reduced_temperature, quantity_data, error_quantity_data):
    "Linear log-log fit to the given quantity. Used to obtain critical exponents"
    xdata  = np.log(reduced_temperature)
    ydata  = np.log(quantity_data)
    yerror = np.log(error_quantity_data)
    popt, pcov = curve_fit(func_linear, xdata, ydata, sigma=yerror, bounds=([-np.inf,-5.],[np.inf,5.]))
    perr = np.sqrt(np.diag(pcov))
    print(popt, perr)
    print("critical exponent = ",popt[1], "+-",perr[1])
    plt.figure()
    plt.errorbar(x=reduced_temperature, y=quantity_data, yerr=error_quantity_data, marker=".",linestyle="")
    plt.plot(reduced_temperature, np.exp(func_linear(np.log(reduced_temperature), *popt)))
    plt.plot(reduced_temperature, np.exp(func_linear(np.log(reduced_temperature), popt[0]+perr[0], popt[1]-perr[1])), linestyle='dashed', color='red')
    plt.plot(reduced_temperature, np.exp(func_linear(np.log(reduced_temperature), popt[0]-perr[0], popt[1]+perr[1])), linestyle='dashed', color='red')
    plt.show()
    return popt[1], perr[1]

def exact_magnetization(temperature):
    return (1. - (np.sinh(2./temperature))**-4 )**(1./8.)

filename_chi= "chit.npy"
chi, error_chi = np.load(filename_chi)[:,0],np.load(filename_chi)[:,1]

plt.figure()
plt.plot(chi)
plt.plot(error_chi)
plt.show()


