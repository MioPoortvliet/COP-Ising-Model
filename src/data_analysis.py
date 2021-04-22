# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:56:03 2021

@author: Jonah Post
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from src.IO_utils import load_json

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

def find_critical_exponents(temperature, quantity_data, error_quantity_data):
    Tc = 2.269
    mask = (temperature<Tc)
    temperature = temperature[mask]
    quantity_data = quantity_data[mask]
    error_quantity_data = error_quantity_data[mask]
    reduced_temperature = np.abs(temperature-Tc)
    "Linear log-log fit to the given quantity. Used to obtain critical exponents"
    xdata  = np.log(reduced_temperature)
    ydata  = np.log(quantity_data)
    yerror = np.log(error_quantity_data)
    popt, pcov = curve_fit(func_linear, xdata, ydata, sigma=yerror, bounds=([-np.inf,-5.],[np.inf,5.]))
    perr = np.sqrt(np.diag(pcov))
    # print(popt, perr)
    print("critical exponent = ",popt[1], "+-",perr[1])
    plt.figure()
    plt.errorbar(x=reduced_temperature, y=quantity_data, yerr=error_quantity_data, marker=".",linestyle="")
    plt.plot(reduced_temperature, np.exp(func_linear(np.log(reduced_temperature), *popt)))
    plt.plot(reduced_temperature, np.exp(func_linear(np.log(reduced_temperature), popt[0]+perr[0], popt[1]-perr[1])), linestyle='dashed', color='red')
    plt.plot(reduced_temperature, np.exp(func_linear(np.log(reduced_temperature), popt[0]-perr[0], popt[1]+perr[1])), linestyle='dashed', color='red')
    plt.xlabel(r"$|T-Tc|$")
    plt.show()
    return popt[1], perr[1]

def exact_magnetization(temperature):
    return (1. - (np.sinh(2./temperature))**-4 )**(1./8.)

mainpath="generated\\data\\"
folders = ["2021-04-22t005212292006\\", "2021-04-22t003241375617\\"]
temperature= np.array([])
tau= np.array([])
magnetization= np.array([])
energy= np.array([])
magnetic_susceptibility = np.array([])
specific_heat =np.array([])

error_temperature= np.array([])
error_tau= np.array([])
error_magnetization= np.array([])
error_energy= np.array([])
error_magnetic_susceptibility = np.array([])
error_specific_heat =np.array([])

for folder in folders:
    path = mainpath+folder
    filename_tau= "taut.npy"
    filename_m = "mt.npy"
    filename_e = "et.npy"
    filename_chi= "chit.npy"
    filename_c = "ct.npy"
    
    settings = load_json(path, "settings.json")
    temperature                                            = np.append(temperature,             np.arange(settings["tmin"], settings["tmax"] + settings["t_step_size"]/2., settings["t_step_size"]) )
    tau, error_tau                                         = np.append(tau,                     np.load(path+filename_tau)[:,0]) ,np.append(error_tau,                      np.load(path+filename_tau)[:,1])
    magnetization, error_magnetization                     = np.append(magnetization,           np.load(path+filename_m)[:,0])   ,np.append(error_magnetization,            np.load(path+filename_m)[:,1] )
    energy, error_energy                                   = np.append(energy,                  np.load(path+filename_e)[:,0])   ,np.append(error_energy,                   np.load(path+filename_e)[:,1])
    magnetic_susceptibility, error_magnetic_susceptibility = np.append(magnetic_susceptibility, np.load(path+filename_chi)[:,0]) ,np.append(error_magnetic_susceptibility,  np.load(path+filename_chi)[:,1])
    specific_heat, error_specific_heat                     = np.append(specific_heat,           np.load(path+filename_c)[:,0])   ,np.append(error_specific_heat,            np.load(path+filename_c)[:,1])

continuous_step_size = 0.001*settings["t_step_size"]
continuous_temperature = np.append(np.arange(settings["tmin"], 2.265, continuous_step_size), np.arange(2.265,2.27,0.00000001))

m="."
ls=""
cs=3

plt.figure()
plt.errorbar(x=temperature, y=tau, yerr= error_tau, marker=m, linestyle=ls, capsize=cs, label="MC Ising Model")
plt.axvline(2.269, color="red", ls="dashed", label=r"Theoretical $T_C$=2.269 ")
plt.xlabel(r"Temperature $T$")
plt.ylabel(r"Correlation time $\tau$")
plt.legend()
plt.savefig("figures\\correlation_time.png")

plt.figure()
plt.errorbar(x=temperature, y=magnetization, yerr= error_magnetization, marker=m, linestyle=ls, capsize=cs, label="MC Ising Model")
plt.plot(continuous_temperature, exact_magnetization(continuous_temperature), label="Exact solution")
plt.axvline(2.269, color="red", ls="dashed", label=r"Theoretical $T_C$=2.269 ")
plt.xlabel(r"Temperature $T$")
plt.ylabel(r"Magnetization per spin $m$")
plt.legend()
plt.axis(ymin=0)
plt.savefig("figures\\magnetization.png")

plt.figure()
plt.errorbar(x=temperature, y=energy, yerr= error_energy, marker=m, linestyle=ls, capsize=cs, label="MC Ising Model")
plt.axvline(2.269, color="red", ls="dashed", label=r"Theoretical $T_C$=2.269 ")
plt.xlabel(r"Temperature $T$")
plt.ylabel(r"Energy per spin $e$")
plt.legend()
plt.savefig("figures\\energy.png")

plt.figure()
plt.errorbar(x=temperature, y=magnetic_susceptibility, yerr= error_magnetic_susceptibility, marker=m, linestyle=ls, capsize=cs, label="MC Ising Model")
plt.axvline(2.269, color="red", ls="dashed", label=r"Theoretical $T_C$=2.269 ")
plt.xlabel(r"Temperature $T$")
plt.ylabel(r"Magnetic susceptibility per spin $\chi_M$")
plt.legend()
plt.axis(ymin=0)
plt.savefig("figures\\magnetic_susceptibility.png")

plt.figure()
plt.errorbar(x=temperature, y=specific_heat, yerr= error_specific_heat, marker=m, linestyle=ls, capsize=cs, label="MC Ising Model")
plt.axvline(2.269, color="red", ls="dashed", label=r"Theoretical $T_C$=2.269 ")
plt.xlabel(r"Temperature $T$")
plt.ylabel(r"Specific heat per spin $C$")
plt.legend()
plt.axis(ymin=0, ymax=2)
plt.savefig("figures\\specific_heat.png")

# find_critical_exponents(temperature, specific_heat, error_specific_heat)
# find_critical_exponents(temperature, magnetic_susceptibility, error_magnetic_susceptibility)
# find_critical_exponents(temperature, magnetization, error_magnetization)

plt.show()