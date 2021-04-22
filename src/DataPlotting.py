# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:56:03 2021

@author: Jonah Post
"""
import numpy as np
import matplotlib.pyplot as plt
from src.IO_utils import load_json

def powerlaw(t,a,b) -> np.ndarray:
    """
    :type t: np.array
    :type a: float
    :type b: float
    """
    return a * t**b

def func_linear(x,a,b) -> np.ndarray:
    """
    :type x: np.array
    :type a: float
    :type b: float
    """
    return a + b*x

def exact_magnetization(temperature) -> np.ndarray:
    """
    :type temperature: np.ndarray
    """
    return (1. - (np.sinh(2./temperature))**-4 )**(1./8.)

mainpath="generated\\data\\"

# specify the folders containing the data to be plotted
folders = ["2021-04-22t182717027002\\", "2021-04-22t193509893215\\"]

# read in the data
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
plt.axis(ymin=0)
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
plt.axis()
plt.savefig("figures\\magnetic_susceptibility.png")

plt.figure()
plt.errorbar(x=temperature, y=specific_heat, yerr= error_specific_heat, marker=m, linestyle=ls, capsize=cs, label="MC Ising Model")
plt.axvline(2.269, color="red", ls="dashed", label=r"Theoretical $T_C$=2.269 ")
plt.xlabel(r"Temperature $T$")
plt.ylabel(r"Specific heat per spin $C$")
plt.legend()
plt.axis(ymax=2)
plt.savefig("figures\\specific_heat.png")

plt.show()