from src.analysis import tau_in_temp_range, find_tau
from src.MetropolisAlgorithm import MetropolisAlgorithm
from src.physics import IsingModel, magnetization
from src.analysis import equilibrilize, calc_chi
from src.visualization import plot_time_trace
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt



if __name__ == "__main__":
	settings = {"size": 50, "dimensions": 2, "initial_distribution": 0.9, "N_tau":2, "tau_sweeps":30}
	# Set up settings
	sweeps = 40

	# Set up model and property functions to be calculated
	im = IsingModel(dimensionless_temperature=1., dims=settings["dimensions"])
	properties = (magnetization,)

	# Initialize monte carlo engine
	mc = MetropolisAlgorithm(model= im, property_functions=properties, settings=settings)
	equilibrilize(mc, settings)

	print("Determining correlation time")
	# If the sweeps are less than the correlation time this is not a good result anyway!
	saved_properties = mc.run_steps(settings["size"] ** settings["dimensions"] * settings["tau_sweeps"])

	# Retrive the magnetization
	magnetization_array = saved_properties[::, 1]

	# Correlation function
	chi = calc_chi(magnetization_array / settings["size"] ** settings["dimensions"])
	chi = chi/chi[0]
	def expo(x, a): return np.exp(-a*(x))
	range = np.arange(np.argwhere(chi<0)[0,0])
	popt, pcov = curve_fit(expo, range, chi[:range.size])
	plt.plot(chi[:range.size*3])
	plt.plot(expo(range, *popt))
	plt.ylabel("$\\chi(t)$")
	plt.xlabel("Steps $t$")
	plt.show()