from src.MetropolisAlgorithm import MetropolisAlgorithm
from src.physics import IsingModel, magnetization
from src.visualization import plot_grid, plot_time_trace
from src.postprocessing import calc_chi
import numpy as np


def tau_in_temp_range(temps, settings):


def run_multiple(
		N=5,
		equilibrize_sweeps=200,
		sweeps=40,
		settings={"size":50, "dimensions":2, "initial_distribution":0.5},
		temperature=1
):

	im = IsingModel(dimensionless_temperature=temperature, dims=settings["dimensions"])
	properties = (magnetization,)

	mc = MetropolisAlgorithm(model= im, property_functions=properties, settings=settings)

	mag = mc.run_steps(settings["size"] ** settings["dimensions"] * equilibrize_sweeps)[::,1]
	print("Presumably in equilibrium")
	plot_time_trace(mag/mc.total_spins, ylabel="Magnetization $m$", ylims=(-1, 1))


	tau = np.zeros(N)
	for n in range(N):
		tau[n] = find_tau(mc, settings, sweeps=sweeps)

	print(tau)
	print(np.mean(tau), np.std(tau, ddof=1))
	plot_time_trace(tau, "$\\tau$")


def find_tau(mc, settings, sweeps=20):
	saved_properties = mc.run_steps(settings["size"]**settings["dimensions"]*sweeps)
	print("Done with monte carlo, postprocessing")

	mag = saved_properties[::,1]
	chi = calc_chi(mag/settings["size"]**settings["dimensions"])
	first_negative_index = np.argwhere(chi < 0)
	if first_negative_index.size > 0:
		tau = np.sum(chi[:first_negative_index[0,0]])/chi[0]
	else:
		tau = 0
		print("Failed to determine tau!")

	plot_time_trace(chi/chi[0], ylabel="$Chi(t)$")

	print(tau)
	#plot_grid(mc.state[::,::])
	#plot_time_trace(mag/mc.total_spins, ylabel="Magnetization $m$", ylims=(-1, 1))
	return tau
	#plot_time_trace(np.cumsum(chi)/chi[0], ylabel="$tau$")
	#plot_time_trace(energy/mc.total_spins, ylabel="Energy $e$")




if __name__ == "__main__":
	#find_tau(10)
	#run_multiple(50)

	settings = {"size": 50, "dimensions": 2, "initial_distribution": 0.5}
	temps = np.arange(1, 5, 0.2)
	tau_in_temp_range(temps)