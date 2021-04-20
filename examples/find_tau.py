from src.MetropolisAlgorithm import MetropolisAlgorithm
from src.physics import IsingModel, magnetization
from src.visualization import plot_grid, plot_time_trace
from src.postprocessing import calc_chi
import numpy as np


def run_multiple(N=5):
	tau = np.zeros(N)
	for n in range(N):
		tau[n] = find_tau(sweeps=100)

	print(tau)
	print(np.mean(tau), np.std(tau, ddof=1))
	plot_time_trace(tau, "$\\tau$")


def find_tau(sweeps=100):
	settings = {"size":50, "dimensions":2, "initial_distribution":0.5}

	im = IsingModel(temperature=0.01, dims=settings["dimensions"])
	properties = (magnetization,)

	mc = MetropolisAlgorithm(model= im, property_functions=properties, settings=settings)

	tau_flag = True
	mag = np.array([])

	while tau_flag:
		saved_properties = mc.run_steps(settings["size"]**settings["dimensions"]*sweeps)
		print("Done with monte carlo, postprocessing")
		mag = np.concatenate((mag, saved_properties[::,1]))
		chi = calc_chi(mag/settings["size"]**settings["dimensions"])
		first_negative_index = np.argwhere(chi < 0)
		if first_negative_index.size > 0:
			tau_flag = False
			tau = np.sum(chi[:first_negative_index[0,0]])


		plot_time_trace(chi/chi[0], ylabel="$Chi(t)$")

	print(tau)
	return tau
	#plot_time_trace(np.cumsum(chi)/chi[0], ylabel="$tau$")
	#plot_grid(mc.state[::,::])
	#plot_time_trace(energy/mc.total_spins, ylabel="Energy $e$")
	#plot_time_trace(mag/mc.total_spins, ylabel="Magnetization $m$", ylims=(-1, 1))




if __name__ == "__main__":
	run_multiple()