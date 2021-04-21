from src.MetropolisAlgorithm import MetropolisAlgorithm
from src.physics import IsingModel, magnetization
from src.visualization import plot_grid, plot_time_trace, plot_xy
from src.postprocessing import calc_chi
import numpy as np


def tau_in_temp_range(temps, settings, N=5, critical_temperature = 2.2, *args, **kwargs):
	"""Calculate tau in given temperature range and plot it"""
	taus = np.zeros((temps.size, N))

	# Loop over every temperature to calculate tau
	for i in range(temps.size):
		"""
		# To help the thermalization process a little
		if temps[i] < critical_temperature:
			settings["initial_distribution"] = 0.9
		else:
			settings["initial_distribution"] = 0.5
		"""
		print(temps[i])

		taus[i] = run_multiple_find_tau(settings=settings, N=N, temperature=temps[i], *args, **kwargs)

		print()

	# plot tau for T
	plot_xy(x=temps, y=np.mean(taus, axis=-1), ylabel="$\\tau$", xlabel="temperatures", yerr=np.std(taus, axis=-1, ddof=1))

	return taus


def run_multiple_find_tau(
		N=5,
		equilibrize_sweeps=200,
		sweeps=40,
		settings={"size":50, "dimensions":2, "initial_distribution":0.5},
		temperature=1.
):

	# Create the ising model physics
	im = IsingModel(dimensionless_temperature=temperature, dims=settings["dimensions"])
	properties = (magnetization,)
	# Feed ising model into metropolis algorithm
	mc = MetropolisAlgorithm(model= im, property_functions=properties, settings=settings)

	# Thermalize: run the simulation and retrieve the data
	mag = mc.run_steps(settings["size"] ** settings["dimensions"] * equilibrize_sweeps)[::,1]
	print("Presumably in equilibrium")

	# To check if it makes sense, you can plot the magnetization
	plot_time_trace(mag/mc.total_spins, ylabel="Magnetization $m$", ylims=(-1, 1))

	# Now we are thermalized, determine tau
	tau = np.zeros(N)
	for n in range(N):
		tau[n] = find_tau(mc, settings, sweeps=sweeps)

	print(np.mean(tau), np.std(tau, ddof=1))
	#plot_time_trace(tau, "$\\tau$")
	return tau


def find_tau(mc, settings, sweeps=20):

	# If the sweeps are less than the correlation time this is not a good result anyway!
	saved_properties = mc.run_steps(settings["size"]**settings["dimensions"]*sweeps)
	print("Done with monte carlo, postprocessing")

	# Retrive the magnetization
	magnetization = saved_properties[::,1]

	# Correlation function
	chi = calc_chi(magnetization/settings["size"]**settings["dimensions"])

	# Find the index where chi dips below zero and calculate chi
	first_negative_index = np.argwhere(chi < 0)
	if first_negative_index.size > 0:
		tau = np.sum(chi[:first_negative_index[0,0]])/chi[0]
	else:
		tau = 0
		print("Failed to determine tau!")

	#plot_time_trace(chi/chi[0], ylabel="$Chi(t)$")

	#print(tau)
	#plot_grid(mc.state[::,::])
	#plot_time_trace(mag/mc.total_spins, ylabel="Magnetization $m$", ylims=(-1, 1))
	#plot_time_trace(np.cumsum(chi)/chi[0], ylabel="$tau$")
	#plot_time_trace(energy/mc.total_spins, ylabel="Energy $e$")

	return tau



if __name__ == "__main__":
	settings = {"size": 50, "dimensions": 2, "initial_distribution": 0.9}
	temps = np.arange(1, 4, 0.2)
	#tau_in_temp_range(temps, settings, equilibrize_sweeps=1000, sweeps=30, N=5)
	#tau_in_temp_range(temps, settings, equilibrize_sweeps=120, sweeps=25, N=25)
	tau_in_temp_range(temps, settings, equilibrize_sweeps=100, sweeps=20, N=1)