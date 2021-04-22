"""
Functions that apply the classes
Authors: Mio Poortvliet, Jonah Post
"""
from src.MetropolisAlgorithm import MetropolisAlgorithm
from src.physics import IsingModel, magnetization
from src.visualization import plot_grid, plot_time_trace, plot_xy
from src.postprocessing import calc_chi, magnetic_susceptibility_per_beta, specific_heat, standard_deviation_of_the_mean
import numpy as np
from typing import Tuple
from warnings import warn


def equilibrilize(mc:object, settings:dict):
	"""Bring the system mc to an equilibrium within the parameters given in settings."""
	# Check if settings contains any of these keys
	if "treshold" in settings.keys():
		treshold = settings["treshold"]
	else:
		treshold = 5e-8

	if "max_sweeps" in settings.keys():
		max_sweeps = settings["max_sweeps"]
	else:
		max_sweeps = 100

	if "sweep_length" in settings.keys():
		sweep_length = settings["sweep_length"]
	else:
		sweep_length = 10

	if "plot" in settings.keys():
		plot = settings["plot"]
	else:
		plot = 5

	# Equilibrilize: run the simulation
	spins = settings["size"] ** settings["dimensions"]

	print("Finding equilibrium")
	for sweep in range(max_sweeps):
		mag = mc.run_steps( spins * sweep_length)[::, 1]
		n=1

		# We could look at the derivative of the series and when it is zero enough (treshold) stop it
		deriv = np.abs(np.mean((np.roll(a=mag, shift=n)[:] - mag[:])[n:])/spins)
		#print(deriv)

		# Check if we reach an 'end value' (derivative does not change)
		if deriv < treshold:
			print(f"Presumably in equilibrium after {sweep+1} sweeps.")
			break

		# At some point we give up, let the user know.
		elif sweep+1 == max_sweeps:
			warn(f"Could not find equilibrium after {sweep+1} sweeps!")
			plot = True

	# To check if it makes sense, you can plot the magnetization
	if plot:
		plot_time_trace(mag / mc.total_spins, ylabel="Magnetization $m$", ylims=(-1, 1))


def full_analysis_in_temp_range(temps:np.array, settings:dict)->Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Spaghetti panic-inducing code that does a full analysis of an Ising spin system. Given a range of temperatures
	and simulation settings it produces correlation times, absolute magnetization, magnetic susceptibility and
	specific heat at these temperatures.

	This is so chaotic, I hate this code.

	--------------Settings at least expected--------------
	"size": 					Size along axis of grid
	"dimensions": 				Dimensions of the grid
	"initial_distribution": 	Chance to have spin down in initialization
	"tmin": 					Minimum temperature
	"tmax": 					Maximum temperature
	"t_step_size": 				Stepsize of temperature
	"equilibrize_sweep_length":	Sublength after which it checks if it is in equilibrium
	"tau_sweeps": 				Length over which correlation function is calculated (does not scale well!)
	"N_tau": 					Number of samples to determine correlation length
	"max_blocks": 				Maximum number of blocks of 16tau to calculate the properties over
	--------------Settings further supported--------------
	"treshold": 				Flatness of slope after which we call it equillibrilized
	"max_sweeps": 				Maximum sweeps after which to cancell atempt to equillibrilize
	"sweep_length": 			Sweep length of equilibrilization process before checking slope
	"plot":						Plot magnetization before continuing? (1 or 0)
	"""

	taus = np.zeros((temps.size, settings["N_tau"]))
	absolute_magnetization_at_temp = np.zeros((temps.size, 2))
	magnetic_susceptibility_at_temp = np.zeros_like(absolute_magnetization_at_temp)
	energy_at_temp = np.zeros_like(absolute_magnetization_at_temp)
	specific_heat_at_temp = np.zeros_like(absolute_magnetization_at_temp)
	taus_at_temp = np.zeros_like(absolute_magnetization_at_temp)

	# Loop over every temperature to calculate tau
	for i in range(temps.size):
		print(f"Temperature: {temps[i]}\n"+"-"*50)
		# Create the ising model physics
		im = IsingModel(dimensionless_temperature=temps[i], dims=settings["dimensions"])
		properties = (magnetization,)
		# Feed ising model into metropolis algorithm
		mc = MetropolisAlgorithm(model=im, property_functions=properties, settings=settings)

		equilibrilize(mc, settings)

		# Determine tau
		for j in range(settings["N_tau"]):
			taus[i, j] = find_tau(mc, settings=settings)
		tau = np.mean(taus[i, ::])

		print(f"Beginning simulation of {settings['max_blocks']} blocks")
		delta_t = int(16*tau)

		# Create a bunch of arrays to store crap
		magnetic_susceptibility_block = np.zeros(settings["max_blocks"])
		specific_heat_block = np.zeros_like(magnetic_susceptibility_block)

		# This one is different from the other two
		total_magnetization_array = np.zeros(delta_t*settings["max_blocks"])
		total_energy_array = np.zeros_like(total_magnetization_array)

		for k in range(settings["max_blocks"]):
			# NOT IMPLEMENTED:
			# You can break the loop here if the error is small enough

			data = mc.run_steps(delta_t)
			energy_array = data[::,0]
			magnetization_array = data[::, 1]
			# Up to -1 bc we already calced step 0 in the previous block.
			# If we say to run x steps we expect x points back, not x+1.
			total_energy_array[k*delta_t:(1+k)*delta_t] = energy_array[:-1]
			total_magnetization_array[k*delta_t:(1+k)*delta_t] = magnetization_array[:-1]

			# Quantity averages (block)
			magnetic_susceptibility_block[k] = magnetic_susceptibility_per_beta(magnetization=magnetization_array, spins=mc.total_spins)
			specific_heat_block[k] = specific_heat(energy=energy_array, spins=mc.total_spins, temp=temps[i])

		# Quantity average
		# we explicitly use k in case we decide to stop the loop (which is NOT IMPLEMENTED)
		absolute_magnetization_at_temp[i, 0] = np.mean( np.abs(total_magnetization_array[:(k+1)*delta_t]) ) / mc.total_spins
		energy_at_temp[i, 0] = np.mean(total_energy_array[:(k+1)*delta_t]) / mc.total_spins
		magnetic_susceptibility_at_temp[i, 0] = np.mean(magnetic_susceptibility_block[:k])
		specific_heat_at_temp[i, 0] = np.mean(specific_heat_block[:k])
		taus_at_temp[i, 0] = np.mean(taus[i, ::])

		# Quantity mean
		absolute_magnetization_at_temp[i, 1] = standard_deviation_of_the_mean( np.abs(total_magnetization_array[:(k+1)*delta_t]), tau=tau, tmax=delta_t ) / mc.total_spins
		energy_at_temp[i, 1] = standard_deviation_of_the_mean(total_energy_array[:(k+1)*delta_t], tau=tau, tmax=delta_t) / mc.total_spins
		magnetic_susceptibility_at_temp[i, 1] = np.std(magnetic_susceptibility_block[:k], ddof=1)
		specific_heat_at_temp[i, 1] = np.std(specific_heat_block[:k], ddof=1)
		taus_at_temp[i, 1] = np.std(taus[i, ::], ddof=1)
		print()

	return taus_at_temp, absolute_magnetization_at_temp, energy_at_temp, magnetic_susceptibility_at_temp, specific_heat_at_temp





def tau_in_temp_range(temps:np.array, settings:dict, *args, **kwargs) -> np.array:
	"""Calculate tau in given temperature range and settings then plot it"""
	if "N_tau" not in settings.keys():
		settings["N_tau"] = 1
	taus = np.zeros((temps.size, settings["N_tau"]))

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

		taus[i] = run_multiple_find_tau(settings=settings, temperature=temps[i])

		print()

	# plot tau for T
	plot_xy(x=temps, y=np.mean(taus, axis=-1), ylabel="$\\tau$", xlabel="temperatures", yerr=np.std(taus, axis=-1, ddof=1))

	return taus


def run_multiple_find_tau(settings:dict,temperature=1.)->np.ndarray:
	"""Determines tau a few times."""
	# Create the Ising model physics
	im = IsingModel(dimensionless_temperature=temperature, dims=settings["dimensions"])
	properties = (magnetization,)
	# Feed ising model into metropolis algorithm
	mc = MetropolisAlgorithm(model= im, property_functions=properties, settings=settings)

	# Thermalize: run the simulation and retrieve the data
	equilibrilize(mc, settings)

	# Now we are thermalized, determine tau
	tau = np.zeros(settings["N_tau"])
	for n in range(settings["N_tau"]):
		tau[n] = find_tau(mc, settings)

	# Intended use: print(np.mean(tau), np.std(tau, ddof=1))

	return tau


def find_tau(mc:object, settings:dict) -> float:
	"""Determine tau given the simulation. mc is the simulation object, settings is the dict of settings.
	Returns the determined value of tau."""

	print("Determining correlation time")
	# If the sweeps are less than the correlation time this is not a good result anyway!
	saved_properties = mc.run_steps(settings["size"]**settings["dimensions"]*settings["tau_sweeps"])

	# Retrive the magnetization
	magnetization_array = saved_properties[::,1]

	# Correlation function
	chi = calc_chi(magnetization_array/settings["size"]**settings["dimensions"])

	# Find the index where chi dips below zero and calculate chi
	first_negative_index = np.argwhere(chi < 0)
	if first_negative_index.size > 0:
		tau = np.sum(chi[:first_negative_index[0,0]])/chi[0]

	else:
		tau = 0
		warn("Failed to determine tau!")

	return tau
