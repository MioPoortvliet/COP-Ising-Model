from src.MetropolisAlgorithm import MetropolisAlgorithm
from src.physics import IsingModel, magnetization
from src.visualization import plot_grid, plot_time_trace, plot_xy
from src.postprocessing import calc_chi, magnetic_susceptibility_per_beta, specific_heat, standard_deviation_of_the_mean
import numpy as np


def equilibrilize(mc, settings):
	# Check if settings contains any of these keys
	if "treshold" in settings.keys():
		treshold = settings["treshold"]
	else:
		treshold = 5e-7

	if "max_sweeps" in settings.keys():
		max_sweeps = settings["max_sweeps"]
	else:
		max_sweeps = 100

	if "sweep_length" in settings.keys():
		sweep_length = settings["sweep_length"]
	else:
		sweep_length = 5

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
			print(f"Presumably in equilibrium after {sweep+1} sweeps")
			break

		# At some point we give up, let the user know.
		elif sweep+1 == max_sweeps:
			print(f"Could not find equilibrium after {sweep+1} sweeps!")
			plot = True

	# To check if it makes sense, you can plot the magnetization
	if plot:
		plot_time_trace(mag / mc.total_spins, ylabel="Magnetization $m$", ylims=(-1, 1))


def full_analysis_in_temp_range(temps, settings):
	"""This is so chaotic, I hate this code."""
	taus = np.zeros((temps.size, settings["N_tau"]))
	absolute_magnetization_at_temp = np.zeros((temps.size, 2))
	magnetic_susceptibility_at_temp = np.zeros_like(absolute_magnetization_at_temp)
	specific_heat_at_temp = np.zeros_like(absolute_magnetization_at_temp)
	taus_at_temp = np.zeros_like(absolute_magnetization_at_temp)

	# Loop over every temperature to calculate tau
	for i in range(temps.size):
		# Create the ising model physics
		im = IsingModel(dimensionless_temperature=temps[i], dims=settings["dimensions"])
		properties = (magnetization,)
		# Feed ising model into metropolis algorithm
		mc = MetropolisAlgorithm(model=im, property_functions=properties, settings=settings)

		equilibrilize(mc, settings)

		# Determine tau
		for j in range(settings["N_tau"]):
			taus[i, j] = find_tau(mc, settings=settings, sweeps=settings["tau_sweeps"])
		tau = np.mean(taus[i, ::])

		delta_t = int(16*tau)

		# Create a bunch of arrays to store crap
		magnetic_susceptibility_block = np.zeros(settings["max_blocks"])
		specific_heat_block = np.zeros_like(magnetic_susceptibility_block)

		# This one is different from the other two
		total_magnetization_array = np.zeros(delta_t*settings["max_blocks"])

		for k in range(settings["max_blocks"]):
			# NOT IMPLEMENTED:
			# You can break the loop here if the error is small enough

			data = mc.run_steps(delta_t)
			energy_array = data[::,0]
			magnetization_array = data[::, 1]
			# Up to -1 bc we already calced step 0 in the previous block.
			# If we say to run x steps we expect x points back, not x+1.
			total_magnetization_array[k*delta_t:(1+k)*delta_t] = magnetization_array[:-1]

			# Quantity averages (block)
			magnetic_susceptibility_block[k] = magnetic_susceptibility_per_beta(magnetization=magnetization_array, spins=mc.total_spins)
			specific_heat_block[k] = specific_heat(energy=energy_array, spins=mc.total_spins, temp=temps[i])

		# Quantity averages
		# we explicitly use k in case we decide to stop the loop (NOT IMPLEMENTED)
		absolute_magnetization_at_temp[i, 0] = np.mean( np.abs(total_magnetization_array[:(k+1)*delta_t]) ) / mc.total_spins
		magnetic_susceptibility_at_temp[i, 0] = np.mean(magnetic_susceptibility_block[:k])
		specific_heat_at_temp[i, 0] = np.mean(specific_heat_block[:k])
		taus_at_temp[i, 0] = np.mean(taus[i, ::])

		# Quantity means
		absolute_magnetization_at_temp[i, 1] = standard_deviation_of_the_mean( np.abs(total_magnetization_array[:(k+1)*delta_t]), tau=tau, tmax=delta_t ) / mc.total_spins
		magnetic_susceptibility_at_temp[i, 1] = np.std(magnetic_susceptibility_block[:k], ddof=1)
		specific_heat_at_temp[i, 1] = np.std(specific_heat_block[:k], ddof=1)
		taus_at_temp[i, 1] = np.std(taus[i, ::], ddof=1)

	return taus_at_temp, absolute_magnetization_at_temp, magnetic_susceptibility_at_temp, specific_heat_at_temp





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
	print("Determining correlation time...")
	# If the sweeps are less than the correlation time this is not a good result anyway!
	saved_properties = mc.run_steps(settings["size"]**settings["dimensions"]*sweeps)

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