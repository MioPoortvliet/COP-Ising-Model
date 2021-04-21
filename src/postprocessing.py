import numpy as np


def standard_deviation_of_the_mean(series, tau, tmax):
	return np.sqrt(2*tau / tmax * (np.mean(series**2) - np.mean(series)**2))


def magnetic_susceptibility_per_beta(magnetization, spins):
	return 1/spins * (np.mean(magnetization**2)-np.mean(magnetization)**2)


def specific_heat(energy, temp, spins):
	return 1/(temp**2 * spins)*(np.mean(energy ** 2) - np.mean(energy) ** 2)


def calc_chi(magnetization):
	tmax = magnetization.size
	chi = np.arange(magnetization.size, dtype=np.float_)
	for time in range(chi.size):
		chi[time] = (np.mean(magnetization[:tmax - time] * magnetization[time:tmax])
		   - np.mean(magnetization[:tmax - time]) * np.mean(magnetization[time:tmax]))
	return chi


def calc_chi_old(mag):
	chi = np.arange(mag.size, dtype=np.float_)
	for time in range(chi.size):
		chi[time] = calc_chi_t_old(mag, time)
	return chi


def calc_chi_t_old(magnetization, time):
	tmax = magnetization.size

	chi = (np.mean(magnetization[:tmax - time] * magnetization[time:tmax])
		   - np.mean(magnetization[:tmax - time]) * np.mean(magnetization[time:tmax]))
	return chi
