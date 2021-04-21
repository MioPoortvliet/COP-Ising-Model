import numpy as np


def magnetic_susceptibility_per_beta(magnetization_per_spin):
	return np.mean(magnetization_per_spin**2)-np.mean(magnetization_per_spin)**2


def specific_heat(energy_per_spin, temp):
	return 1/(temp**2)*(np.mean(energy_per_spin ** 2) - np.mean(energy_per_spin) ** 2)


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
