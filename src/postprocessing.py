import numpy as np


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
