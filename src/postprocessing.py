"""
To calculate statistical properties
Authors: Mio Poortvliet, Jonah Post
"""
import numpy as np


def standard_deviation_of_the_mean(series: np.array, tau: float, tmax: float) -> float:
	"""Does as it says on the tin"""
	return np.sqrt(2 * tau / tmax * (np.mean(series ** 2) - np.mean(series) ** 2))


def magnetic_susceptibility_per_beta(magnetization: np.ndarray, spins: int) -> float:
	"""Does as it says on the tin"""
	return 1 / spins * (np.mean(magnetization ** 2) - np.mean(magnetization) ** 2)


def specific_heat(energy: np.ndarray, temp: float, spins: int) -> float:
	"""Does as it says on the tin"""
	return 1 / (temp ** 2 * spins) * (np.mean(energy ** 2) - np.mean(energy) ** 2)


def calc_chi(magnetization: np.ndarray) -> np.ndarray:
	"""chi as a function of time. Needs a time trace of the magnetization. Careful, it computes the correlation function
	and is thus horribly slow."""

	tmax = magnetization.size
	chi = np.arange(magnetization.size, dtype=np.float_)

	for time in range(chi.size):
		chi[time] = (
				np.mean(magnetization[:tmax - time] * magnetization[time:tmax])
				- np.mean(magnetization[:tmax - time]) * np.mean(magnetization[time:tmax])
		)

	return chi
