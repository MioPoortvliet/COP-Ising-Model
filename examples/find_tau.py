from src.analysis import tau_in_temp_range
import numpy as np



if __name__ == "__main__":
	settings = {"size": 50, "dimensions": 2, "initial_distribution": 0.9, "N_tau":2, "tau_sweeps":10}
	temps = np.arange(1, 4.1, 2)
	tau_in_temp_range(temps, settings)