from src.analysis import tau_in_temp_range
import numpy as np



if __name__ == "__main__":
	settings = {"size": 50, "dimensions": 2, "initial_distribution": 0.9}
	temps = np.arange(1, 4, 0.2)
	#tau_in_temp_range(temps, settings, equilibrize_sweeps=1000, sweeps=30, N=5)
	tau_in_temp_range(temps, settings, equilibrize_sweeps=120, sweeps=25, N=25)
	#tau_in_temp_range(temps, settings, equilibrize_sweeps=100, sweeps=20, N=1)