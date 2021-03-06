from src.analysis import full_analysis_in_temp_range
from src.visualization import plot_xy
from src.IO_utils import to_file, ensure_dir, slugify, to_json
import numpy as np
from datetime import datetime

settings = {
	"size": 50,						# Size along axis of grid
	"dimensions": 2,				# Dimensions of the grid
	"initial_distribution": 0.75,	# Chance to have spin down in initialization
	"tmin": 1,						# Minimum temperature
	"tmax": 4,						# Maximum temperature
	"t_step_size": 0.2,				# Stepsize of temperature
	"equilibrize_sweep_length": 10,	# Sublength after which it checks if it is in equilibrium
	"tau_sweeps": 20,				# Length over which correlation function is calculated (does not scale well!)
	"N_tau": 15,					# Number of samples to determine correlation length
	"max_blocks": 25,				# Maximum number of blocks of 16tau to calculate the properties over
	"treshold": 5e-8,				# Flatness of slope after which we call it equillibrilized
	"max_sweeps": 100,				# Maximum sweeps after which to cancell atempt to equillibrilize
	"sweep_length": 5,				# Sweep length of equilibrilization process before checking slope
	"plot": 0						# Plot magnetization before continuing? (1 or 0)
}

critical_temp_settings = {
	"size": 50,						# Size along axis of grid
	"dimensions": 2,				# Dimensions of the grid
	"initial_distribution": 0.75,	# Chance to have spin down in initialization
	"tmin": 2,						# Minimum temperature
	"tmax": 2.5,						# Maximum temperature
	"t_step_size": 0.05,				# Stepsize of temperature
	"equilibrize_sweep_length": 10,	# Sublength after which it checks if it is in equilibrium
	"tau_sweeps": 50,				# Length over which correlation function is calculated (does not scale well!)
	"N_tau": 5,					# Number of samples to determine correlation length
	"max_blocks": 25,				# Maximum number of blocks of 16tau to calculate the properties over
	"treshold": 5e-8,				# Flatness of slope after which we call it equillibrilized
	"max_sweeps": 100,				# Maximum sweeps after which to cancell atempt to equillibrilize
	"sweep_length": 10,				# Sweep length of equilibrilization process before checking slope
	"plot": 0						# Plot magnetization before continuing? (1 or 0)
}

quick_settings = {
	"size": 50,
	"dimensions": 2,
	"initial_distribution": 0.75,
	"tmin": 1,
	"tmax": 4,
	"t_step_size": 1,
	"equilibrize_sweep_length": 5,
	"tau_sweeps": 10,
	"N_tau": 2,
	"max_blocks": 5,
	"treshold": 5e-6,
	"max_sweeps": 50,
	"sweep_length": 5,
	"plot": 0
}

quick_settings_3d = {
	"size": 20,
	"dimensions": 3,
	"initial_distribution": 0.75,
	"tmin": 1,
	"tmax": 4,
	"t_step_size": 1,
	"equilibrize_sweep_length": 5,
	"tau_sweeps": 10,
	"N_tau": 2,
	"max_blocks": 5,
	"treshold": 5e-6,
	"max_sweeps": 50,
	"sweep_length": 5,
	"plot": 0
}

def main(settings:dict, root_path:str) -> None:
	"""
	Plot correlation time, absolute magnetization, magnetic susceptibility and heat capacity given settings.
	:param settings: contains at least: size, dimensions, initial_distribution, tmin, tmax, t_step_size,
										equilibrize_sweep_length, tau_sweeps, N_tau, max_blocks
	:type settings: dict
	:param fpath: path to write data and plots to
	:type fpath: str
	"""
	fpath = f"{root_path}/{slugify(datetime.now().isoformat())}/"
	ensure_dir(fpath)
	# Save settings
	to_json(fpath + "settings.json", settings)

	# Create temperature array, ensure we include the last point in the range
	temps = np.arange(settings["tmin"], settings["tmax"]+settings["t_step_size"]/2, settings["t_step_size"])

	calced_values = full_analysis_in_temp_range(temps, settings)

	value_names = ("$\\tau(T)$", "$|m(T)|$", "$e(T)$", "chi$(T)$", "$C(T)$")

	# Save and plot
	for value, name in zip(calced_values, value_names):
		# Users are stupid and might have removed the dir in the time the code ran.
		ensure_dir(fpath)

		to_file(fpath + slugify(name), value)
		plot_xy(x=temps, y=value[::, 0], ylabel=name, xlabel="Dimensionless Temperature $T$", yerr=value[::, 1],
				dir=fpath + "img/")


if __name__ == "__main__":
	# What settings? Uncomment for testing purposes:
	#settings  = critical_temp_settings
	#settings = quick_settings_3d	# Poor statistics but it will take too long otherwise.
	settings = quick_settings	# Poor statistics but it will take too long otherwise.

	fpath = f"generated/data"
	main(settings, fpath)
