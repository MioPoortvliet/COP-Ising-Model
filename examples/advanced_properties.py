from src.analysis import full_analysis_in_temp_range
from src.visualization import plot_xy
from src.IO_utils import to_file, ensure_dir, slugify, to_json
import numpy as np
from datetime import datetime



if __name__ == "__main__":
	settings = {
		"size": 50,
		"dimensions": 2,
		"initial_distribution": 0.75,
		"tmin":1,
		"tmax":4,
		"t_step_size":0.2,
		"equilibrize_sweep_length":5,
		"tau_sweeps":50,
		"N_tau":25,
		"max_blocks":50,
		"treshold":5e-7,
		"max_sweeps":100,
		"sweep_length":10,
		"plot":0
	}

	quick_settings = {
		"size": 50,
		"dimensions": 2,
		"initial_distribution": 0.75,
		"tmin":1,
		"tmax":4,
		"t_step_size":2,
		"equilibrize_sweep_length":5,
		"tau_sweeps":2,
		"N_tau":2,
		"max_blocks":2,
		"treshold":5e-5,
		"max_sweeps":1,
		"sweep_length":1,
		"plot":1
	}

	# Quick? Uncomment for quick.
	#settings = quick_settings

	fpath=f"generated/data/{slugify(datetime.now().isoformat())}/"
	ensure_dir(fpath)

	temps = np.arange(settings["tmin"], settings["tmax"], settings["t_step_size"])


	to_json(fpath+"settings.json", settings)
	calced_values = full_analysis_in_temp_range(temps, settings)


	value_names = ("$\\tau(T)$", "$|m(T)|$", "chi$(T)$", "$C(T)$")

	for value, name in zip(calced_values, value_names):
		plot_xy(x=temps, y=value[::,0], ylabel=name, xlabel="Dimensionless Temperature $T$", yerr=value[::,1], dir=fpath+"img/")
		to_file(fpath+slugify(name), value)
