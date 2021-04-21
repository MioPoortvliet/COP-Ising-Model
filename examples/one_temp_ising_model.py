from src.MetropolisAlgorithm import MetropolisAlgorithm
from src.physics import IsingModel, magnetization
from src.visualization import plot_grid, plot_time_trace


if __name__ == "__main__":
	# Set up settings
	sweeps = 20
	settings = {"size":50, "dimensions":2, "initial_distribution":0.5}

	# Set up model and property functions to be calculated
	im = IsingModel(dimensionless_temperature=1, dims=settings["dimensions"])
	properties = (magnetization,)

	# Initialize monte carlo engine
	mc = MetropolisAlgorithm(model= im, property_functions=properties, settings=settings)

	# Plot initial grid
	#plot_grid(mc.state)

	# Run simulation
	saved_properties = mc.run_steps(settings["size"]**settings["dimensions"]*sweeps)

	# Plots after simulating for a little
	plot_grid(mc.state[::,::])
	plot_time_trace(saved_properties[::,0]/mc.total_spins, ylabel="Energy $e$")
	plot_time_trace(saved_properties[::,1]/mc.total_spins, ylabel="Magnetization $m$", ylims=(-1, 1))
