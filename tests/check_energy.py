from src.MetropolisAlgorithm import MetropolisAlgorithm
from src.physics import IsingModel, magnetization
from src.visualization import plot_grid, plot_time_trace
from src.analysis import equilibrilize


if __name__ == "__main__":
	# Set up settings
	sweeps = 40
	settings = {"size":50, "dimensions":2, "initial_distribution":0.25}

	# Set up model and property functions to be calculated
	im = IsingModel(dimensionless_temperature=1., dims=settings["dimensions"])
	properties = (magnetization,)

	# Initialize monte carlo engine
	mc = MetropolisAlgorithm(model= im, property_functions=properties, settings=settings)
	#equilibrilize(mc, settings)
	for i in range(10):
		mc.step()
		total_energy = im.hamiltonian(mc.state)
		plot_grid(mc.state)
		print(total_energy, mc.current_energy)
		assert total_energy == mc.current_energy

	# Plot initial grid
	plot_grid(mc.state)

	# Run simulation
	saved_properties = mc.run_steps(settings["size"]**settings["dimensions"]*sweeps)

	# Plots after simulating for a little
	plot_grid(mc.state[::,::])
	plot_time_trace(saved_properties[::,0]/mc.total_spins, ylabel="Energy $e$")
	plot_time_trace(saved_properties[::,1]/mc.total_spins, ylabel="Magnetization $m$", ylims=(-1, 1))
