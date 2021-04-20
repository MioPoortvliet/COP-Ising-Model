from src.MetropolisAlgorithm import MetropolisAlgorithm
from src.physics import IsingModel, magnetization
from src.visualization import plot_grid, plot_time_trace

if __name__ == "__main__":
	settings = {"size":50, "dimensions":2, "initial_distribution":0.5}

	im = IsingModel(temperature=1, dims=settings["dimensions"])
	properties = (magnetization,)

	mc = MetropolisAlgorithm(model= im, property_functions=properties, settings=settings)
	#plot_grid(mc.state)
	saved_properties = mc.run_steps(settings["size"]**settings["dimensions"]*500)
	plot_grid(mc.state[::,::])
	plot_time_trace(saved_properties[::,0]/mc.total_spins, ylabel="Energy $e$")
	plot_time_trace(saved_properties[::,1]/mc.total_spins, ylabel="Magnetization $m$", ylims=(-1, 1))
