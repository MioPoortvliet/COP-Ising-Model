from src.MetropolisAlgorithm import MetropolisAlgorithm
from src.physics import IsingModel, magnetization
from src.visualization import plot_grid, plot_magnetization

if __name__ == "__main__":
	settings = {"size":50, "dimensions":2, "initial_distribution":0.5}

	im = IsingModel(temperature=0.01, dims=settings["dimensions"])
	properties = (magnetization,)

	mc = MetropolisAlgorithm(model= im, property_functions=properties, settings=settings)
	#plot_grid(mc.state)
	mc.run_steps(settings["size"]**2*100)
	plot_grid(mc.state)
	plot_magnetization(mc.saved_properties[::,0])
