from src.MetropolisAlgorithm import MetropolisAlgorithm
from src.physics import IsingModel, magnetization
from src.visualization import plot_grid

if __name__ == "__main__":
	settings = {"size":50, "dimensions":2, "initial_distribution":0.5}

	im = IsingModel(temperature=0.1, dims=settings["dimensions"])
	properties = (magnetization,)

	mc = MetropolisAlgorithm(model= im, property_functions=properties, settings=settings)
	plot_grid(mc.state)
	mc.run_steps(settings["size"]**2*20)
	plot_grid(mc.state)
	print(magnetization(mc.state))
