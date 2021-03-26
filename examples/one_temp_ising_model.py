from src.MetropolisAlgorithm import MetropolisAlgorithm
from src.physics import IsingModel
from src.visualization import plot_grid

if __name__ == "__main__":
	settings = {"size":50, "dimensions":2, "initial_distribution":0.5}

	im = IsingModel(temperature=300)
	distribution = im.probability_distribution

	mc = MetropolisAlgorithm(distribution, settings)
	plot_grid(mc.state)
