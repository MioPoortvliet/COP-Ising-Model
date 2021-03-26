from src.MetropolisAlgorithm import MetropolisAlgorithm
from src.physics import IsingModel
from src.visualization import plot_grid

if __name__ == "__main__":
	settings = {"size":50, "dimensions":2, "initial_distribution":[0.5, 0.5]}

	im = IsingModel(temperature=300)
	distributions = {"probability":im.distributions}

	mc = MetropolisAlgorithm(distributions, settings)
	plot_grid(mc.grid)
