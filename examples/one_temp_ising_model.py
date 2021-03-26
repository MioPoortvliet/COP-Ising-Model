from src.MetropolisAlgorithm import MetropolisAlgorithm
from src.physics import IsingModel

if __name__ == "__main__":
	settings = {"size":50, "dimensions":2}
	probability_distribution = IsingModel(temperature=300)

	mc = MetropolisAlgorithm(probability_distribution, settings)