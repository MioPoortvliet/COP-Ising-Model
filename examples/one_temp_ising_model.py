from src.MonteCarlo import MonteCarlo
from src.physics import IsingModel

if __name__ == "__main__":
	settings = {"size":50, "dimensions":2}
	distributions = IsingModel(temperature=300)

	mc = MonteCarlo(distributions, settings)