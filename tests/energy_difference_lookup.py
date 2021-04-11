from src.physics import *
from src.MetropolisAlgorithm import *
from src.visualization import plot_grid

settings = {"size": 50, "dimensions": 2, "initial_distribution": 0.5}

im = IsingModel(temperature=0.1, dims=settings["dimensions"])
properties = (magnetization,)

mc = MetropolisAlgorithm(model=im, property_functions=properties, settings=settings)

index = tuple(np.array([mc.state[(20,20)], *im.get_neighbour_spins((20,20), mc.state)]).astype(int))
print(index)
print(mc.state[(20,20)], im.get_neighbour_spins((20,20), mc.state))
print("=========energy_difference_nearest_neighbours===========")
print(im.energy_difference_nearest_neighbours(mc.state[(20,20)], im.get_neighbour_spins((20,20), mc.state)))
print("=========energy_difference_nn_lookup===========")
print(im.energy_difference_nn_lookup[index])
print("=========energy_difference_old===========")

print(im.energy_difference_old((20,20), mc.state))

plot_grid(mc.state[19:22, 19:22])
print(im.get_neighbour_spins((20,20), mc.state))