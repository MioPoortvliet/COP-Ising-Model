import matplotlib.pyplot as plt


def plot_grid(grid):
	plt.imshow(grid)
	plt.show()

def plot_magnetization(time_series):
	plt.plot(time_series)
	plt.xlabel("Steps")
	plt.ylabel("Magnetization $m$")
	plt.ylim(-1, 1)
	plt.show()
