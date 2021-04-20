import matplotlib.pyplot as plt


def plot_grid(grid):
	plt.imshow(grid)
	plt.show()

def plot_time_trace(time_series, ylabel="", ylims=None):
	plt.plot(time_series)
	plt.xlabel("Steps")
	plt.ylabel(ylabel)
	plt.ylim(ylims)
	plt.show()
