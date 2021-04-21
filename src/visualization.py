import matplotlib.pyplot as plt
from datetime import datetime
from src.IO_utils import ensure_dir, slugify
from numpy import array


def plot_grid(grid:array) -> None:
	"""Make a simple plot of the grid (needs to be 2D)"""
	plt.figure()
	plt.imshow(grid)
	plt.show()


def plot_time_trace(time_series:array, ylabel="", ylims=None) -> None:
	"""Plot a variable as a time trace."""
	plt.figure()
	plt.plot(time_series)
	plt.xlabel("Steps")
	plt.ylabel(ylabel)
	plt.ylim(ylims)

	plt.show()


def plot_xy(x:array, y:array, xlabel:str, ylabel:str, dir="generated/plot_xy", *args, **kwargs) -> None:
	"""Make a plot of x against y. Pass an xlabel and an ylabel. It saves to dir.
	Args and kwargs are passed to plt.errorbar ."""
	plt.errorbar(x, y, *args, **kwargs)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)

	ensure_dir(dir)
	plt.savefig(f"{dir}/{slugify(xlabel)}-{slugify(ylabel)}-{slugify(datetime.now().isoformat())}.pdf")
	plt.savefig(f"{dir}/{slugify(xlabel)}-{slugify(ylabel)}-{slugify(datetime.now().isoformat())}.png")

	plt.show()
