import matplotlib.pyplot as plt
from datetime import datetime
from src.IO_utils import ensure_dir, slugify


def plot_grid(grid):
	plt.imshow(grid)
	plt.show()


def plot_time_trace(time_series, ylabel="", ylims=None):
	plt.plot(time_series)
	plt.xlabel("Steps")
	plt.ylabel(ylabel)
	plt.ylim(ylims)
	plt.show()


def plot_xy(x, y, xlabel, ylabel, dir="generated/plot_xy", *args, **kwargs):
	plt.errorbar(x, y, *args, **kwargs)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	ensure_dir(dir)
	plt.savefig(f"{dir}/{slugify(xlabel)}-{slugify(ylabel)}-{slugify(datetime.now().isoformat())}.png")
	plt.show()
