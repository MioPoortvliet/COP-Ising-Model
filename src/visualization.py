import matplotlib.pyplot as plt
from datetime import datetime
import unicodedata
import re
from src.IO_utils import ensure_dir


def slugify(value, allow_unicode=False):
	"""
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
	value = str(value)
	if allow_unicode:
		value = unicodedata.normalize('NFKC', value)
	else:
		value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
	value = re.sub(r'[^\w\s-]', '', value.lower())
	return re.sub(r'[-\s]+', '-', value).strip('-_')


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
