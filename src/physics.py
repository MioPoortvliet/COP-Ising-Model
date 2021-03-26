import numpy as np


class IsingModel\
:
	def __init__(self, temperature=300):
		self.temperature = temperature
		self.distributions = None