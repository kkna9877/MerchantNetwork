import numpy as np

class Merchant:
	#Class variables, the count (id) of the merchants and the number of living merchants: count ge number

	count: int = 0
	number: int = 0

	#Common initialisations
	initial_wealth = 100.
	initial_no_connections: int = 4
	initial_skill = 0.
	#Initial seperation of merchants
	initial_distance: float = 10.

	def __init__(self):
		Merchant.count += 1
		Merchant.number += 1

		self.id: int = Merchant.count
		self.skill = Merchant.initial_skill
		self.connections = []
		self.projects = []


class Project:
	# Class variables, the count gives the project id
	count: int = 0

	def __init__(self, k, theta):
		if (k>0) and (theta>0):
			Project.count += 1
			self.id: int = Project.count
			self.k: float = k
			self.k: float = theta
			self.expectation: float = k * theta
			self.variance: float = theta * self.expectation
			self.payoff: float = float(np.random.gamma(k, theta, 1))
		else:
			raise ValueError(f'Project initialisation: k ({k}) and theta ({theta}) must be positive')