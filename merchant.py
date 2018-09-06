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

	def Connect_Merchant(merchant_obj_from, merchant_obj_to , distance):
		# i and j should be objects
		#Allow for asymmetric connections (i.e. should be called Connect_Merchant(merchant_obj_2,merchant_obj_1, distance)

		#check if merchant_obj_to.id is in merchant_obj_from's connections
		merchant_found = False
		if len(merchant_obj_from.connections)>0:
			for i in range(len(merchant_obj_from.connections)):
				if (merchant_obj_from.connections[i][0] == merchant_obj_to.id):
					# merchant_obj_2 is here, update the distance
					merchant_obj_from.connections[i][1] = distance
					merchant_found = True

		if not merchant_found:
			merchant_obj_from.connections.append([merchant_obj_to.id,distance])

		return merchant_found





class Project:
	# Class variables, the count gives the project id
	count: int = 0

	theta_min: float = 0.5
	theta_max: float = 2.0

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