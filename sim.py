import numpy as np

from merchant import Merchant, Project

no_merchants: int = 5
no_projs: int = 5

merchants = np.array(no_merchants)

#Create the merchants
for i in range(no_merchants):
	merchants[i] = Merchant()

#Create the connections
for i in range(no_merchants):
	no_connections = Merchant.initial_no_connections
	lower = i - int(np.floor(no_connections/2))
	upper = i + int(np.ceil(no_connections/2))

	for j in range(lower, upper):
		# Connect from here to there
		existing_connection = Merchant.Connect(merchants[i], merchants[j], Merchant.initial_distance)
		existing_connection = Merchant.Connect(merchants[j], merchants[i], Merchant.initial_distance)


project_list = []
thetas = np.random.uniform(Project.theta_min, Project.theta_max, no_projs)

#Associate a project with a merchant to get the skill component.
# Might be different number of projects to merchants so assign them randomly (but in sequence)
merch_id = np.random.randint(no_merchants)
for i in range(no_projs):

	if merch_id == no_merchants:
		merch_id = 0

	k = np.random.beta(2, 2)+merchants[merch_id].skill  # This will need developing to account for growth etc

	project_list.append(Project(k, thetas[i]))









