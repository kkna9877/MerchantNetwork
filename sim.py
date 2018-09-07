import numpy as np

from merchant import Merchant, Project




merchants = np.empty(shape=(Merchant.number,), dtype=object)
economy_size = 5.  #Needs working on

print(merchants)
#Create the merchants
for i in range(Merchant.number):
	merchants[i] = Merchant(i)

adj_matrix=Matrix.ConnectionsM(merchants)

print(adj_matrix)

projects = np.empty(shape=(Project.number,), dtype=object)
thetas = np.random.uniform(Project.theta_min, Project.theta_max, Project.number)

#Associate a project with a merchant to get the skill component.
# Might be different number of projects to merchants so assign cyclically (they are all random anyway)
merch_id = 0
for i in range(Project.number):

	if merch_id == no_merchants:
		merch_id = 0

	k = (np.random.beta(2, 2) * economy_size ) + merchants[merch_id].status  # This will need developing to account for growth etc

	projects[i] = Project(k,theta[i],merch_id)
	merchants[merch_id].projects.append(i)


	if merchants[merch_id].cash > projects[i].expectation:
		merchants[merch_id].cash = merchants[merch_id].cash - projects[i].expectation
		projects[i].investors[merch_id] = projects[i].expectation
		projects[i].funded = True













