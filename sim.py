import numpy as np
import pickle

from merchant import Merchant, Project




merchants = np.empty(shape=(Merchant.number,), dtype=object)
economy_size = 5.  #Needs working on


#Create the merchants
for i in range(Merchant.number):
	merchants[i] = Merchant(i)


shortest_paths = Merchant.ShortestPaths(merchants)


projects = np.empty(shape=(Project.number,), dtype=object)
thetas = np.random.uniform(Project.theta_min, Project.theta_max, Project.number)
ks= np.zeros(Project.number)

#Associate a project with a merchant to get the skill component.
# Might be different number of projects to merchants so assign cyclically (they are all random anyway)
merch_id = 0
for i in range(Project.number):

	if merch_id == Merchant.number:
		merch_id = 0

	ks[i] = (np.random.beta(2, 2) * economy_size ) + merchants[merch_id].status  # This will need developing to account for growth etc


	projects[i] = Project(i, ks[i],thetas[i],merch_id)
	merchants[merch_id].projects.append(i)


	if (merchants[merch_id].cash - merchants[merch_id].reserve) > projects[i].expectation:
		Project.AllocateFunds(merchants[merch_id], merch_id, projects[i].expectation, projects[i], i)
		print(f'Project {i} self funded by {merch_id}')

	Project.Print(projects[i])

	merch_id += 1

#Store these for debugging
pickle.dump(ks, open('k.p',"wb"))
pickle.dump(thetas, open('thetas.p',"wb"))

#Sort un-funded projects according to mu/sigma=sqrt(k)
unfunded_list = Project.Unfunded(projects)

Project.Fund(unfunded_list,projects,merchants )


print(f'\n\nOutput')
for i in range(Project.number):
	Project.Print(projects[i])












