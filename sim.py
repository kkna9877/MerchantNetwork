import numpy as np

from merchant import Merchant, Project




merchants = np.empty(shape=(Merchant.number,), dtype=object)
economy_size = 5.  #Needs working on


#Create the merchants
for i in range(Merchant.number):
	merchants[i] = Merchant(i)


shortest_paths = Merchant.ShortestPaths(merchants)


projects = np.empty(shape=(Project.number,), dtype=object)
thetas = np.random.uniform(Project.theta_min, Project.theta_max, Project.number)

#Associate a project with a merchant to get the skill component.
# Might be different number of projects to merchants so assign cyclically (they are all random anyway)
merch_id = 0
for i in range(Project.number):

	if merch_id == Merchant.number:
		merch_id = 0

	k = (np.random.beta(2, 2) * economy_size ) + merchants[merch_id].status  # This will need developing to account for growth etc

	projects[i] = Project(i, k,thetas[i],merch_id)
	Project.Print(projects[i])
	merchants[merch_id].projects.append(i)


	if (merchants[merch_id].cash - merchants[merch_id].reserve) > projects[i].expectation:
		Project.AllocateFunds(merchants[merch_id], merch_id, projects[i].expectation, projects[i], i)
		print(f'Project {i} self funded by {merch_id}')
		projects[i].funded = True


	merch_id += 1

#Sort un-funded projects according to mu/sigma=sqrt(k)
unfunded_list = Project.Unfunded(projects)

Project.Fund(unfunded_list,projects,merchants )


#print(f'\n\nOutput')
#for i in range(Project.number):
#	Project.Print(projects[i])












