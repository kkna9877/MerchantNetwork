import numpy as np


from merchant import Merchant, Project




merchants = np.empty(shape=(Merchant.number,), dtype=object)
  #Needs working on


#Create the merchants
for i in range(Merchant.number):
	merchants[i] = Merchant(i)


shortest_paths = Merchant.ShortestPaths(merchants)

projects=Project.OwnerPrefProjects(merchants)

#Sort un-funded projects according to mu/sigma=sqrt(k)
unfunded_list = Project.Unfunded(projects)

Project.Fund(unfunded_list,projects,merchants )


print(f'\n\nOutput')
for i in range(Project.number):
	Project.Print(projects[i])

Project.Profit(projects,merchants)
Merchant.Status(merchants)












