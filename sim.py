import numpy as np


from merchant import Merchant, Project



np.random.seed(int(str(Merchant.seed)[-8:]))#Random seed must be <2**32-1 so take D:H:mm:ss
merchants = np.empty(shape=(Merchant.number,), dtype=object)
  #Needs working on


#Create the merchants
for i in range(Merchant.number):
	merchants[i] = Merchant(i)

while Project.count<30:
	shortest_paths = Merchant.ShortestPaths(merchants)

	projects=Project.OwnerPrefProjects(merchants)

	#Sort un-funded projects according to mu/sigma=sqrt(k)
	unfunded_list = Project.Unfunded(projects)

	Project.Fund(unfunded_list,projects,merchants )

	print(f'\n\nSTEP {Merchant.step}')
	for j in range(Project.number):
		print(Project.Print(projects[j]))

	for j in range(Merchant.number):
		print(Merchant.Print(merchants[j]))

	Merchant.ProjectAllocation(merchants,projects)



	Project.Profit(projects,merchants)
	Merchant.step += 1



	Project.Dump(projects)
	Project.Delete(projects)
	Merchant.EndStep(merchants)
	print(f'\nEnd of Turn {Merchant.step}')
	for j in range(Merchant.number):
		print(Merchant.Print(merchants[j]))




#Project.Dump(projects)












