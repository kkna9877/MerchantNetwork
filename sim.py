import numpy as np


from merchant import Merchant, Project



np.random.seed(int(str(Merchant.seed)[-8:]))#Random seed must be <2**32-1 so take D:H:mm:ss
merchants = np.empty(shape=(Merchant.number,), dtype=object)
  #Needs working on


#Create the merchants
for i in range(Merchant.number):
	merchants[i] = Merchant(i)

while Merchant.count<15:
	shortest_paths = Merchant.ShortestPaths(merchants)

	projects=Project.OwnerPrefProjects(merchants)

	#Sort un-funded projects according to mu/sigma=sqrt(k)
	unfunded_list = Project.Unfunded(projects)

	Project.Fund(unfunded_list,projects,merchants )

	Merchant.ProjectAllocation(merchants,projects)


	print(f"\n\nAFTER FUNDING step {Merchant.step}")
	print(Merchant.connections)

	for j in range(Project.number):
		print(Project.Print(projects[j]))
	for j in range(Merchant.number):
		print(Merchant.Print(merchants[j]))

	Project.Profit(projects,merchants)
	Merchant.step += 1

	Merchant.Status(merchants)
	Project.Dump(projects)
	Project.Delete(projects)
	Merchant.EndStep(merchants)

	print("\n\nAFTER STATUS")
	print(Merchant.connections)


#Project.Dump(projects)












