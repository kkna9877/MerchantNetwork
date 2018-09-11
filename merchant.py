import numpy as np
import igraph as ig
import typing

class Merchant:
	#Class variables, the count (id) of the merchants and the number of living merchants: count ge number
	count: int = 0
	number: int = 10
	neighbour:int = 30
	reserve: float = 2.5
	status_cash_conversion: float = 4.



	#Common initialisations
	initial_wealth = 100.
	initial_no_connections: int = 3
	initial_status: float = 0.
	initial_distance: float = 10.
	initial_cash: float = 5.

	initial_connections =np.zeros(initial_no_connections+1, dtype = np.float16)
	lower = int(np.floor(initial_no_connections / 2))
	upper = int(np.ceil(initial_no_connections / 2))
	initial_connections[:lower] = initial_distance
	initial_connections[-upper:] = initial_distance




	def __init__(self, n):
		Merchant.count += 1
		self.id: int = Merchant.count
		self.idx: int = n #slot in the array of merchant objects
		self.status = Merchant.initial_status
		self.cash = Merchant.initial_cash
		self.reserve = Merchant.reserve #What the merchant uses to live on, increases with status but can be used to avoid bankruptcy at the expense of status
		self.projects = []
		self.cur_funding = np.zeros(Merchant.number, dtype=np.float16)
		self.connections = np.ones(Merchant.number, dtype=np.float16)*1000.
		self.distances = np.zeros(Merchant.number, dtype=np.float16)

		if n-Merchant.lower < 0:
			#Need to add connections at the end of the connections array
			self.connections[n-Merchant.lower:] = Merchant.initial_connections[0:Merchant.lower-n]
			self.connections[:Merchant.initial_no_connections - Merchant.lower -n+1] = Merchant.initial_connections[Merchant.lower-n:]
		elif n+Merchant.upper >= Merchant.number:
			#Need to add connections a the beginning of the array
			no_mers_over = n+Merchant.upper-Merchant.number+1
			self.connections[:no_mers_over] = Merchant.initial_connections[-no_mers_over:]
			self.connections[no_mers_over-Merchant.initial_no_connections-1:] = Merchant.initial_connections[:Merchant.initial_no_connections + 1 -no_mers_over]

		else:
			self.connections[n-Merchant.lower:n+Merchant.upper+1]=Merchant.initial_connections


	def Print(merchant):
		print(f'Project id: {merchant.id:4}, Index :{merchant.idx:4},  Status: {merchant.status:5.3}, Cash: {merchant.cash:5.3}, Projects: {merchant.projects}')
		print(f'\t\tDistances: {merchant.distances}')


	def ConnectionsM(merchants):#Merchants is an array of merchant objects
		length = len(merchants)
		adj_matrix=np.zeros((length,length), dtype= np.float16)
		#First index is from, second is to
		for i in range(length):
			adj_matrix[i,:]=merchants[i].connections

		return adj_matrix

	def ShortestPaths(merchants):#adj_matrix is the adjacency matrix
		adj_matrix = Merchant.ConnectionsM(merchants)
		#np.save('adj_matrix',adj_matrix)

		g=ig.Graph.Weighted_Adjacency(adj_matrix.tolist())
		outarr=np.asarray(g.shortest_paths(weights='weight'))
		for i in range(Merchant.number):
			merchants[i].distances = outarr[i,:]

		return outarr


## Need to rethink how connections are being handled because we don't want this too complicated.
## Is it better to keep things simple here or in the Merchant object. Is using a list best
## or would an array for each merchant be better


class Project:
	# Class variables, the count gives the project id
	count: int = 0
	number: int = Merchant.number

	theta_min: float = 0.5
	theta_max: float = 2.0

	def __init__(self, idx, k, theta, owner):
		if (k>0) and (theta>0):
			Project.count += 1
			self.id: int = Project.count
			self.idx: int = idx
			self.k: float = k
			self.theta: float = theta
			self.expectation: float = round(k * theta,2)
			self.variance: float = theta * self.expectation
			self.payoff: float = round(float(np.random.gamma(k, theta, 1)),2)
			self.owner :int = owner # idx of owner
			self.investors= np.zeros(Merchant.number, dtype=np.float16) #each merchant's investment in project
			self.funded = False #has the project been fully funded
		else:
			raise ValueError(f'Project initialisation: k ({k}) and theta ({theta}) must be positive')

	def Unfunded(projects):#Input an array of project objects, return a LIST of unfunded projects
		unfunded=[]
		for i in range(len(projects)):
			if projects[i].funded == False:
				unfunded.append(i)

		return unfunded

	def Print(project):
		if not project.funded:
			print(f'Project id: {project.id:4}, Index :{project.idx:4},  k: {project.k:5.3}, theta: {project.theta:5.3}, Payoff: {project.payoff:5.3}, Owner: {project.owner:4}, \n Funded: {project.funded}\n')
		else:
			print(f'Project id: {project.id:4}, Index :{project.idx:4},  k: {project.k:5.3}, theta: {project.theta:5.3}, Payoff: {project.payoff:5.3}, Owner: {project.owner:4}, \n Funded: {project.investors}\n')

	def AllocateFunds(merchant, merch_id, cash, project, project_id):
		merchant.cash = merchant.cash - cash
		merchant.projects.append(project.idx)
		merchant.cur_funding[project_id] = cash
		project.investors[merch_id] = cash



	def Fund(unfunded_list,projects,merchants):#Take a list of unfunded projects idxs and fund themm from the merchant objects in array merchants
		#First order the unfunded projects according to the "sharpe ratio"
		number = len(unfunded_list)
		preferences = np.zeros((number))

		for i in range(number):
			preferences[i] = projects[unfunded_list[i]].k #Should be sqrt(k) to give share ration of Gamma, but we only are ordering

		ranked_unfunded_projects = []
		for i in list(reversed(np.argsort(preferences).tolist())): #List of
			ranked_unfunded_projects.append(unfunded_list[i])
		for i in ranked_unfunded_projects:
			owner=projects[i].owner
			contacts = merchants[owner].distances
			funding=np.zeros(Merchant.number)#Store the funding before commiting it
			owner_cash= merchants[owner].cash -  merchants[owner].reserve

			for j in  np.where(contacts < Merchant.neighbour)[0].tolist():#Loop through contacts
				funds_needed = projects[i].expectation - owner_cash - np.sum(funding)
				if funds_needed>0.:
					if j != owner:#The last investor is to be the owner, who will dip into status to fund the project
						if len(merchants[j].projects) == 0:#Merchant hasn't yet invested, don't go all in
							funding[j] = min(max(merchants[j].cash - 2. * merchants[j].reserve, 0),funds_needed)
						else: #Merchant has invested,
							own_project_funded = False
							for proj in merchants[j].projects:# but have they invested their own project
								if projects[proj].owner == j:
									own_project_funded=True
							if own_project_funded:
								funding[j] = min(max(merchants[j].cash -  merchants[j].reserve, 0),funds_needed)
							else:
								funding[j] = min(max(merchants[j].cash - 2. * merchants[j].reserve, 0),funds_needed)

			#Looped over all other investors, now see if the owner can fund
			if projects[i].expectation - owner_cash - np.sum(funding)>0.: #Need to dip into status
				funding_short = projects[i].expectation - owner_cash - np.sum(funding)
				if merchants[owner].status > funding_short/Merchant.status_cash_conversion:
					merchants[owner].status = merchants[owner].status - funding_short/Merchant.status_cash_conversion
					owner_cash = owner_cash + funding_short



			if round(projects[i].expectation - owner_cash - np.sum(funding)) == 0.:  #This project is funded
				Project.AllocateFunds(merchants[owner], owner, owner_cash, projects[i], i)
				for j in range(Merchant.number):
					if funding[j]>0.:
						if j==owner:
							print(f'Problem: double allocation?? merchants[{j}], {funding[j]}, projects[{i}]')
						Project.AllocateFunds(merchants[j], j, funding[j], projects[i], i)

				if round(float(np.sum(projects[i].investors)),2) == round(projects[i].expectation,2):
					projects[i].funded = True
				else:
					print(f'Problem funding project {i}: Sum of investments = {round(float(np.sum(projects[i].investors)),2)}, cost = {round(projects[i].expectation,2)}')
					for i in range(Project.number):
						Project.Print(projects[i])
					for i in range(Merchant.number):
						Merchant.Print(merchants[i])









