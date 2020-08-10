def compute(tribes,adj_matrix,precision):

	import networkx as nx

	spectra = []

	for tribe in tribes:
		G = nx.from_numpy_array(adj_matrix[np.ix_(tribe,tribe)],create_using=nx.DiGraph)

		# Find the largest connected component of the graph 
		largest = max(nx.strongly_connected_components(G), key=len)

		# Compute the Chung's laplacian matrix of tribe's largest connected component 
		L = nx.directed_laplacian_matrix(G.subgraph(largest))

		# Find the eigenvalues
		eig = LA.eigvals(L)

		# Order the non-zero eigenvalues and round to desired precision
		spectrum = np.round(np.unique(eig[np.nonzero(eig)]),precision)

		spectra.append(spectrum)

	return(spectra)
