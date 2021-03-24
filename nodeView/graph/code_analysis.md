The ```load``` function will load a dataset and return its adjacency matrix, diffusion matrix, feature matrix and a list contains the number of nodes in each graph.

For the MUTAG data set, the return value is:

* ```adj```: (188, 28, 28), 188 is the number of graphs, 28 is the maximum number of nodes a single graph contains in the dataset
* ```diff```: (188, 28, 28)
* ```feat```: (188, 28, 11), 11 is the feature size for each node
* ```labels```: (188, ), it is the labels for the 188 graphs
* ```num_nodes```: [17, 21, ...], a list, each element represents the number of nodes in the graph, the maximum value is 28