import collections
import dgl
import matplotlib.pyplot as plt

def get_information(dataset):
    """
    This function is designed to collect some statistical information about the dataset.
    """
    dataset = dgl.data.TUDataset(dataset)
    print(dataset[0])
    graph_nodes = []
    graph_edges = []
    graph_in_degrees = []
    for graph, label in dataset:
        graph_nodes.append(graph.num_nodes())
        graph_edges.append(graph.num_edges())
        mean_in_degrees = sum(graph.in_degrees()) / len(graph.in_degrees())
        graph_in_degrees.append(mean_in_degrees)
    print("average number of nodes: {}".format(sum(graph_nodes) / len(graph_nodes)))
    print("average number of edges: {}".format(sum(graph_edges) / len(graph_edges)))
    print("average number of in degrees: {}"\
                .format(sum(graph_in_degrees) / len(graph_in_degrees)))
    nodes_counter = collections.Counter(graph_nodes)
    print("counter of node number is {}".format(nodes_counter))
    plt.bar(nodes_counter.keys(), nodes_counter.values())
    plt.savefig("dataset_information/Distribution of number of nodes.png")
    plt.close()
    edges_counter = collections.Counter(graph_edges)
    print("counter of edge number is {}".format(edges_counter))
    plt.bar(edges_counter.keys(), edges_counter.values())
    plt.savefig("dataset_information/Distribution of number of edges.png")
    plt.close()
    degrees_counter = collections.Counter(graph_in_degrees)
    print("counter of in degrees number is {}".format(degrees_counter))
    plt.bar(degrees_counter.keys(), degrees_counter.values())
    plt.savefig("dataset_information/Distribution of number of in degrees.png")
    plt.close()
    
if __name__ == "__main__":
    get_information("MUTAG")