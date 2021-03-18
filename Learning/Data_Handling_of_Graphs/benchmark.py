from torch_geometric.datasets import TUDataset

dataset = TUDataset(root = "/tmp/ENZYMES", name = "ENZYMES")
print("dataset = {}".format(dataset))
print("len of dataset = {}".format(len(dataset)))
print("number of classes = {}".format(dataset.num_classes))
print("number of node features {}".format(dataset.num_node_features))

data = dataset[0]
print("The first graph is {}".format(data))