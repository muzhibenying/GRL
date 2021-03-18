from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

dataset = TUDataset(root = "/tmp/ENZYMES", name = "ENZYMES", use_node_attr = True)
loader = DataLoader(dataset, batch_size = 32, shuffle = True)

for batch in loader:
    print("batch = {}".format(batch))

    print("number of graphs in batch = ".format(batch.num_graphs))