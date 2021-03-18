from torch_scatter import scatter_mean
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

dataset = TUDataset(root = "tmp/ENZYMES", name = "ENZYMES", use_node_attr = True)
loader = DataLoader(dataset, batch_size = 32, shuffle = True)

for data in loader:
    print("data = {}".format(data))
    print("number of graphs = {}".format(data.num_graphs))
    x = scatter_mean(data.x, data.batch, dim = 0)
    print("size of x = {}".format(x.size()))