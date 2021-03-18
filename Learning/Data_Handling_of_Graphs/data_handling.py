import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype = torch.long)
x = torch.tensor([[-1], [0], [1]], dtype = torch.float)

data = Data(x, edge_index = edge_index.t().contiguous())
print("data = {}".format(data))
print("keys of data = {}".format(data.keys))
print("features of data = {}".format(data["x"]))
for key, item in data:
    print("{} of {} found in data".format(item, key))