import dgl
import torch 

class Generator(torch.nn.Module):
    """
    The generator will generate a graph based on a low-dimensional
    feature sampled from the normal distribution."""
    def __init__(self, max_nodes, node_size, graph_size):
        super(Generator, self).__init__()

        self.max_nodes = max_nodes
        self.node_size = node_size
        self.graph_size = graph_size

        self.gat_layer_update_state_vector = dgl.nn.pytorch.conv.GATConv(
            self.node_size, self.node_size, num_heads = 3, 
            allow_zero_in_degree = True
        )

        self.linear1_add_node = torch.nn.Linear(self.node_size * 2, 1)
        self.relu1_add_node = torch.nn.ReLU()
        self.linear2_add_node = torch.nn.Linear(1, self.node_size)

        self.linear1_add_edge = torch.nn.Linear(self.node_size * 4, 1)
        self.relu1_add_edge = torch.nn.ReLU()

        self.s = None

    """
    After a node or edge is generated, the state vector for the generated
    graph will be updated.

    For the Graph G(V, E) with h_{i} as node feature and e_{i} as edge_feature
                            h_{i}^{'} = GraphConv(h)
                            s = \sum h_{i}^{'}]
    """
    def update_state_vector(self, g):
        h = g.ndata["h"]
        h = torch.mean(self.gat_layer_update_state_vector(g, h), dim = 1)
        g.ndata["h"] = h
        s = dgl.mean_nodes(g, "h")
        return s
    
    """
    Every time when the generator decide whether to generate a new node, it
    will be based on the generated vector z and state vector s
                            token = relu(W([z, s]))
    * If token = 0, no node will be generated.
    * If token > 0, the initial feature of the node will be
                            h = W(token)
    """
    def add_node(self, g, z):
        token = self.relu1_add_node(self.linear1_add_node(torch.cat([z, self.s], 
                                                          dim = -1)))
        if token <= 0 or g.num_nodes() >= self.max_nodes:
            return 0
        else:
            init_node_feature = self.linear2_add_node(token)
            g.add_nodes(1)
            if "h" in g.ndata.keys():
                g.ndata["h"][g.num_nodes() - 1] = init_node_feature
            else:
                g.ndata["h"] = init_node_feature
            return 1
    
    """
    The new added node will connect with other nodes in the graph, it 
    will be based on the generated vector z state vector s and the features
    of the source node and destination node.
                         token = relu(W[z, s, h_u, h_v])
    * If token = 0, the edge will not be added
    * If token > 0, the token will be the weight of the edge  
    """
    def add_edge(self, g, z):
        for i in range(g.num_nodes()):
            token = self.relu1_add_edge(self.linear1_add_edge(
                torch.cat([z, self.s, g.ndata["h"][g.num_nodes() - 1].unsqueeze(0), g.ndata["h"][i].unsqueeze(0)],
            dim = -1)))
            if token < 1e-4:
                return 0
            else:
                g.add_edges(torch.tensor([g.num_nodes() - 1]), torch.tensor([i]))
                if "w" in g.edata.keys():
                    g.edata["w"][g.num_edges() - 1] = token
                else:
                    g.edata["w"] = token

    def forward(self, z):
        self.s = z
        g = dgl.DGLGraph()
        g.add_nodes(1)
        initial_feature = self.linear2_add_node(self.relu1_add_node(
            self.linear1_add_node(torch.cat([z, self.s], dim = -1))
        ))
        g.ndata["h"] = initial_feature
        self.s = self.update_state_vector(g)
        while self.add_node(g, z) == 1:
            self.add_edge(g, z)
            self.s = self.update_state_vector(g)
        return g

class Encoder(torch.nn.Module):
    def __init__(self, node_size, hidden_size, out_size, num_layers):
        super(Encoder, self).__init__()
        self.layers = []
        self.num_layers = num_layers
        self.layers.append(dgl.nn.pytorch.conv.GATConv(
            node_size, hidden_size, num_heads = 3,
            allow_zero_in_degree = True 
        ))
        for __ in range(num_layers - 1):
            self.layers.append(dgl.nn.pytorch.conv.GATConv(
                hidden_size, hidden_size, num_heads = 3,
                allow_zero_in_degree = True
            ))
    
    def forward(self, g, h):
        h_1 = torch.mean(self.layers[0](g, h), dim = 1)
        g.ndata["h_1"] = h_1
        h_1g = dgl.mean_nodes(g, "h_1")
        for idx in range(self.num_layers - 1):
            h_1 = torch.mean(self.layers[idx + 1](g, h_1), dim = 1)
            g.ndata["h_1"] = h_1
            #h_1g = torch.cat([dgl.mean_nodes(g, "h"), h_1g])
            h_1g = dgl.mean_nodes(g, "h_1")
        return h_1g

class Discriminator(torch.nn.Module):
    def __init__(self, node_size, hidden_size, out_size):
        super(Discriminator, self).__init__()
        self.node_size = node_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.gat_layer = dgl.nn.pytorch.conv.GATConv(
            node_size, hidden_size, num_heads = 3,
            allow_zero_in_degree = True
        )
        self.linear1 = torch.nn.Linear(self.out_size, self.hidden_size)
        self.linear2 = torch.nn.Linear(self.hidden_size * 2, 1)

    def forward(self, g, h, z):
        h_d = torch.mean(self.gat_layer(g, h), dim = 1)
        g.ndata["h_d"] = h_d
        z = self.linear1(z)
        h_d = dgl.mean_nodes(g, "h_d")
        h_d = torch.cat([h_d, z], dim = 1)
        score = torch.sigmoid(self.linear2(h_d))
        return score
