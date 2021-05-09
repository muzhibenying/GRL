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

        self.layer_update_state_vector1 = dgl.nn.pytorch.GraphConv(
            self.node_size, self.node_size, weight = True,
        )
        self.update_activation1 = torch.nn.ReLU()
        self.layer_update_state_vector2 = dgl.nn.pytorch.GraphConv(
            self.node_size, self.node_size, weight = True
        )
        self.update_activation2 = torch.nn.ReLU()
        self.update_edge_weights = torch.nn.Linear(self.node_size * 2 + 1, 1)
        self.update_activation_edges = torch.nn.ReLU()

        self.linear1_add_node = torch.nn.Linear(self.node_size * 2, 1)
        self.relu1_add_node = torch.nn.ReLU()
        self.linear2_add_node = torch.nn.Linear(self.node_size * 2 + 1, self.node_size)

        self.linear1_add_edge = torch.nn.Linear(self.node_size * 4, 1)
        self.relu1_add_edge = torch.nn.ReLU()

        self.s = None

    def update_edges(self, edges):
        print("size of node feature is {}".format(edges.src["h"].size()))
        print("size of edge feature is {}".format(edges.data["w"].size()))
        w = self.update_edge_weights(torch.cat([edges.src["h"], edges.dst["h"], edges.data["w"]], dim = -1))
        w = self.update_activation_edges(w).squeeze(0)
        return {"w": w}

    """
    After a node or edge is generated, the state vector for the generated
    graph will be updated.

    For the Graph G(V, E) with h_{i} as node feature and e_{i} as edge_feature
                            h_{i}^{'} = GraphConv(h)
                            s = \sum h_{i}^{'}]
    """
    def update_state_vector(self, g):
        h = g.ndata["h"]
        h = self.layer_update_state_vector1(g, h)
        h = self.update_activation1(h)
        g.ndata["h"] = h
        print("size of the edge features of the graph is {}".format(g.edata["w"]))
        g.apply_edges(self.update_edges)
        h = self.layer_update_state_vector2(g, h)
        h = self.update_activation2(h)
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
            init_node_feature = self.linear2_add_node(torch.cat([z, self.s, token], dim = -1))
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
    """def add_edge(self, g, z):
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
                    g.edata["w"] = token"""
    def add_edge(self, g, z):
        for i in range(g.num_nodes()):
            token = self.relu1_add_edge(self.linear1_add_edge(
                torch.cat([z, self.s, g.ndata["h"][g.num_nodes() - 1].unsqueeze(0), g.ndata["h"][i].unsqueeze(0)],
            dim = -1
            )))
            g.add_edges(torch.tensor([g.num_nodes() - 1]), torch.tensor([i]))
            if "w" in g.edata.keys():
                g.edata["w"][g.num_edges() - 1] = token
                print("size of token is {}".format(token.size()))
            else:
                g.edata["w"] = token
                print("size of token is {}".format(token.unsqueeze(0).size()))
  
    def forward(self, z):
        self.s = z
        g = dgl.DGLGraph()
        g.add_nodes(1)
        initial_feature = self.linear2_add_node(self.relu1_add_node(
            torch.cat([z, self.s, self.linear1_add_node(torch.cat([z, self.s], dim = -1))], dim = -1)
        ))
        g.ndata["h"] = initial_feature
        g.add_edges(torch.tensor([0]), torch.tensor([0]))
        g.edata["w"] = torch.tensor([[1.]])
        self.s = self.update_state_vector(g)
        while self.add_node(g, z) == 1:
            self.add_edge(g, z)
            self.s = self.update_state_vector(g)
        return g

class Encoder(torch.nn.Module):
    def __init__(self, node_size, hidden_size, out_size, num_layers):
        """
        node_size: the size of the node features
        hidden_size: the size of node features after GCN
        out_size: the size of the global feature
        num_layers: the number of GCN layers used in the encoder
        """
        super(Encoder, self).__init__()
        self.node_size = node_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.num_layers = num_layers
        self.layers = []
        self.layers.append(dgl.nn.pytorch.GraphConv(node_size, hidden_size, weight = True))
        self.layers.append(torch.nn.ReLU())
        for __ in range(self.num_layers - 2):
            self.layers.append(dgl.nn.pytorch.GraphConv(hidden_size, hidden_size, weight = True))
            self.layers.append(torch.nn.ReLU())
        self.layers.append(dgl.nn.pytorch.GraphConv(hidden_size, out_size, weight = True))
    
    def forward(self, g, h):
        h_1 = self.layers[1](self.layers[0](g, h))
        for idx in range(self.num_layers - 2):
            h_1 = self.layers[2 * idx + 3](self.layers[2 * idx + 2](g, h_1))
        h_1 = self.layers[-1](g, h_1)
        g.ndata["h_1"] = h_1
        h_g = dgl.mean_nodes(g, "h_1")
        return h_g

class Discriminator(torch.nn.Module):
    def __init__(self, node_size, hidden_size, out_size):
        """
        node_size: the size of the node features
        hidden_size: the size of the features after GCN layers
        out_size: the size of the global features
        """
        super(Discriminator, self).__init__()
        self.node_size = node_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.graphLayer1 = dgl.nn.pytorch.GraphConv(node_size, hidden_size, weight = True, allow_zero_in_degree = True)
        self.activation1 = torch.nn.ReLU()
        self.graphLayer2 = dgl.nn.pytorch.GraphConv(hidden_size, hidden_size, weight = True, allow_zero_in_degree = True)
        self.activation2 = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(self.out_size, self.hidden_size)
        self.linear2 = torch.nn.Linear(self.hidden_size * 2, 1)
    
    def forward(self, g, h, z):
        h_d = self.activation1(self.graphLayer1(g, h))
        h_d = self.activation2(self.graphLayer2(g, h_d))
        g.ndata["h_d"] = h_d
        h_d = dgl.mean_nodes(g, "h_d")
        z = self.linear1(z)
        h = torch.cat([h_d, z], dim = 1)
        score = self.linear2(h)
        return score


