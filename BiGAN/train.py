import argparse
import dgl
import model
import numpy as np
import sklearn
from sklearn import model_selection
from sklearn import svm
import torch

def  create_data_loader(opt):
    dataset = dgl.data.TUDataset(opt.dataset)
    dataset_info = {}
    dataset_info["num_nodes"] = dataset.max_num_node
    data_loader = dgl.dataloading.GraphDataLoader(dataset)
    max_in_degrees = 0
    max_node_labels = 0
    for graph, label in dataset:
        if "node_attr" not in graph.ndata.keys():
            in_degree = max(graph.in_degrees())
            node_label = max(graph.ndata["node_labels"])
            if in_degree >= max_in_degrees:
                max_in_degrees = int(in_degree)
            if node_label >= max_node_labels:
                max_node_labels = int(node_label)
    for graph, label in dataset:
        if "node_attr" not in graph.ndata.keys():
            graph.ndata["h"] = torch.cat([
                torch.nn.functional.one_hot(graph.in_degrees(), num_classes = max_in_degrees + 1),
                torch.nn.functional.one_hot(graph.ndata["node_labels"].type(torch.LongTensor),\
                                                                                    num_classes = max_node_labels + 1)
            ], dim = 1).type(torch.FloatTensor)
        else:
            graph.ndata["h"] = graph.ndata["node_attr"].type(torch.FloatTensor)
        if "w" not in graph.edata.keys() and "edge_labels" not in graph.edata.keys():
            graph.edata["w"] = torch.ones(graph.num_edges())
        elif "edge_labels" in graph.edata.keys():
            graph.edata["w"] = graph.edata["edge_labels"]
    dataset_info["node_feature_size"] = dataset[0][0].ndata["h"].size()[1]
    return data_loader, dataset_info

def main():
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required = True, help = "give the name of a graph dataset")
    parser.add_argument("--device", default = "cpu", help = "use gpu or cpu to run the program")
    parser.add_argument("--lr", default = 1e-3, help = "learning rate")
    parser.add_argument("--num_epochs", default = 100, type = int, help = "the number of epochs to train the BiGAN")
    opt = parser.parse_args()
    print(opt)

    with open("results.csv", "w") as f:
        f.write("{},{},{}\n".format("epoch", "mean accuracy", "std of accuracies"))
    with open("observation.csv", "w") as f:
        f.write("epoch,loss_d,loss_g,num_nodes,mun_edeges\n")

    data_loader, dataset_info = create_data_loader(opt)
    print("dataset information\t{}".format(dataset_info))
    
    generator = model.Generator(max_nodes = dataset_info["num_nodes"], 
                                                    node_size = dataset_info["node_feature_size"],
                                                    graph_size = dataset_info["node_feature_size"]).to(opt.device)
    encoder = model.Encoder(node_size = dataset_info["node_feature_size"],
                                               hidden_size = 512, out_size = dataset_info["node_feature_size"],
                                               num_layers = 2).to(opt.device)
    discriminator = model.Discriminator(node_size = dataset_info["node_feature_size"],
                                                                 hidden_size = 512, 
                                                                 out_size = dataset_info["node_feature_size"])
    optimizerG = torch.optim.Adam([{"params": encoder.parameters()},
                                    {"params": generator.parameters()}], lr = opt.lr, betas = (0.5, 0.999))
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr = opt.lr, betas = (0.5, 0.999))
    critierion = torch.nn.BCELoss()

    for epoch in range(opt.num_epochs):
        
        for i, (data, label) in enumerate(data_loader):
            real_label = torch.tensor([[1.]])
            fake_label = torch.tensor([[0.]])
            
            d_real = data.to(opt.device)

            z_fake = torch.randn((1, dataset_info["node_feature_size"]))
            d_fake = generator(z_fake)

            z_real = encoder(d_real, d_real.ndata["h"])

            mu, log_sigma = z_real[:, :dataset_info["node_feature_size"]], \
                                        z_real[:, dataset_info["node_feature_size"] - 1:]
            sigma = torch.exp(log_sigma)
            epsilon = torch.randn(1, dataset_info["node_feature_size"])

            output_z = mu + epsilon * sigma

            output_real = discriminator(d_real, d_real.ndata["h"], z_real)
            output_fake = discriminator(d_fake, d_fake.ndata["h"], z_fake)

            loss_d = critierion(output_real, real_label) + critierion(output_fake, fake_label)
            loss_g = critierion(output_fake, real_label) + critierion(output_real, fake_label)

            """if loss_g.item() < 1.5:
                optimizerD.zero_grad()
                loss_d.backward(retain_graph = True)
                #optimizerD.step()
            
            optimizerG.zero_grad()
            loss_g.backward()
            optimizerD.step()
            optimizerG.step()"""
            ite = 0
            while loss_g.item() < 1.0 and ite < 5:
                optimizerD.zero_grad()
                loss_d.backward()
                optimizerD.step()
                z_fake = torch.randn((1, dataset_info["node_feature_size"]))
                d_fake = generator(z_fake)
                z_real = encoder(d_real, d_real.ndata["h"])
                output_real = discriminator(d_real, d_real.ndata["h"], z_real)
                output_fake = discriminator(d_fake, d_fake.ndata["h"], z_fake)
                loss_d = critierion(output_real, real_label) + critierion(output_fake, fake_label)
                loss_g = critierion(output_fake, real_label) + critierion(output_real, fake_label)
                ite += 1
            optimizerG.zero_grad()
            loss_g.backward()
            optimizerG.step()


            if i % 100 == 0:
                with open("observation.csv", "a") as f:
                    f.write("{},{},{},{},{}\n".format(epoch, loss_d, loss_g, d_fake.num_nodes(), 
                                                                        d_fake.num_edges()))

        
        if epoch % 1 == 0:
            x = []
            y = []
            with torch.no_grad():
                for (data, label) in data_loader:
                    x.append(encoder(data, data.ndata["h"]).numpy())
                    y.append(label.numpy())
            x = np.array(x).squeeze(1)
            print("shape of x is {}".format(x.shape))
            y = np.array(y)
            print("shape of y is {}".format(y.shape))
            params = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            kf = model_selection.StratifiedKFold(n_splits = 10, 
                                                shuffle = True, random_state = None)
            accuracies = []
            for train_index, test_index in kf.split(x, y):
                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]
                classifier = model_selection.GridSearchCV(
                    svm.LinearSVC(max_iter = 1000), params, cv = 5, scoring = "accuracy", verbose = 0
                )
                classifier.fit(x_train, y_train)
                accuracies.append(sklearn.metrics.accuracy_score(
                    y_test, classifier.predict(x_test)
                ))
            print(np.mean(accuracies), np.std(accuracies))
            with open("results.csv", "a") as f:
                f.write("{},{},{}\n".format(epoch, np.mean(accuracies), np.std(accuracies)))

if __name__ == "__main__":
    main()