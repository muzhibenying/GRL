import argparse
import dgl
import model

def  create_data_loader(opt):
    dataset = dgl.data.TUDataset(opt.dataset)
    dataset_info = {}
    dataset_info["num_nodes"] = dataset.max_num_node
    data_loader = dgl.dataloading.GraphDataLoader(dataset)
    for graph, label in data_loader:
        print(graph)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required = True, help = "give the name of a graph dataset")
    parser.add_argument("--device", default = "cpu", help = "use gpu or cpu to run the program")
    opt = parser.parse_args()
    print(opt)

    create_data_loader(opt)

if __name__ == "__main__":
    main()