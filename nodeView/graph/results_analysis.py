import pandas as pd 
import matplotlib.pyplot as plt
import os

dataFrame = pd.read_csv("results.csv")

os.makedirs("results", exist_ok = True)
for layer in [2, 8, 12]:
    for batch in [32, 64, 128, 256]:
        accuracies = []
        for epoch in [20, 40, 100]:
            accuracy = dataFrame[(dataFrame["Layer"] == layer) & \
                                 (dataFrame["Epoch"] == epoch) & \
                                 (dataFrame["Batch"] == batch)].loc[:, "Accuracy"].mean()
            accuracies.append(accuracy)
        plt.plot([20, 40, 100], accuracies, "-o", label = "batch size = {}".format(batch))
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join("results", "layer" + str(layer) + ".png"))
    plt.close()  