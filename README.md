# GRL
The github repository for project of graph representation learning

## nodeView
This project is cloned from [https://github.com/kavehhassani/mvgrl][https://github.com/kavehhassani/mvgrl].

Now I have modified the code for graph classification under the ```graph``` directory to add the mutual information among different node features.


## BiGAN
The model will use BiGAN method to learn graph representation.

* ```generator```: the generator is similar with DGMG, but is modified to learn from the generated graph other than actions
* ```encoder```: the encoder is a GCN with graph attention layers
* ```discriminator```: a GCN combined with NN
