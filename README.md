# GraphGeneration

In this repository, we will explore differents architectures to do graph generation. 

## Exploration

We will explore different architecture here : 

- GraphVAE (autoencoding with graph)
TODO later

- GraphGAN (GAN adaptation to graph)

- Diffusion (adaptation to latent diffusion model)

DiGress paper : https://arxiv.org/pdf/2209.14734.pdf (later)

GraphGDP paper : https://arxiv.org/pdf/2212.01842.pdf (currently coding)

- GRAN (recurrent generation) (implementation is done here)

Paper : https://arxiv.org/abs/1910.00760

## Running experimentation

```bash
cd scripts_training/
python train_erdos.py --dataset_type grid --batch_size 32
```

## Results

We did the experimentation for two algorithm GRAN (auto regressive) and graphGDP (diffusion / score generative model)

#### GRAN implementation

For the GRAN implementation, we achieve (visual) good generation :

For exemple for the grid graph the GRAN algorithm achieve almost perfectly to generate those graph :


Same thing for the watts strogatz graph (annealing graph : https://en.wikipedia.org/wiki/Watts%E2%80%93Strogatz_model)

Generated with GRAN :

![image](https://github.com/Forbu/graphgenerator/assets/11457947/28c76ed1-e0e8-43a5-80b1-b84db4990ae9)

Random generation with the networkx algorithm : 

![image](https://github.com/Forbu/graphgenerator/assets/11457947/1c08121a-753d-4f33-8ab3-44ca36a96c65)


For simple grid graph the GRAN achieve perfect representation :

GRAN generation :

![image](https://github.com/Forbu/graphgenerator/assets/11457947/892612b1-0a8f-407d-9052-643db4bc18dd)


Networkx generation :

![image](https://github.com/Forbu/graphgenerator/assets/11457947/0774de4e-7814-4970-b5c2-7b1681cacc8d)


#### GrappgGDP implementation

Good result for the algorithm were harder to obtain, we only test it for grid like generation.

The generated graph look like this after training : 

![image](https://github.com/Forbu/graphgenerator/assets/11457947/543401ea-8dea-4069-a28c-ae1eb0f588b9)

It feels that the model could (almost) generate a planar graph, but were far from grid like.

For bigger grid size (10x10), we were also far although it has some planar vibe :

![image](https://github.com/Forbu/graphgenerator/assets/11457947/1ed4cc8b-6beb-4e18-8efa-915d583420e6)


For this graphGDP, I feel I have to do some progress in the GNN architecture because the model have a good overall generated adjacency matrix
but a very bad topology :

Generated adjacency matrix :

![image](https://github.com/Forbu/graphgenerator/assets/11457947/08cdf146-9212-4121-8c1e-ab222a994959)

Typical adjacency matrix for grid graph (networkx) :

![image](https://github.com/Forbu/graphgenerator/assets/11457947/9c7c5ea4-2a49-4c22-b6a0-64745be23542)

The forward diffusion process also look good and the reverse too but the resulting topology doesn't look like a grid one ...














