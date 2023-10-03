# GraphGeneration

In this repository, we will explore differents architectures to do graph generation. 

## Exploration

We will explore different architecture here : 

- GraphVAE (autoencoding with graph)

- GraphGAN (GAN adaptation to graph)

- Diffusion (adaptation to latent diffusion model)

- GRAN (recurrent generation)


## Runnig experimentation

```bash
cd scripts_training/
python train_erdos.py --dataset_type grid --batch_size 32
```