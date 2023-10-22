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

## Runnig experimentation

```bash
cd scripts_training/
python train_erdos.py --dataset_type grid --batch_size 32
```