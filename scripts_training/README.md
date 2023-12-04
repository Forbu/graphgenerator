## Training

This is the training scripts for the project. The training is done in two steps: first, the model is trained on the training set, then the model is fine-tuned on the validation set. The training is done using the [PyTorch](https://pytorch.org/) framework.

### Training graphGDP

To train a graph GDP we can run :

```bash
python3 train_diffusion.py
```