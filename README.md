## Denoising Autoencoder for Grid Color Sequence Reconstruction

This repository contains an implementation of a **Denoising Autoencoder (DAE)** using PyTorch to reconstruct original grid color sequences from versions where **one color has been masked entirely**. This model is inspired by denoising and representation learning literature and is applied to symbolic color sequences.

---

## Problem Statement

Given a sequence of integers (representing colors {0, 1, 2, 3}), randomly select one color per sequence to mask entirely (with a `[MASK]` token or integer 4). The model must learn to **infer the masked positions** and **reconstruct the original sequence**.

---

## Model Architecture

The model is a sequence-to-sequence autoencoder with the following structure:

- **Input Embedding**: Learnable embeddings for integers 0–4
- **Encoder**: LSTM processes the corrupted sequence
- **Decoder**: LSTM generates the reconstructed sequence
- **Output Layer**: Linear layer projecting hidden states to vocabulary size (4)

Loss function: `CrossEntropyLoss`  
Optimizer: `Adam`  

---

## Dataset

We use a grid sequence dataset available at:

 [http://slides.adjih.net/grids-35x1-5x1-c4-conflicts0.txt](http://slides.adjih.net/grids-35x1-5x1-c4-conflicts0.txt)

- Each row contains 40 tokens from {0, 1, 2, 3}
- Dataset is parsed into a `.csv` file and loaded using pandas

---

## Installation

```bash
git clone https://github.com/naqi7272/INRIA-France
```

```bash
cd INRIA-France
```

```bash
pip install torch pandas numpy matplotlib seaborn scikit-learn
```


## References

- Vincent, P., Larochelle, H., Bengio, Y., & Manzagol, P. A. (2008). *Extracting and composing robust features with denoising autoencoders*. Journal of Machine Learning Research, 11(12), 3371–3408.
- Bengio, Y., Courville, A., & Vincent, P. (2013). *Representation learning: A review and new perspectives*. IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(8), 1798–1828.
- PyTorch Official Docs: https://pytorch.org

---

