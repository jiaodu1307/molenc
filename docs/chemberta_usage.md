# ChemBERTa Encoder Usage Guide

This guide explains how to use the ChemBERTa encoder in the MolEnc library.

## Overview

ChemBERTa is a RoBERTa-based transformer model pre-trained on molecular SMILES strings to learn contextualized molecular representations. It was developed by Seyone Chithrananda and is available on Hugging Face.

## Installation

To use ChemBERTa, you need to install the required dependencies:

```bash
pip install molenc[pretrained]
```

Or install the specific dependencies:

```bash
pip install torch transformers
```

## Basic Usage

### 1. Simple Encoding

```python
from molenc import MolEncoder

# Create ChemBERTa encoder
encoder = MolEncoder('chemberta')

# Encode a single molecule
smiles = "CCO"  # Ethanol
embedding = encoder.encode(smiles)
print(f"Embedding shape: {embedding.shape}")  # (768,)

# Encode multiple molecules
smiles_list = ["CCO", "CC(=O)O", "c1ccccc1"]
embeddings = encoder.encode_batch(smiles_list)
print(f"Embeddings shape: {embeddings.shape}")  # (3, 768)
```

### 2. Custom Configuration

```python
from molenc import MolEncoder

# Create ChemBERTa encoder with custom parameters
encoder = MolEncoder(
    'chemberta',
    model_name='seyonec/ChemBERTa-zinc-base-v1',  # Specific pre-trained model
    max_length=256,                               # Maximum sequence length
    pooling_strategy='mean'                      # How to pool token embeddings
)

# Encode molecules
embedding = encoder.encode("CCO")
```

## Advanced Usage

### 1. Pooling Strategies

ChemBERTa supports different pooling strategies to convert token embeddings to sentence-level representations:

- `cls`: Use the [CLS] token embedding (default)
- `mean`: Mean pooling over non-padded tokens
- `max`: Max pooling over non-padded tokens

```python
from molenc import MolEncoder

# Mean pooling
encoder = MolEncoder('chemberta', pooling_strategy='mean')
embedding = encoder.encode("CCO")

# Max pooling
encoder = MolEncoder('chemberta', pooling_strategy='max')
embedding = encoder.encode("CCO")
```

### 2. Device Selection

You can specify which device to use for computation:

```python
from molenc import MolEncoder

# Use CPU
encoder = MolEncoder('chemberta', device='cpu')

# Use GPU (if available)
encoder = MolEncoder('chemberta', device='cuda')

# Auto-detect (default)
encoder = MolEncoder('chemberta')  # Uses GPU if available, otherwise CPU
```

## Performance Comparison

Here's a comparison of different encoders in MolEnc:

| Encoder    | Dimension | Speed     | Memory Usage | Accuracy |
|------------|-----------|-----------|--------------|----------|
| Morgan     | 2048      | Very Fast | Very Low     | Medium   |
| MACCS      | 166       | Very Fast | Very Low     | Medium   |
| ChemBERTa  | 768       | Medium    | High         | High     |
| MolBERT    | 768       | Medium    | High         | High     |
| UniMol     | 512       | Slow      | Very High    | Very High|

## Example Applications

### 1. Molecular Similarity

```python
import numpy as np
from molenc import MolEncoder

# Create encoder
encoder = MolEncoder('chemberta')

# Encode molecules
smiles1 = "CCO"
smiles2 = "CCCO"
emb1 = encoder.encode(smiles1)
emb2 = encoder.encode(smiles2)

# Calculate cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarity = cosine_similarity(emb1, emb2)
print(f"Similarity between {smiles1} and {smiles2}: {similarity:.4f}")
```

### 2. Batch Processing

```python
from molenc import MolEncoder

# Create encoder
encoder = MolEncoder('chemberta')

# Large list of SMILES
smiles_list = [
    "CCO", "CC(=O)O", "c1ccccc1", "CCN(CC)CC", 
    "C1CCCCC1", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
]

# Efficient batch encoding
embeddings = encoder.encode_batch(smiles_list)
print(f"Batch embeddings shape: {embeddings.shape}")
```

## API Reference

### MolEncoder('chemberta', ...)

Creates a ChemBERTa encoder instance.

**Parameters:**
- `model_name` (str): Name of the pre-trained model (default: "seyonec/ChemBERTa-zinc-base-v1")
- `max_length` (int): Maximum sequence length (default: 512)
- `pooling_strategy` (str): How to pool token embeddings (default: "cls")
- `device` (str): Device to run on ("cpu", "cuda", or None for auto)

**Methods:**
- `encode(smiles)`: Encode a single SMILES string
- `encode_batch(smiles_list)`: Encode a batch of SMILES strings
- `get_output_dim()`: Get the output dimension
- `get_config()`: Get encoder configuration

### ChemBERTaEncoder

The underlying encoder class.

**Additional Methods:**
- `get_feature_names()`: Get feature names for dimensions

## Related Encoders

- **MolBERT**: Another transformer-based molecular encoder
- **UniMol**: 3D structure-based encoder


## References

1. Chithrananda, S., & Filez, G. (2020). ChEMBL fingerprint and Deep Learning models. Zenodo. https://doi.org/10.5281/zenodo.3872359
2. Liu, Y., Ott, M., Goyal, N., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv preprint arXiv:1907.11692.