# Empathic Embeddings

Light‑weight research toolkit for affect‑augmented word embeddings. The package
adds an affect axis to existing word vectors via supervised lexical projection
or an unsupervised signed‑graph spectral method. A small Typer CLI mirrors the
Python API for quick experiments.

## Install

```bash
pip install emphatic-embeddings
```

For development install with extras:

```bash
pip install .[dev]
```

Optional extras provide additional functionality:

```bash
pip install emphatic-embeddings[viz]   # plotting utilities
pip install emphatic-embeddings[hf]    # HuggingFace integration
```

## Quickstart

```python
from empathic_embeddings import EmpathicEmbeddings
from empathic_embeddings.io import read_vec, load_vad

vectors = read_vec("glove.txt")
lexicon = {"happy": 1.0, "sad": 0.0}
model = EmpathicEmbeddings(vectors)
model.fit_supervised(lexicon)
print(model.nearest_neighbors("happy"))
```

## CLI

The command line interface exposes training and query helpers:

```bash
# train and save model
empathic-embeddings augment embeddings.vec --vad lexicon.txt --out model.npz

# query neighbours
empathic-embeddings query model.npz happy --topk 3
```

Run `empathic-embeddings --help` for the full list of commands.

## API

`EmpathicEmbeddings` is the main entry point. Key methods include:

- `fit_supervised(lexicon)`
- `fit_unsupervised()`
- `nearest_neighbors(word)`
- `opposite_neighbors(word)`
- `phrase_vector(text, method="sif")`

Each method is fully typed and documented in the source.

## Citing

If you use this project in academic work please cite:

```
Faruk Alpay. Empathic (Affect-Augmented) Word Embeddings. 2025.
```
