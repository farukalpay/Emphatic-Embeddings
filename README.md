# Empathic Word Embeddings

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

This repository contains a Python implementation for creating **Empathic (Affect-Augmented) Word Embeddings**. The model takes standard word embeddings (like GloVe or word2vec) and enriches them with a new dimension representing emotional affect (e.g., positivity vs. negativity).

This allows for novel semantic operations, such as finding words that are semantically similar but emotionally opposite (e.g., finding `sad` as an "opposite-affect" neighbor of `happy`).

The model supports two modes of operation:
1.  **Supervised Mode:** Learns an affect dimension by projecting embeddings onto known sentiment scores from a lexicon (e.g., a Valence-Arousal-Dominance lexicon).
2.  **Unsupervised Mode:** Discovers a principal affect dimension directly from the geometry of the embedding space by constructing a signed graph and solving a generalized eigenvalue problem. It identifies antonyms using common prefixes (like `un-`, `in-`, `dis-`) to create negative edges in the graph.

## Key Features

-   **Affect Augmentation:** Adds a new, continuous dimension for emotional valence to any set of pre-trained word embeddings.
-   **Supervised & Unsupervised Training:** Can learn from explicit sentiment labels or discover affect axes automatically.
-   **Opposite-Affect Queries:** Find words with opposite emotional polarity (e.g., the opposite of "courage" might be "fear").
-   **Spectral-Cap Certificate:** In unsupervised mode, provides a mathematical certificate (R, θ) to quantify the confidence in the discovered affect dimension.
-   **Self-Contained & Ready-to-Run:** The main script includes a built-in demo with dummy data, so you can run it immediately after installation.

## Installation

The project can now be installed like any other Python package.

```bash
pip install emphatic-embeddings
```

This provides both the library and an `empathic-embeddings` command‑line tool.
To install from a local clone instead, run:

```bash
pip install .
```

## Quick Start

### Library

```python
from empathic_embeddings import EmpathicSpace, read_vec
E = dict(read_vec("embeddings.txt"))
space = EmpathicSpace(E)
print(space.nearest(space.vector("good")))
```

### Command line

```bash
empathic-embeddings path/to/your/glove.6B.100d.txt --probe king
```

To see the original demonstration used during development:

```bash
python -c "import empathic_embeddings as ee; ee.main()"
```

## Using the Command-Line Interface (CLI)

For use with your own data, the package installs a command-line tool named `empathic-embeddings`. You will need:
1.  A file with pre-trained word embeddings in `word2vec` text format.
2.  (Optional) A VAD lexicon file for supervised mode. This should be a 4-column text file: `word V A D`.

### CLI Arguments

-   `vecs`: (Required) Path to your `.vec` or `.txt` embedding file.
-   `--vad`: (Optional) Path to a VAD lexicon. If provided, runs in **supervised mode**. If omitted, runs in **unsupervised mode**.
-   `--save`: Path to save the trained model metadata (default: `empathic.json.gz`).
-   `--probe`: A word to use for the demo queries (default: `happy`).
-   `--topk`: Number of neighbors to show in the demo (default: `5`).

### Example: Unsupervised Mode

This command will train a model using your embeddings and run demo queries for the word "king".

```bash
empathic-embeddings path/to/your/glove.6B.100d.txt --probe king
```

### Example: Supervised Mode

This command will use a VAD lexicon to train the model in supervised mode.

```bash
empathic-embeddings path/to/your/glove.6B.100d.txt --vad path/to/your/vad-lexicon.txt --probe happy
```

## How It Works: A Brief Overview

### Supervised Mode
When a VAD lexicon is provided, the model learns a linear projection (using Ridge regression) from the high-dimensional word embedding space to the 3D VAD (Valence, Arousal, Dominance) space. The valence dimension from this projection is then appended to the original embeddings to create the empathic space.

### Unsupervised Mode
Without a lexicon, the model builds a **signed graph** where nodes are words.
-   **Positive edges** connect words that are close in the original embedding space (synonyms or semantically related words).
-   **Negative edges** connect words that are likely antonyms. These are discovered by looking for pairs like `happy` and `unhappy`.

The model then solves the generalized eigenvalue problem **L⁺z = λL⁻z**, where `L⁺` and `L⁻` are the graph Laplacians for the positive and negative graphs, respectively. The eigenvector `z` corresponding to the smallest non-trivial eigenvalue represents the principal axis of affect that best separates synonyms from antonyms. This vector `z` provides the affect scores for each word.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This code is an implementation of the concepts described in the paper **"Empathic (Affect-Augmented) Word Embeddings"**. Please cite the original authors if you use this work in your research.
*Alpay, F., & Kilictas, B. (2025). Alpay Algebra VI: The Universal Semantic Virus and Transfinite Embedding Alignment. Zenodo. https://doi.org/10.5281/zenodo.15939980*
