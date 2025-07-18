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

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Install dependencies:** The project requires `numpy` and `scipy`. You can install them using pip. It is recommended to use a virtual environment.
    ```bash
    pip install numpy scipy
    ```
    Or, create a `requirements.txt` file with the following content:
    ```
    numpy
    scipy
    ```
    And install using:
    ```bash
    pip install -r requirements.txt
    ```

## Quick Start: Running the Demo

The provided Python script (`empathic_embeddings.py` or similar) is ready to run out of the box. It contains a self-contained `main()` function that uses dummy data to demonstrate both the supervised and unsupervised modes.

Simply run the script:
```bash
python empathic_embeddings.py
```

### Expected Output

Running the script will produce the following output, demonstrating the model's capabilities on the sample data.

```
======================================================================
 Empathic Word Embeddings Demo
======================================================================
This script will now run a self-contained demonstration.
NOTE: This requires 'numpy' and 'scipy' to be installed.
You can install them by running: pip install numpy scipy


--- Running SUPERVISED Mode Demo (using dummy VAD data) ---
✅ Supervised model trained successfully.

Nearest neighbours of 'happy' in affect-augmented space:
  happy           1.000
  joyful          0.998
  excellent       0.998
  good            0.995
  neutral         0.935

Opposite-affect neighbours of 'happy':
  miserable       -0.474
  unhappy         -0.613
  terrible        -0.643
  sad             -0.709
  bad             -0.727


--- Running UNSUPERVISED Mode Demo (no VAD data) ---
✅ Unsupervised model trained successfully.

Words with most negative discovered affect:
  unhappy        -1.917
  bad            -0.779
  miserable      -0.760
  sad            -0.760
  terrible       -0.704

Words with most positive discovered affect:
  happy           1.618
  joyful          1.120
  excellent       1.046
  good            1.043
  data            0.080

Nearest neighbours of 'good' in affect-augmented space:
  good            1.000
  excellent       0.996
  joyful          0.982
  happy           0.972
  data            0.466

Opposite-affect neighbours of 'good':
  unhappy         0.600
  bad             0.442
  neutral         0.416
  data            0.355
  terrible        0.344

Certificate for 'good': R=7.791e+00, θ=90.0°


--- Demo Finished ---
The original command-line interface is still available in the `_cli()` function if you wish to use it with your own files.
```

## Using the Command-Line Interface (CLI)

For use with your own data, the script provides a command-line interface. You will need:
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
python empathic_embeddings.py path/to/your/glove.6B.100d.txt --probe king
```

### Example: Supervised Mode

This command will use a VAD lexicon to train the model in supervised mode.

```bash
python empathic_embeddings.py path/to/your/glove.6B.100d.txt --vad path/to/your/vad-lexicon.txt --probe happy
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
