"""Typer-based command line interface."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from . import api, io, utils

app = typer.Typer(add_completion=False)


@app.command()
def augment(vec_path: Path, vad: Optional[Path] = None, out: Path = Path("model.npz")) -> None:
    """Train an empathic embedding model and save it."""

    vectors = io.read_vec(vec_path)
    if vad:
        lex = io.load_vad(vad)
        model = api.train_supervised(vectors, lex)
    else:
        model = api.train_unsupervised(vectors)
    model.save(out)
    typer.echo(f"saved model to {out}")


@app.command()
def query(path: Path, word: str, topk: int = 5) -> None:
    """Query nearest neighbours of ``word`` from a saved model."""

    model = api.load(path)
    for w, s in model.nearest_neighbors(word, k=topk):
        typer.echo(f"{w}\t{s:.3f}")


@app.command()
def opposites(path: Path, word: str, topk: int = 5) -> None:
    """Query opposite-affect neighbours."""

    model = api.load(path)
    for w, s in model.opposite_neighbors(word, k=topk):
        typer.echo(f"{w}\t{s:.3f}")


@app.command()
def phrase(path: Path, text: str) -> None:
    """Compute a phrase vector."""

    model = api.load(path)
    vec = model.phrase_vector(text)
    typer.echo(" ".join(f"{x:.4f}" for x in vec))


if __name__ == "__main__":
    app()
