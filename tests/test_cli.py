from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from empathic_embeddings import cli


runner = CliRunner()


def write_vec(path: Path, vectors: dict[str, np.ndarray]) -> None:
    dim = len(next(iter(vectors.values())))
    with open(path, "w", encoding="utf-8") as f:
        for w, v in vectors.items():
            f.write(w + " " + " ".join(str(x) for x in v) + "\n")


def test_cli_roundtrip(tmp_path, toy_vectors):
    vec_path = tmp_path / "toy.vec"
    write_vec(vec_path, toy_vectors)
    vad_path = tmp_path / "vad.txt"
    with open(vad_path, "w") as fh:
        fh.write("good 1 0 0\n")
        fh.write("bad 0 0 0\n")
    model_path = tmp_path / "model.npz"
    res = runner.invoke(
        cli.app, ["augment", str(vec_path), "--vad", str(vad_path), "--out", str(model_path)]
    )
    assert res.exit_code == 0
    assert model_path.exists()
    out = runner.invoke(cli.app, ["query", str(model_path), "good", "--topk", "1"])
    assert out.exit_code == 0
