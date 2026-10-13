#!/usr/bin/env python3
"""Package the current list and pipeline for upload from the Colab notebook."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

FILES = (
    "corpus-list.py", "corpus_registry.py", "prepare_corpus.py", "cleaning_rules.json",
    "training_data.py", "train_gpt2.py", "training_config.json", "generate.py",
    "requirements-colab.txt", "README.md",
)


def main():
    root = Path(__file__).resolve().parent
    destination = root / "colab_project.zip"
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(root / name, arcname=name)
    print(f"Created {destination} ({destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
