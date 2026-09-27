# Nicholson

Four-patch Nicholson system with normalized parameter inputs.

The manuscript forward implementation is `four_methods_normalized_sensitivity_5seeds/`. Start from the repository root:

```bash
python run.py smoke nicholson
python run.py train nicholson --device cuda:0 --cpus 16 --output-dir outputs/nicholson
```

See [reproduction](../../docs/REPRODUCTION.md) and the [paper map](../../docs/PAPER_MAP.md) for the exact inverse/application protocol and historical variants. Tests are included in this project and its selected variant directories.

Within the selected variant, `code/equation.py` defines the DDE and reference solver, `code/data.py` generates datasets, `code/model.py` defines the neural operators, `code/training.py` implements training, and `code/scripts/pipeline.py` is the original entry point.
