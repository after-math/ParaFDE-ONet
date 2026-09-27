# Cstr

Delayed-cooling CSTR with frozen LM and operating-point search.

The manuscript forward implementation is `./`. Start from the repository root:

```bash
python run.py smoke cstr
python run.py train cstr --device cuda:0 --cpus 16 --output-dir outputs/cstr
```

See [reproduction](../../docs/REPRODUCTION.md) and the [paper map](../../docs/PAPER_MAP.md) for the exact inverse/application protocol and historical variants. Tests are included in this project and its selected variant directories.

`cstr_forward/` contains the equation, data, model and training; `cstr_pilot/` provides cached inference and LM; `cstr_industrial/` implements event-wise recalibration and operating-point selection. Archived results are under `results/paper/cstr/` at repository level.
