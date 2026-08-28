# Project structure

This project contains related but distinct implementations and experiments.
Keeping the model backend, dataset, and artifact type visible in each path is
important because their outputs are not interchangeable.

## Source directories

```text
high_strain_CNN/
  tensorflow_reference/       Author TensorFlow reference implementation
  pytorch_autophasenn/        PyTorch port and AutoPhaseNN workflow
  simulation/                 Shared synthetic-data generation and inference
  tools/                      Conversion, parity, and data-validation commands
  configs/                    Versioned data and simulation configuration
  requirements/               Backend-specific dependency files
  tests/                      Fast source-level tests
  artifacts/                  Training, evaluation, model, and figure outputs
```

### TensorFlow reference

`tensorflow_reference/train_upstream.py` is the vendored author training code.
It expects author-style `.npz` files containing `I` and `phi`, and it defines
the published TensorFlow architecture and WCA training objective. Keep this
file close to upstream so that it remains a trustworthy reference.

The official `model_paper.h5` belongs in `artifacts/models/`. Future evaluation
on AutoPhaseNN data should be added as a separate TensorFlow adapter under
`tensorflow_reference/`, without folding adapter logic into the upstream file.

### PyTorch and AutoPhaseNN

`pytorch_autophasenn/` is one complete workflow:

```text
AutoPhaseNN memmaps
  -> data.py
  -> model.py + losses.py
  -> train.py
  -> checkpoint
  -> evaluate.py
  -> reconstruction.py
  -> shared AutoPhaseNN post-processing and metrics
  -> visualize.py
```

The evaluation records currently checked into this project were all produced
by this workflow. They are stored under
`artifacts/evaluations/autophasenn_pytorch/`; they are not evaluations of the
official TensorFlow H5 model.

### Simulation

`simulation/` is backend-neutral. It generates author-compatible `.npz`
samples and can send the same sample through either the official TensorFlow H5
model or the converted PyTorch `published` model:

```text
configs/simulation_paper.json
  -> simulation/generate_dataset.py
  -> sample_*.npz
  -> simulation/run_paper_model.py
  -> reciprocal phase, complex real-space reconstruction, metrics, figures
```

This path is also the right place to compare TensorFlow and PyTorch on exactly
the same input before introducing AutoPhaseNN-specific post-processing.

### Tools

- `convert_keras_weights.py`: TensorFlow H5 to PyTorch checkpoint.
- `export_tensorflow_reference.py`: deterministic TensorFlow tensors.
- `verify_pytorch_parity.py`: numerical comparison with converted PyTorch.
- `validate_data.py`: AutoPhaseNN memmap schema and finite-value validation.

## Artifact directories

```text
artifacts/
  training/pytorch_autophasenn/         tracked lightweight run records
  evaluations/autophasenn_pytorch/      tracked PyTorch evaluation tables/logs
  evaluations/autophasenn_tensorflow/   reserved for future TF evaluation
  models/                               local H5/PT weights, ignored
  parity/                               generated parity tensors, ignored
  simulation/                           generated datasets/results, ignored
  visualizations/autophasenn_pytorch/   generated PyTorch figures, ignored
  visualizations/autophasenn_tensorflow/ reserved for future TF figures
```

Large checkpoints remain in the configured external checkpoint root. A run
manifest stores their paths so the lightweight Git record remains traceable.
Historical manifests and logs retain their original absolute server paths.

## Planned TensorFlow evaluation

The next addition should implement an AutoPhaseNN input adapter and TensorFlow
inference entry point, then reuse common reconstruction, post-processing, and
metric definitions. Recommended files and outputs are:

```text
tensorflow_reference/evaluate_autophasenn.py
tensorflow_reference/visualize_autophasenn.py
artifacts/evaluations/autophasenn_tensorflow/<run-name>/
artifacts/visualizations/autophasenn_tensorflow/<run-name>/
```

The TensorFlow and PyTorch reports should record at least: backend, model
variant, weight source, dataset version, input preprocessing, ambiguity mode,
support threshold, Git commit, and evaluator version. This makes comparisons
meaningful without forcing both models to use the same support threshold.

## Naming rule

Use `<dataset>_<backend>` in artifact namespaces, and include the model variant
and important evaluation condition in the run name. Do not place generated
weights, datasets, or images beside source files.
