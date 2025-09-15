# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### `fl-manager-core`

#### Added

- Federated data simulation using client site number.
- `FLRunner` based on JSON config file.
- Executor based on `nvflare` ClientAPI, supporting tasks: _train_, _validate_, _submit_model_, _export_model_ and _fit_and_export_model_.
- Dataflows based on `sf-hamilton` directed acyclic graphs.
- Meta registry with main components.
- Core components: `readers`, `distributors`, `splitters`, `formatters`, `checkers`, `validators`, `preprocessors`, `datasets` and `models`.

### `fl-manager-lightning`

#### Added

- Support for **Ditto** federated algorithm with custom executor.
- Support for **FedAvg**, **FedBN**, **FedOpt** and **FedProx** federated algorithms.
- Support for _stateless_ (each round, new trainer) and _stateful_ (each round, same trainer) training.
- Dataflow for training with `lightning`.
- Custom `callbacks` and `datamodule` components.

### `fl-manager-utils-job-generator`

#### Added

- Client components customization for `readers` and `formatters`.
- Support to generate jobs for **FedAvg**, **FedBN**, **FedOpt** and **FedProx** federated algorithms.
- `nvflare` job config generator based on `fl-manager` components.

### `fl-manager-utils-nvflare-extensions`

#### Added

- Weight initializer executor for `pytorch`.
- Workflows for model export and personalized model export (personalized federated learning).

### `fl-manager-components-readers-huggingface`

#### Added

- Reader for huggingface `datasets` library with subset selection.

### `fl-manager-components-preprocessors-torchvision`

#### Added

- Preprocessor for bytes to tensor conversion.
- Preprocessor for `PIL.Image` to tensor conversion.
- Preprocessor for tensor normalization.

### `fl-manager-components-models-lightning`

#### Added

- MNIST based models (for demo purposes).
- TinyBert classifier model (for demo purposes).

### `fl-manager-components-formatters-pillow`

#### Added

- Formatter for image path to bytes conversion.

### `fl-manager-components-datasets-torch`

#### Added

- Torch dataset wrapper for `fl-manager` DataFrame dataset component. Supports transforms with the preprocessor component.
