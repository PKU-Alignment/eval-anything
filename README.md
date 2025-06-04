[![Organization](https://img.shields.io/badge/Organization-PKU--Alignment-blue)](https://github.com/PKU-Alignment) [![License](https://img.shields.io/github/license/PKU-Alignment/eval-anything?label=license)](#license)

# Introduction

**Eval-anything** aims to track the performance of all modality large models (any-to-any models) on safety tasks and evaluate their true capabilities.

* **Datasets**

  - **Self-developed Dataset**: A dataset specifically designed for assessing all-modality safety of large models.

  - **Integration of Over 50 Open-source Datasets**: Diverse data sources for comprehensive safety assessment.

  - **Five Core Evaluation Dimensions** with 35 sub-dimensions.

- **Embodied Safety Evaluation Framework:**
  - **Covering Various Modality Evaluations**: Text, image, video, speech, and action.
  - **Defining Major Task Categories in Embodied Safety**: Corner cases, blind spots, fragile collections, critical points, and dangerous equipment.
  - **Proposing Major Goals of Embodied Safety Evaluation**: Execution safety, long-range trajectory safety, and hardware safety.

* **Platform Integration**
  - Eval-anything seamlessly integrates with [FlagEval](https://flageval.baai.ac.cn/) to enhance assessment effectiveness.

# Contents

[Introduction](#Introduction)

[Contents](#Contents)

[Tasks and Datasets](#Tasks and Datasets)

[Quick Start](#Quick Start)

[Contributing a New Bench](#Contributing a New Bench)

[Reporting Issues](#Reporting Issues)

[Acknowledgements](#Acknowledgements)



# Tasks and Datasets

# Quick Start

* Step1: Install `eval-anything` by:

  ```bash
  pip install -e .
  conda create -n eval-anything python==3.11
  ```

* Step2: Set up [configuration files](eval_anything/configs/evaluate.yaml).

* Step3: Run the evaluation task by:

  ```bash
  bash scripts/run.sh
  ```

# Contributing a New Bench

We are accepting PRs for new benchmarks. Please read this section carefully before you contribute your benchmark.

## Project Brief

This subsection introduces core frameworks and interfaces in this project and how developers are advised to utilize them.

### Pipeline

- `eval_anything/pipeline/base_task.py` provides the entire process from reading the evaluation parameters to saving the evaluation results. Do not change this file.

### Benchmark

- `eval_anything/benchmark/text-to-text/gsm8k/eval.py` inherits from the `T2TBenchmark` base class in `t2t_benchmark.py`. The code adapts to text-to-text benchmark evaluation processes, including reading benchmark configurations, loading datasets, running model inference and evaluation, and saving results. You can rewrite some functions in this file.

- `eval_anything/benchmark/text-to-text/gsm8k/configs.yaml` contains various benchmark configurations.

### Dataloader

- `eval_anything/dataloader/t2t_dataloader.py` inherits from the `BaseDataLoader` base class in  `base_dataloader.py`. It implements the reading and splicing of text-to-text evaluation data.
  - `eval_anything/dataloader/mm_dataloader.py`: inherits from the `BaseDataLoader` base class in `base_dataloder.py`.  It implements the reading and splicing of multimodal evaluation data.

- `eval_anything/dataloader/format_mm_dataset.py` implements the preprocessing of multimodal datasets for subsequent data loading, given the high complexity of multimodal data.

### Inference

- `eval_anything/models`: Various backend inferences are implemented under this path.

### Evaluation

- `eval_anything/evaluate_tools/metrics.py` contains commonly used metric calculation. We recommend you add your new metrics here rather than in the benchmark separately.
- `eval_anything/evaluate_tools/t2t_tools.py` integrates common evaluation tools such as regular expression matching and judgment of similarity, etc. New tools should be integrated in an object-oriented manner and considering the versatility of the tools.

### Utils

- `utils.py`: Common toolkits for project development.

- `data_type.py`: Common data types for data transfer between different modules

- `cache_manager.py`: Cache mechanisms

- `regiser.py`: Commonly used registry（template, metric, benchmark...）

- `logger.py`: Log mechanisms

- `mm_data_manager`: The format conversion and conversation compilation of various modal data

## General Guidelines

1. Create a directory under the `benchmarks` folder
   - Write a configuration file `configs.yaml`. Here is an [example](eval_anything/benchmarks/text_to_text/gsm8k/configs.yaml).
   - Develop `eval.py` after inheriting from the corresponding benchmark base class.
2. (Optional) Create `metrics.py` and `tools.py` and make registers for new metrics and evaluation tools. Here is an [example](eval-anything/eval_anything/benchmarks/text_to_text/TruthfulQA).
3. Configure your evaluation tasks in `eval_anything/configs/evaluate.yaml`.
4. Run `eval_anything/scripts/run.sh` to test your benchmark.

## Configuring Any-to-text Datasets

When developing multimodal datasets, you may encounter difficulty to unify the loading methods of multimodal data. Please follow the steps below for dataset preprocessing:

1. Register your dataset in `dataloader/format_mm_dataset.py`
2. Process benchmark-specific key-value pairs and construct prompts in `dataloader/format_mm_dataset.py` with predefined PromptBuilders. You may customize your own PromptBuilder in the `_to_InferenceInput` method if necessary.
3. Convert images, audios and videos to `base64` with `utility classes` in `utils/mm_data_manager.py` for conversation embedding.
4. Use the utility classes in `utils/mm_data_manager.py` to customize the form of conversation for this  task. For example, `ImageManager.prompt_to_conversation` implements the conversation construction for common graphic and text inputs.
5. The inference and evaluation part is the same as introduced in [General Guidelines](##General Guidelines)

## Developing VLA Benchmarks

1. Configuring `objaverse`

```bash
python -m objathor.dataset.download_annotations --version 2023_07_28 --path /path/to/objaverse_assets
python -m objathor.dataset.download_assets --version 2023_07_28 --path /path/to/objaverse_assets
```

2. Configuring `house`

```bash
python scripts/download_objaverse_houses.py --save_dir /path/to/objaverse_houses --subset val
```

or

```bash
python scripts/download_objaverse_houses.py --save_dir /path/to/objaverse_houses --subset train
```

3. Downloading Datasets

```bash
python scripts/download_dataset.py --save_dir /path/to/dataset
```

4. Configuring Environments

```bash
pip install -e .[vla]
pip install --extra-index-url https://ai2thor-pypi.allenai.org ai2thor==0+966bd7758586e05d18f6181f459c0e90ba318bec
pip install -e "git+https://github.com/allenai/allenact.git@d055fc9d4533f086e0340fe0a838ed42c28d932e#egg=allenact&subdirectory=allenact" --no-deps
pip install -e "git+https://github.com/allenai/allenact.git@d055fc9d4533f086e0340fe0a838ed42c28d932e#egg=allenact_plugins[all]&subdirectory=allenact_plugins" --no-deps
```

5. Running tasks

```bash
bash scripts/run_vla.sh
```

## Pre-commit

This project uses pre-commit to ensure code quality and consistency. Pre-commit will automatically run the configured inspection tool before each commit.

### Installation

1. Install pre-commit

```bash
pip install pre-commit
```

2. Initialize pre-commit in the root directory

```bash
pre-commit install
```

### Usage

Before each code submission, pre-commit will run automatically. It is also possible to run the check manually:

```bash
# check all files
pre-commit run --all-files

# check temporary files
pre-commit run
```

If there are unsolved problems, pre-commit will prevent the commit and display error messages. After fixing the problem, re-add the file and submit it:

```bash
git add .
git commit -m "MESSAGE"
git push
```

To skip the check and commit directly:

```bash
git commit -m "MESSAGE" --no-verify
```

### Inspection tools

The following inspection tools are configured for this project:

- **pre-commit-hooks**: Basic file checks (line end Spaces, file endings, YAML/TOML formats, large file checks, etc.)
- **isort**: Sorting Python import statements
- **black**: Python code formatting
- **pyupgrade**: Upgrade the Python syntax to a later version
- **autoflake**: Remove unused imports and variables
- **codespell**: spell check

See `.pre-commit-config.yaml` for detailed configuration.

## Tips

- Please follow *Occam's Razor* principle in development and minimize the rewriting of existing functions
- please add new dependency libraries needed in `pyproject.toml`.
- We welcome any suggestions regarding current base classes!

# Reporting Issues

If you have any questions in the process of using align-anything, don't hesitate to ask your questions on [the GitHub issue page](https://github.com/PKU-Alignment/eval-anything/issues/new/choose), we will reply to you in 2-3 working days.

# Acknowledgements

This repository benefits from multiple open-source projects. Thanks for their wonderful works and their efforts for promoting the LLM research. 

This work is supported by the Beijing Academy of Artificial Intelligence, Peking University and Beijing University of Posts and Telecommunications.

