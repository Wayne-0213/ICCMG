# ICCMG

**ICCMG** is the official implementation of **A Classification-Aware In-Context Learning Framework for Commit Message Generation**.

ICCMG aims to generate high-quality commit messages from code changes by incorporating commit-category information into in-context learning. Instead of using a uniform generation strategy for all commits, ICCMG first distinguishes different maintenance intents and then retrieves category-specific demonstrations to guide large language models in generating more accurate and targeted commit messages.

## Overview

Commit messages help developers understand what has changed in a software project and why the change was made. However, real-world commit messages are often missing, too short, or low-quality, which can reduce project maintainability.

ICCMG addresses this problem with a classification-aware framework for commit message generation. The framework consists of three main stages:

1. **Data Preparation**
   The raw dataset is cleaned and filtered to retain high-quality commit messages that contain both “What” and “Why” information. Commits are then categorized into three maintenance types:

   * `perfective`: code improvement, refactoring, or performance enhancement
   * `adaptive`: feature addition or adaptation to a new environment
   * `corrective`: bug fixing or fault correction

2. **Prompt Construction**
   For each target code change, ICCMG retrieves similar examples from the same commit category. The prompt includes:

   * a task description
   * category-specific maintenance information
   * retrieved in-context examples
   * code identifiers extracted from code changes
   * the target code diff

3. **Commit Message Generation**
   The constructed prompt is sent to a large language model to generate the final commit message.

## Repository Structure

```text
ICCMG/
├── ICCMG_RQ1/              # Experiments for comparing ICCMG with baseline methods
├── ICCMG_RQ2/              # Ablation experiments
├── ICCMG_RQ3/              # Experiments related to category-aware settings
├── ICCMG_RQ4/              # Retrieval strategy experiments
├── ICCMG_RQ5/              # Demonstration-number experiments and plotting scripts
├── dataset/
│   └── Java/               # Java commit-message-generation dataset
├── utils.py                # Utility functions, OpenAI API calls, token counting, and retrieval helpers
└── README.md
```

## Dataset

The dataset is stored under:

```text
dataset/Java/
```

The main files include:

```text
train.json
test.json
train_0.json
train_1.json
train_2.json
test_0.json
test_1.json
test_2.json
valid_0.json
valid_1.json
valid_2.json
train_id_type3_0.txt
train_id_type3_1.txt
train_id_type3_2.txt
test_id_type3_0.txt
test_id_type3_1.txt
test_id_type3_2.txt
```

The category index is defined as follows:

| Index | Category     | Description                                                                |
| ----- | ------------ | -------------------------------------------------------------------------- |
| `0`   | `perfective` | Improvements such as refactoring, optimization, or performance enhancement |
| `1`   | `adaptive`   | Feature additions or changes for adapting to a new environment             |
| `2`   | `corrective` | Bug fixes and fault corrections                                            |

The identifier files contain code identifier information corresponding to each commit sample.

## Environment

The experiments were conducted with:

```text
Python 3.9
```

A minimal Python environment can be prepared with:

```bash
pip install openai tiktoken tqdm rank-bm25 nltk numpy
```

Depending on the experiment script you run, additional packages may be required.

## API Configuration

ICCMG uses the OpenAI API for commit message generation. Before running the scripts, set your API key in `utils.py`:

```python
openai.api_key = "YOUR_API_KEY"
```

You can also configure a custom API base URL through the environment variable:

```bash
export OPENAI_API_BASE="https://api.openai.com/v1"
```

Please do not commit your API key to the repository.

## Running Experiments

Each research-question folder contains the corresponding experiment scripts and result files.

For example, to run the RQ1 experiments:

```bash
cd ICCMG_RQ1
python RQ1_0.py
python RQ1_1.py
python RQ1_2.py
```

The scripts should be executed from inside their corresponding experiment folders because they use relative paths to access the dataset.

Generated outputs are saved under the `Result/` directory of each experiment folder.

## Research Questions

This repository contains code and results for the following experimental settings:

### RQ1: Comparison with Baselines

Evaluates ICCMG against existing commit message generation methods.

Main folder:

```text
ICCMG_RQ1/
```

### RQ2: Ablation Study

Evaluates the contribution of major components, including category information, code identifiers, and “What & Why” guidance.

Main folder:

```text
ICCMG_RQ2/
```

### RQ3–RQ5: Retrieval and Demonstration Settings

These folders contain additional experiments related to category-aware generation, retrieval strategies, and the number of retrieved demonstrations.

Main folders:

```text
ICCMG_RQ3/
ICCMG_RQ4/
ICCMG_RQ5/
```

## Results

ICCMG consistently improves commit message generation quality across three commit categories: perfective, adaptive, and corrective.

The experimental results show that classification-aware prompt construction and category-specific demonstration retrieval help large language models better capture the intent of code changes and generate more accurate commit messages.

Detailed results can be found in the `Result/` directory of each experiment folder.

