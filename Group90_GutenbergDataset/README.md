# Group 90 - N-Gram Language Modeling with Unknown Words

This repository contains the submission for **Problem Statement 9: N-gram Language Modeling with Unknown Words**. The project uses only the Gutenberg Corpus distributed through NLTK and implements a beginner-friendly trigram language model experiment.

## Team members

| Member | Name | Student ID |
|---|---|---|
| Member 1 | HARI KRISHNA MOORTHY T | 2025ag05393 |
| Member 2 | SHREEJA MAHAPATRA | 2025ag05424 |
| Member 3 | NAVEEN RANJOLKAR | 2025ag0539 |
| Member 4 | SOMVANSHI PRASAD DIGAMBAR | 2025ag05285 |

## Assignment details

| Field | Value |
|---|---|
| Group number | 90 |
| Problem statement | 9 - N-gram Language Modeling with Unknown Words |
| Dataset | NLTK Gutenberg Corpus |
| Corpus scope | All 18 books available in NLTK Gutenberg |
| Language model | Trigram model |
| Rare-word threshold | Training frequency `< 3` |
| Unknown-word token | `<UNK>` |
| Data split | Reproducible random 80% training and 20% testing split by sentence |

## Project objective

The project investigates how a trigram language model behaves when it encounters rare or previously unseen words. It covers the following objectives:

- inspect the books available in the NLTK Gutenberg Corpus;
- load and preprocess all available Gutenberg books;
- analyze token and vocabulary frequencies;
- build a trigram language model;
- replace training words occurring fewer than three times with `<UNK>`;
- predict the next word from a preceding context;
- predict a word hidden inside a sentence;
- compare models with and without unknown-word handling; and
- evaluate accuracy, OOV rate, coverage, and perplexity.

## Dataset

The notebook uses all 18 books returned by `nltk.corpus.gutenberg.fileids()`. Their sentences and tokens are combined into one corpus while sentence boundaries are preserved.

No external dataset or manually downloaded Project Gutenberg text is used. The official NLTK packages are included locally under `data/nltk_data/`:

- `gutenberg`
- `punkt`
- `punkt_tab`

The notebook adds this folder to `nltk.data.path` before loading the corpus. If any package is missing, the setup cell downloads it into the same local folder.

## Text preprocessing

The preprocessing pipeline:

1. preserves sentence boundaries supplied by NLTK;
2. converts words to lowercase;
3. retains alphabetic words and words containing internal apostrophes;
4. removes standalone punctuation and unwanted tokens;
5. adds two `<START>` tokens and one `<END>` token when counting trigrams; and
6. determines rare words using training data only.

Complete sentences are randomly shuffled using the notebook's fixed seed and then divided into 80% training and 20% testing. The split is performed before learning the `<UNK>` vocabulary, which prevents test information from leaking into the model while keeping sentence boundaries intact.

## Trigram model

The model estimates the probability of a word from the two words immediately before it:

```text
P(current word | previous word 1, previous word 2)
    = trigram count / two-word context count
```

For example, the model uses the words `she was` to estimate a likely next word.

The notebook maintains reusable functions for trigram and context counting, probability calculation, next-word prediction, missing-word prediction, test-token mapping, and test-set perplexity.

## Unknown-word handling

Training words with a frequency below three are replaced by `<UNK>`. During testing, a word outside the learned `<UNK>` model vocabulary is mapped to `<UNK>` only when it is supplied to that model. The original test sentences are retained for display and evaluation.

The notebook distinguishes between:

- **rare words:** words with low frequency in the training data;
- **unknown words:** words unavailable to the model at prediction time; and
- **OOV words:** words outside the model's training vocabulary.

## Model configurations

| Model | `<UNK>` handling |
|---|---|
| 1 | No |
| 2 | Yes |

Both models use direct trigram maximum-likelihood probabilities. An unseen trigram receives probability zero.

## Prediction tasks

### Next-word prediction

Given a word sequence, the model predicts the most likely next word based on the final two context words.

### Missing-word prediction

One word is hidden inside a sentence. Candidate words are scored using the local trigram probabilities before and after the blank. Log probabilities are used to avoid numerical underflow.

The diagnostic test cases are derived from held-out Gutenberg sentences and include known-word contexts, observed continuations, unseen contexts, naturally occurring OOV words, and deliberately inserted unseen words such as `spaceship` or `quantum`.

The notebook contains 24 next-word cases and 20 missing-word cases. These diagnostic cases provide understandable success and failure examples. Perplexity is calculated over the full held-out test set for a broader evaluation.

## Evaluation metrics

The following metrics are calculated during notebook execution:

- **Next-word accuracy:** correctly predicted next words divided by next-word cases.
- **Missing-word accuracy:** correctly predicted hidden words divided by missing-word cases.
- **OOV rate:** test tokens absent from the original training vocabulary divided by all test tokens.
- **Coverage:** test tokens present in the original training vocabulary divided by all test tokens.
- **Perplexity:** the model's average uncertainty over the complete held-out test set; lower finite values indicate less surprise.

Infinite perplexity is reported explicitly when any evaluated event receives probability zero.

## Tables and visualizations

The notebook includes pandas tables for corpus statistics, frequency statistics, vocabulary size, generated test cases, predictions, the final model comparison, and sample error analysis.

It also includes seven required visualizations:

1. top 20 word frequencies;
2. word-frequency distribution;
3. rare-word frequency distribution;
4. vocabulary size before and after `<UNK>`;
5. known versus OOV test-token coverage;
6. prediction accuracy comparison; and
7. perplexity comparison.

## Project structure

```text
Group90_GutenbergDataset/
├── Group90_GutenbergDataset.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── data/
    ├── README.md
    └── nltk_data/
        ├── corpora/
        │   └── gutenberg.zip
        └── tokenizers/
            ├── punkt.zip
            └── punkt_tab.zip
```

The `data/README.md` file documents the origin and purpose of the local NLTK resources.

## Requirements

Python 3.10 or newer is recommended. The project uses NLTK, NumPy, pandas, Matplotlib, and Jupyter.

```bash
python -m pip install -r requirements.txt
```

## Running the notebook

From the `Group90_GutenbergDataset` directory:

### macOS or Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook Group90_GutenbergDataset.ipynb
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter notebook Group90_GutenbergDataset.ipynb
```

After opening the notebook, select **Kernel → Restart & Run All**. The included local NLTK resources allow the corpus to load without downloading it again. Internet access is needed only if one of those local package files is removed.

## Makefile commands

The included `Makefile` provides an end-to-end workflow. Run these commands from the `Group90_GutenbergDataset` directory:

```bash
make help          # List all available commands
make setup         # Create .env and install requirements
make data          # Download/verify local NLTK resources
make run           # Execute the notebook and save its outputs
make validate      # Check for unexecuted or failed code cells
make export        # Create .py, .html, and .pdf exports
make all           # Run the complete workflow from setup through export
```

Individual export commands are also available:

```bash
make export-py
make export-html
make export-pdf
```

Exports are written to:

```text
exports/
├── Group90_GutenbergDataset.py
├── Group90_GutenbergDataset.html
└── Group90_GutenbergDataset.pdf
```

The PDF command uses Google Chrome in headless mode to print the self-contained HTML export, including notebook tables and plots. If Chrome is installed elsewhere, provide its path:

```bash
make export-pdf CHROME=/path/to/google-chrome
```

## Reproducibility and correctness

- A fixed random seed is defined in the notebook.
- Sentence boundaries are preserved during splitting.
- Rare words are learned from training data only.
- Original test sentences are not permanently changed.
- Metrics and predictions are calculated by code rather than hard-coded.
- Zero-probability events are reported rather than hidden.
- No external dataset is used.

## Git workflow

The repository was developed using small commits covering project setup, preprocessing, exploratory analysis, trigram implementation, unknown-word handling, prediction cases, evaluation, visualizations, fixes, and documentation.

```bash
git log --oneline --decorate
```

The `.git` directory is retained in the development workspace but is normally omitted from an academic submission ZIP unless the instructor explicitly requests repository metadata.

## Submission

The expected submission archive is:

```text
Group90_GutenbergDataset.zip
```

It contains the project files and the local official NLTK resource packages. Temporary environments, notebook checkpoints, and unrelated generated files are excluded.

## References

1. [NLTK Book — Accessing Text Corpora and Lexical Resources](https://www.nltk.org/book/ch02.html)
2. [NLTK Corpus HOWTO](https://www.nltk.org/howto/corpus.html)
3. Daniel Jurafsky and James H. Martin, [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/), chapter on N-gram language models
