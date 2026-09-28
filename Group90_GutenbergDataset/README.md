# Group 90 - N-Gram Language Modeling with Unknown Words

This project is a reproducible NLP assignment using one book from the NLTK Gutenberg Corpus. The executed notebook builds and compares trigram language models with and without `<UNK>` handling and with no smoothing or Laplace smoothing.

## Assignment details

- **Group:** 90
- **Problem statement:** 9 - N-gram Language Modeling with Unknown Words
- **Dataset:** NLTK Gutenberg Corpus only
- **Selected book:** Chosen and justified inside the notebook

## Project structure

```text
Group90_GutenbergDataset/
├── Group90_GutenbergDataset.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── data/
    └── README.md
```

## Setup and execution

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
jupyter notebook Group90_GutenbergDataset.ipynb
```

In Jupyter, choose **Kernel → Restart & Run All**. The setup cell checks for the required NLTK resources and downloads them when absent. Internet access is needed only for that first NLTK download. No external dataset is used.

## Experiments

The notebook uses an 80/20 sentence-level split, builds four trigram configurations, creates real-corpus and deliberately OOV test cases, evaluates next-word and missing-word prediction, and reports accuracy, coverage, OOV rate, and perplexity. All randomness uses a fixed seed.

## Git workflow

The repository uses small commits for project setup, preprocessing/EDA, model development, unknown-word handling, prediction/evaluation, and final documentation. Run `git log --oneline --decorate` to inspect the history.

