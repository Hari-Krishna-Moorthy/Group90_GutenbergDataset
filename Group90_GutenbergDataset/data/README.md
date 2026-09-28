# Local NLTK data

This directory contains the official NLTK resource packages required by the notebook:

```text
nltk_data/
├── corpora/
│   └── gutenberg.zip
└── tokenizers/
    ├── punkt.zip
    └── punkt_tab.zip
```

The files were downloaded from the official [`nltk_data`](https://github.com/nltk/nltk_data) repository. The Gutenberg books are accessed only through `nltk.corpus.gutenberg`; no external dataset has been added.

The notebook registers this directory with `nltk.data.path`. If a resource is missing, its setup cell calls `nltk.download(..., download_dir=...)` to restore it here.
