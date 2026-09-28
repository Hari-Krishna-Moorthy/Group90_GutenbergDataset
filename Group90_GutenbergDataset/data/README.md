# Dataset note

This project does not store a downloaded book or any external dataset.

The notebook obtains the Gutenberg Corpus through `nltk.corpus.gutenberg`. Its setup cell calls `nltk.download("gutenberg")` (and the required tokenizer resources) only when those resources are not already installed. NLTK manages its own local data cache.

