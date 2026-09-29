# End-to-end commands for the Group 90 NLP assignment.
# Run `make help` to see the available targets.

NOTEBOOK := Group90_GutenbergDataset.ipynb
NOTEBOOK_STEM := Group90_GutenbergDataset
VENV := .env
PYTHON := $(VENV)/bin/python
NBCONVERT := $(VENV)/bin/jupyter-nbconvert
SETUP_STAMP := $(VENV)/.setup-complete
EXPORT_DIR := exports
LOCAL_NLTK_DATA := data/nltk_data
PROJECT_DIR := $(notdir $(CURDIR))
ZIP_ARCHIVE := ../$(PROJECT_DIR).zip
PACKAGE_FILES := \
	$(NOTEBOOK) \
	README.md \
	requirements.txt \
	.gitignore \
	Makefile \
	scripts/prepare_print_html.py \
	$(EXPORT_DIR)/$(NOTEBOOK_STEM).py \
	$(EXPORT_DIR)/$(NOTEBOOK_STEM).html \
	$(EXPORT_DIR)/$(NOTEBOOK_STEM).pdf \
	data/README.md \
	data/nltk_data/corpora/gutenberg.zip \
	data/nltk_data/tokenizers/punkt.zip \
	data/nltk_data/tokenizers/punkt_tab.zip

# Override this when Chrome is installed at a different path:
# make export-pdf CHROME=/path/to/google-chrome
CHROME ?= /Applications/Google Chrome.app/Contents/MacOS/Google Chrome

.PHONY: help all setup data run validate export export-files export-py export-html export-pdf package clean-exports

help:
	@echo "Available commands:"
	@echo "  make setup        Create .env and install Python requirements"
	@echo "  make data         Download NLTK resources into data/nltk_data"
	@echo "  make run          Execute every notebook cell and save outputs"
	@echo "  make validate     Check that all code cells ran without errors"
	@echo "  make export-py    Export the notebook to exports/*.py"
	@echo "  make export-html  Export the notebook to exports/*.html"
	@echo "  make export-pdf   Export HTML and print it to exports/*.pdf"
	@echo "  make package      Create the exports and submission ZIP"
	@echo "  make export       Create the PY, HTML, PDF, and submission ZIP"
	@echo "  make all          Setup, download data, run, validate, and export"
	@echo "  make clean-exports Remove only generated files from exports/"

all: setup data run validate export

# Create the environment once and reinstall only when requirements.txt changes.
setup:
	@if [ ! -x "$(PYTHON)" ]; then python3 -m venv $(VENV); fi
	@if [ ! -f "$(SETUP_STAMP)" ] || [ requirements.txt -nt "$(SETUP_STAMP)" ]; then \
		$(PYTHON) -m pip install --upgrade pip && \
		$(PYTHON) -m pip install -r requirements.txt && \
		touch "$(SETUP_STAMP)"; \
	fi
	@echo "Python environment is ready: $(PYTHON)"

# Keep the official NLTK packages inside the project for reproducible runs.
data: setup
	@mkdir -p $(LOCAL_NLTK_DATA)
	@if [ -f "$(LOCAL_NLTK_DATA)/corpora/gutenberg.zip" ] && \
		[ -f "$(LOCAL_NLTK_DATA)/tokenizers/punkt.zip" ] && \
		[ -f "$(LOCAL_NLTK_DATA)/tokenizers/punkt_tab.zip" ]; then \
		echo "Local NLTK resources are already available."; \
	else \
		$(PYTHON) -m nltk.downloader -d $(LOCAL_NLTK_DATA) gutenberg punkt punkt_tab; \
	fi

# Execute the notebook in place so tables, metrics, and plots are saved.
run: setup data
	$(NBCONVERT) \
		--to notebook \
		--execute \
		--inplace \
		--ExecutePreprocessor.timeout=600 \
		$(NOTEBOOK)

# Fail when a code cell is unexecuted or contains a saved exception.
validate: setup
	$(PYTHON) -c 'import nbformat; n=nbformat.read("$(NOTEBOOK)", 4); c=[x for x in n.cells if x.cell_type=="code"]; assert all(x.execution_count is not None for x in c), "Unexecuted code cell found"; assert not any(any(o.output_type=="error" for o in x.get("outputs", [])) for x in c), "Saved cell error found"; print(f"Validated {len(c)} executed code cells with zero errors.")'

$(EXPORT_DIR):
	@mkdir -p $(EXPORT_DIR)

export: package

export-files: export-py export-html export-pdf

export-py: setup | $(EXPORT_DIR)
	$(NBCONVERT) \
		--to script \
		--output $(NOTEBOOK_STEM) \
		--output-dir $(EXPORT_DIR) \
		$(NOTEBOOK)

export-html: setup | $(EXPORT_DIR)
	$(NBCONVERT) \
		--to html \
		--output $(NOTEBOOK_STEM).html \
		--output-dir $(EXPORT_DIR) \
		$(NOTEBOOK)
	$(PYTHON) scripts/prepare_print_html.py $(EXPORT_DIR)/$(NOTEBOOK_STEM).html

# Chrome prints the self-contained HTML export to PDF, including tables and plots.
export-pdf: export-html
	@test -x "$(CHROME)" || (echo "Chrome not found. Pass CHROME=/path/to/chrome" && exit 1)
	"$(CHROME)" \
		--headless \
		--disable-gpu \
		--no-sandbox \
		--no-pdf-header-footer \
		--print-to-pdf="$(abspath $(EXPORT_DIR)/$(NOTEBOOK_STEM).pdf)" \
		"file://$(abspath $(EXPORT_DIR)/$(NOTEBOOK_STEM).html)"

# Build a self-contained submission archive after all exports are ready.
# Running zip from the parent directory preserves the project folder in the ZIP.
package: export-files
	@command -v zip >/dev/null || (echo "zip command not found" && exit 1)
	@cd .. && zip -FS -9 "$(PROJECT_DIR).zip" $(addprefix $(PROJECT_DIR)/,$(PACKAGE_FILES))
	@unzip -t "$(ZIP_ARCHIVE)"
	@echo "Submission archive created: $(ZIP_ARCHIVE)"

clean-exports:
	@mkdir -p $(EXPORT_DIR)
	@find $(EXPORT_DIR) -maxdepth 1 -type f \
		\( -name '$(NOTEBOOK_STEM).py' -o -name '$(NOTEBOOK_STEM).html' -o -name '$(NOTEBOOK_STEM).pdf' \) \
		-delete
	@echo "Generated exports removed."
