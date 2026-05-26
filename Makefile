PYTHON ?= python3

.PHONY: run corpus evaluate

run: corpus evaluate

corpus:
	$(PYTHON) src/build_corpus.py

evaluate:
	$(PYTHON) src/evaluate.py
