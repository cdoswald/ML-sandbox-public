PYTHON = python3

.PHONY: compile lint format checks

compile:
	$(PYTHON) -m py_compile src/*.py

lint:
	$(PYTHON) -m ruff check src/ --fix

format:
	$(PYTHON) -m ruff format src/

checks: format compile lint 
