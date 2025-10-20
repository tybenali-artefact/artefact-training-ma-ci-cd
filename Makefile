SHELL := /bin/bash

PYTHON_VERSION := 3.11

.DEFAULT_GOAL = help

# help: help					- Display this makefile's help information
.PHONY: help
help:
	@grep "^# help\:" Makefile | grep -v grep | sed 's/\# help\: //' | sed 's/\# help\://'

# help: install					- Create a virtual environment and install dependencies
.PHONY: install
install:
	@bash bin/install_with_uv.sh $(PYTHON_VERSION)

# help: install_precommit			- Install pre-commit hooks
.PHONY: install_precommit
install_precommit:
	@pre-commit install -t pre-commit
	@pre-commit install -t pre-push

# help: format			- format code using the precommits
.PHONY: format
format:
	@pre-commit run -a

# help: serve_docs_locally			- Serve docs locally on port 8001
.PHONY: serve_docs_locally
serve_docs_locally:
	@mkdocs serve --livereload -a localhost:8001

# help: deploy_docs				- Deploy documentation to GitHub Pages
.PHONY: deploy_docs
deploy_docs:
	@mkdocs build
	@mkdocs gh-deploy

# help: run_tests			- Run repository's tests
.PHONY: run_tests
run_tests:
	@pytest tests/
