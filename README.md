# Artefact Data Science Boilerplate

Welcome to **Artefact Data Science Boilerplate**: a repository to build Data Science repositories.

This documentation contains the following sections:

- [How to use the created repository](#2-use-the-created-repository)
- [Why use this boilerplate?](#why-use-this-boilerplate?)
  - [Repository structure](#repository-structure)
  - [Eased development](#eased-development)
  - [Git collaboration & workflow](#git-collaboration-&-workflow)
  - [Eased documentation](#documentation)

# How-to use

This section contains a step-by-step guide to build your own Data Science repository.

## 1. Create a new repository using this template

### 1.1. To create a repo on Artefact's Artefactory
To create a new repository using this template, just select "artefactory/boilerplate-datascience-artefact" in the template section in github when creating a new repo.

### 1.2. On client infrastructure
When working outside of Artefact's Github, you cannot directly use the template from github. But don't worry! Just download it/copy it/email it to your client's environment and use it as the first building
block for new projects.

## 2. Setup the created repository

As this repository is not a full cookiecutter, you have a few follow-up actions to complete to make it fully yours:
- Delete the .skaff folder.
- Look up the "TODO: " markers and replace the various values (organisation, repo name, authors, repo structure...) by your information.
- Configure pyproject.toml, e.g. to activate/deactivate ruff linting rules.

## 3. Installation & Common usage

Once this is done, your repository is ready to be used! You can now start developping. Explore the Makefile to discover some basic command shorcuts you might want to use often in your project. To begin with, installing your dependencies in a virtual environment:

```bash
make install
```

By default, this command will create a uv environment named "boilerplate-datascience-artefact" and install the required packages in it.

Dependencies and precommit hooks are now installed. By default, this will lint & format your code everytime you commit, and run your tests everytime you push code to your branch.

Don't forget to often:
- Commit your code & run tests
- Complete `dependencies` section in pyproject.toml to add the required packages for your project
- Complete `dependency-groups` dev section to add packages that are only required for development (libraries used for instance in your CI/CD e.g. `pytest`)
- Complete the `README.md` file to describe your repository and how to use it


## 4. Documentation

The tool used to build the documentation is [MkDocs](https://www.mkdocs.org/). To develop it, run the following command:

```bash
mkdocs serve
```

This will serve the documentation locally on your computer.
Then, modify the files in the `docs` folder to build your documentation as well as the `nav` section in the `mkdocs.yaml` file to add the different pages of your documentation.

To publish your documentation on Github pages, you need to enable it in the repository settings: Settings > Pages > Source: "Deploy from a branch" / Branch: "gh-pages" / Folder: "/(root)"

No other action is required, the documentation will be automatically built and published on Github pages on each push to the `main` branch. If you desire to build it and deploy it manually, use the `make deploy_docs` Makefile entry.


# Why use this boilerplate?
This boilerplate is though to concentrate the current best practices promoted by Artefact for its Data Scientists and its clients. It mainly focuses on:
- Repository structure
- Eased Development
- Git collaboration & workflow
- Documentation


## Repository structure

This template provides a recommended repository structure for best practice development:

- `/bin`: This directory is designated for executable scripts. It typically contains the entry points for your application, such as shell scripts or Python scripts that initiate the execution of your codebase.

- `/config`: This folder is used for storing configuration files. These files can include both static and dynamic configurations that your project requires, such as environment settings or widespread variables. Keeping configurations separate from code helps in maintaining clean code and allows for easy updates and environment-specific settings.

- `/docs`: This directory is designated for documentation files managed by MkDocs. It should include all essential documentation for your project, such as user guides, API documentation, and any other pertinent information that aids users and developers in comprehending and utilizing the project effectively.

- `/lib`: This folder is used for storing the core library code of your project. It contains the main modules and packages that implement the primary functionality of your application. Organizing your code in this way promotes modularity and reusability.

- `/notebooks`: This directory is for Jupyter notebooks. It is a good practice to separate notebooks from the main codebase to keep exploratory data analysis, prototyping, and documentation of workflows organized and distinct from production code.

- `/secrets`: This folder is used for storing sensitive information such as API keys, passwords, and other credentials. It is crucial to ensure that this directory is properly secured and not included in version control systems to prevent unauthorized access.

- `/tests`: This directory is dedicated to test files. It should contain all unit tests, integration tests, and any other test scripts that verify the functionality and reliability of your code. Organizing tests in a separate directory helps maintain a clear structure and facilitates continuous integration and testing processes.

## Eased Development

This template repository comes with a `pyproject.toml` file that allows you to easily install the repository code in editable mode. It also enables to setup the different tools used in this repository (e.g. black, isort, ruff, etc...)
Using a `pyproject.toml` file is the recommended way to manage your project dependencies and tools. It is the new standard in Python, following [PEP 621](https://peps.python.org/pep-0621/).

## Git collaboration & workflow

### Pre-commit hooks

This template repository includes a [pre-commit](https://pre-commit.com/) [configuration file](https://github.com/artefactory/boilerplate-datascience-artefact/blob/master/repository/.pre-commit-config.yaml) that automates code cleaning during commits. Pre-commit hooks are essential for maintaining code quality and adherence to standards. The following tools are integrated to enhance code quality upon committing:

- Use [ruff](https://beta.ruff.rs/docs/) to lint, format, and sort imports in your code. Ruff is a high-performance linter/formatter written in Rust, offering features from flake8, isort, and black. It ensures your code adheres to the [PEP8](https://peps.python.org/pep-0008/) style guide, the standard for Python code style.

- Detect security vulnerabilities with [bandit](https://bandit.readthedocs.io/en/latest/), a tool specifically designed to identify common security issues in Python code.

- Utilize [nbstripout](https://github.com/kynan/nbstripout) to remove output from Jupyter notebooks, preventing conflicts when collaborating on the same notebook. This tool focuses on preserving code changes only.

- Execute tests using [pytest](https://docs.pytest.org/en/7.3.x/) during push events. Running tests before pushing code is a best practice. Tests are located in the `tests` directory and are executed on push to avoid slowing down the commit process. For enhanced coverage analysis, consider using [pytest-cov](https://pytest-cov.readthedocs.io/en/latest/).

- Employ [additional cleaning tools](https://github.com/pre-commit/pre-commit-hooks) to handle tasks like removing trailing whitespaces, ensuring end-of-file newlines, and checking for large files.

### CI pipeline

A CI pipeline is automatically triggered on push to check that your code is clean (it runs the pre-commits again) and that your tests pass. This is an additional security which, contrary to pre-commits, is run on the remote repository (on Github) and not on your local machine. This ensures that your code is always clean and that your tests pass before merging your code to the main branch.
The CI is defined in the [`.github/workflows/ci.yml`]

### Pull request template

A [pull request template](https://github.com/artefactory/boilerplate-datascience-artefact/blob/master/repository/.github/PULL_REQUEST_TEMPLATE.md) is also included in this template repository to help you write better pull requests. It is a good practice to write a good description of your pull request to help the reviewer understand what you did and why. This template provides a structure to help you write a good description as well as a checklist to ensure that you did not forget anything. It will be automatically displayed when you create a pull request.

### More

For more information about a good usage of git and other engineering topics, please refer to the [DS Engineering Documentation](https://ds-eng.artechfact.fr/)

## Documentation

This template repository comes with a [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) configuration file (`mkdocs.yaml`) that allows you to easily generate a nice documentation website for your project (and deploy it on Github pages).
MkDocs is a popular tool to generate documentation websites from markdown files, with a lot of plugins, and is used by many open source projects (FastAPI, Pydantic...)
Check the [How-to](../how_to) section for more details.
