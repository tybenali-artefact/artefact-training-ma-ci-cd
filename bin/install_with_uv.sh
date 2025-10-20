#!/bin/bash -e

UV_CMD="uv"
PROJECT_NAME="boilerplate-datascience-artefact"
PYTHON_VERSION=${1:-3.11}

read -p "Want to create uv env .venv? (y/n) " answer
if [ "$CI" = "true" ] || [ "$answer" = "y" ]; then
  echo "Creating uv virtual environment with Python '${PYTHON_VERSION}'..."
  $UV_CMD venv --python "$PYTHON_VERSION"

  echo "Installing requirements from pyproject.toml..."
  $UV_CMD sync

  echo "Installing IPython kernel..."
  $UV_CMD run python -m ipykernel install --user --name="$PROJECT_NAME"

  echo "Installation complete!"
else
  echo "Installation of uv env aborted!"
fi
