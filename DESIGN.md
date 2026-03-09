# Design

## Goals
- Keep a clean, modular structure with reusable utilities.
- Provide both interactive and argparse CLIs.
- Save artifacts to output/ and models/ for reproducibility.

## Architecture
- src/models: model training and pipeline orchestration.
- src/utils: data loading, preprocessing, evaluation, plotting, downloads.
- main.py: menu-driven CLI.
- main_argparse.py: argparse-based CLI.

## Style
- Follow PEP 8.
- Use type hints where practical.
- Keep functions small and focused.
