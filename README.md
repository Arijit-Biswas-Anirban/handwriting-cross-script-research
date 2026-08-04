# Handwriting Cross-Script Research

Research on self-supervised, open-set, cross-script writer verification using offline handwriting datasets.

## Researcher

**Arijit Biswas Anirban**  
Department of Computer Science and Engineering  
Pabna University of Science and Technology

## Supervisor

**Dr. Md. Abdur Rahim**  
Department of Computer Science and Engineering  
Pabna University of Science and Technology

## Current Research Direction

This project investigates whether writer-specific handwriting style can be preserved across different scripts while reducing script-related and text-content bias.

The planned research includes:

- Writer-disjoint verification
- Arabic–English cross-script evaluation using QUWI
- Self-supervised representation learning
- Script and content leakage analysis
- Open-set writer verification
- Uncertainty and score calibration
- External Bangla–Roman validation
- Future Bangla–English visual–kinematic analysis

## Main Datasets

- **QUWI** — primary Arabic–English dataset
- **BRS-ID** — planned Bangla–Roman external validation dataset
- **BDSHWA** — future Bangla–English visual–kinematic extension

Datasets are stored outside this repository and will not be uploaded to GitHub.

## Environment

- Python 3.12
- PyTorch 2.11
- Torchvision 0.26
- CUDA-enabled NVIDIA RTX 3050 Laptop GPU
- uv for dependency and environment management
- Jupyter notebooks through Visual Studio Code

## Installation

Clone the repository:

    git clone <repository-url>
    cd handwriting-cross-script-research

Create and synchronize the environment:

    uv sync

Activate the environment:

    source .venv/bin/activate

Open the project in Visual Studio Code:

    code .

## Project Structure

    handwriting-cross-script-research/
    ├── notebooks/
    │   └── 00_environment_check.ipynb
    ├── src/
    │   └── handwriting_cross_script_research/
    │       └── __init__.py
    ├── .gitignore
    ├── .python-version
    ├── pyproject.toml
    ├── uv.lock
    └── README.md

## Current Status

- QUWI public subset downloaded
- 1,900 QUWI document images extracted
- QUWI filename and page-condition mapping verified
- Official writer-disjoint train/test metadata inspected
- Train writers: 1–282
- Test writers: 283–475
- NVIDIA driver and CUDA access verified
- PyTorch GPU tensor test passed
- ResNet-18 forward and backward passes passed
- Jupyter notebook kernel connected to the project virtual environment

## Reproducibility

The following files are committed to GitHub:

- Source code
- Jupyter notebooks
- pyproject.toml
- uv.lock
- Research configuration files
- Dataset manifests and split definitions

The following items are excluded:

- Raw datasets
- Virtual environment
- Model checkpoints
- Experiment outputs
- Logs
- Private credentials

## License

A project license will be selected before the repository is publicly released.
