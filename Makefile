# Minimal Makefile to scaffold a 14-lecture MLOps course repo with notebooks and slides

SHELL := /bin/bash
PY ?= python3

.PHONY: help scaffold setup requirements

help:
	@echo "Targets:"
	@echo "  requirements  - Create a minimal requirements.txt if missing"
	@echo "  setup         - Create .venv and install requirements"
	@echo "  scaffold      - Create folders, notebooks (with pip install cell), and slides"

requirements:
	@if [ ! -f requirements.txt ]; then \
		cat > requirements.txt << 'EOF'; \
jupyter
numpy
pandas
scikit-learn
matplotlib
seaborn
mlflow
dvc
prefect
fastapi
uvicorn
pydantic
great-expectations
evidently
pytest
black
isort
flake8
mypy
python-pptx
EOF \
	; fi
	@echo "requirements.txt ready"

setup: requirements
	@$(PY) -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt
	@echo "Environment ready: source .venv/bin/activate"

scaffold: requirements
	@mkdir -p notebooks slides src tests configs data docker .github/workflows
	@$(PY) - << 'PY'
import os, json, re
from pathlib import Path

root = Path(".")
nb_dir = root / "notebooks"
slides_dir = root / "slides"
nb_dir.mkdir(parents=True, exist_ok=True)
slides_dir.mkdir(parents=True, exist_ok=True)

lectures = [
    (1,  "What is MLOps + local stack setup"),
    (2,  "Reproducibility and configuration"),
    (3,  "Data management and versioning"),
    (4,  "Data quality and validation"),
    (5,  "Experiment tracking"),
    (6,  "Project structure and testing for ML"),
    (7,  "Orchestration and pipelines"),
    (8,  "CI for ML with GitHub Actions"),
    (9,  "Model packaging and registry"),
    (10, "Serving: batch and real-time"),
    (11, "Monitoring and observability"),
    (12, "Deployment strategies"),
    (13, "Security, governance, and responsible AI"),
    (14, "Capstone, SLAs, incident response, and postmortems"),
]

def slugify(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')

def write_notebook(idx: int, title: str):
    slug = slugify(title)
    nb_path = nb_dir / f"lecture_{idx:02d}_{slug}.ipynb"
    if nb_path.exists():
        return
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}
        },
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Install course requirements\n",
                    "import sys, subprocess\n",
                    "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', '../requirements.txt'], check=False)\n"
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# Lecture {idx:02d}: {title}\n",
                    "\n",
                    "Use this notebook for hands-on exercises. Replace cells as needed.\n"
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Your code here\n",
                    "print('Hello MLOps - Lecture %02d')" % idx\n"
                ],
            },
        ],
    }
    with nb_path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

def write_slide(idx: int, title: str):
    slug = slugify(title)
    pptx_path = slides_dir / f"lecture_{idx:02d}_{slug}.pptx"
    if pptx_path.exists():
        return
    try:
        from pptx import Presentation
        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[0])
        slide.shapes.title.text = f"MLOps Lecture {idx:02d}"
        slide.placeholders[1].text = title + "\n\nStarter deck - replace with your content"
        prs.save(pptx_path.as_posix())
    except Exception:
        # Fallback: create a simple placeholder text file if python-pptx is unavailable
        txt_path = slides_dir / f"lecture_{idx:02d}_{slug}.txt"
        txt_path.write_text(f"MLOps Lecture {idx:02d}: {title}\nInstall python-pptx and re-run `make scaffold` to generate .pptx", encoding="utf-8")

for i, t in lectures:
    write_notebook(i, t)
    write_slide(i, t)

print("Scaffold complete: notebooks/ and slides/ populated.")
PY
	@echo "Scaffold created. Open notebooks/*.ipynb to start."
