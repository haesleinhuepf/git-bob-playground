# git-bob: playground

In this repository, you can [create an issue](https://github.com/haesleinhuepf/git-bob-playground/issues/new/choose) which will then be answered by [git-bob](https://github.com/haesleinhuepf/git-bob) an AI-assistant. To make sure that the system does not get overwhelmed with requests, [@haesleinhuepf](https://github.com/haesleinhuepf) has to review requests and can also help guiding the AI-assistant so that useful results come out.

![](banner3.png)

This is a research project that serves exploring how we humans need to interact with AI-assistants to get reliable and trustworthy results. 
It may be changed or shut down at any time.

**Note:** Your images and the text you enter here may be sent to [OpenAI](https://openai.com/)'s online service ([Third party terms of use](https://openai.com/policies/row-terms-of-use/)) or [Anthropic's claude](https://www.anthropic.com/api) online service ([Terms of service](https://www.anthropic.com/legal/consumer-terms)) or [Google AI](https://ai.google.dev/) ([Terms of service](https://ai.google.dev/gemini-api/terms)) where we use a large language model to answer your request. 
Do not upload any data you cannot share openly. Also, do not enter any private or secret information. By submitting this GitHub issue, you confirm that you understand these conditions.

## Feedback welcome

If you tried git-bob on your own, you also directly provide feedback to the AI-assistant by [opening an issue on its repository](https://github.com/haesleinhuepf/git-bob).

## MLOps course starter (applied, 14 lectures)

This repository will host a minimal starter structure for an applied MLOps semester (14 lectures). Slides and example notebooks for each lecture will be added in folders listed below. Each notebook will include, near the top, a cell that installs its Python requirements via pip (example cell provided here).

Syllabus (one lecture per week)
- L1. What is MLOps + local stack setup
- L2. Reproducibility and configuration
- L3. Data management and versioning
- L4. Data quality and validation
- L5. Experiment tracking
- L6. Project structure and testing for ML
- L7. Orchestration and pipelines
- L8. CI for ML with GitHub Actions
- L9. Model packaging and registry
- L10. Serving: batch and real-time
- L11. Monitoring and observability
- L12. Deployment strategies
- L13. Security, governance, responsible AI
- L14. Capstone, SLAs, incident response, postmortems

Intended repository layout
- notebooks/lec01_setup.ipynb … notebooks/lec14_capstone.ipynb
- slides/lec01_setup.pptx … slides/lec14_capstone.pptx
- src/ (reusable Python package code)
- tests/ (unit/integration tests)
- configs/ (Hydra/config files)
- data/ (tracked with DVC; small sample only)
- docker/ (Dockerfiles, compose)
- .github/workflows/ (CI)
- Makefile (common tasks)

Example pip-install cell to include near the top of each notebook
- This is the exact pattern to use inside notebooks so they self-install dependencies.

```
import sys, subprocess

pkgs = [
    "numpy", "pandas", "scikit-learn",
    "mlflow", "dvc[s3]", "prefect",
    "fastapi", "uvicorn", "pydantic",
    "hydra-core", "great-expectations", "evidently",
    "matplotlib", "seaborn",
    "pytest", "black", "isort", "flake8", "mypy", "pre-commit"
]

subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=False)
```

Getting started (local)
- Use Python 3.10+.
- Create a virtual environment (venv or conda) and start JupyterLab/Notebook.
- Open notebooks in notebooks/ in order (lec01 → lec14). Run the first cell to install requirements, then proceed.

Note
- Large datasets and cloud services are not required; everything runs locally with small samples.
