# Symbipredict

Symbipredict is an educational desktop application that uses a TensorFlow model to show symptom-based disease-pattern estimates. Its Tkinter interface runs locally; it does not start a web server.

> **Not medical advice:** Estimates are not diagnoses or clinically validated probabilities. Consult a qualified healthcare professional for medical concerns.

## Project structure

```text
Symbipredict/
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── dataset/
│   │   └── symbipredict_2022.csv
│   ├── models/
│   │   ├── label_encoder.pkl
│   │   ├── model.h5
│   │   └── model_metadata.json
│   └── legacy/
│       └── g.py
├── .gitattributes
├── .gitignore
└── README.md
```

`backend/legacy/g.py` is an older standalone version; use `backend/app.py` to run the current application. Local virtual environments (`venv/`, `.venv/`, and `.venv-*`) are excluded from Git.

## GitHub setup

The dataset is larger than GitHub's 100 MB per-file limit, so it is configured for [Git LFS](https://git-lfs.com/).

1. Install Git LFS, then run `git lfs install`.
2. Initialize or clone your GitHub repository in this project folder.
3. Before staging the dataset for the first time, confirm `git lfs track` lists `backend/dataset/*.csv` (the `.gitattributes` file configures this pattern).
4. Stage and push the project normally. Git LFS stores the dataset; the local virtual environments are ignored.

Git LFS has storage and bandwidth quotas. If you prefer not to use LFS, host the dataset separately and place it at `backend/dataset/symbipredict_2022.csv` before running the app.

## Run locally

Use **Python 3.11** for this project. The dependencies pin TensorFlow 2.13.0, which is not available for Python 3.14.

If Python 3.11 is not installed, install a Python 3.11 release from the [official Python downloads page](https://www.python.org/downloads/), including the Python Launcher for Windows. Open a new PowerShell window and confirm the launcher can find it:

```powershell
py -0p
```

Run the following commands from the **project root** (the folder containing this README). They create a local virtual environment using Python 3.11, verify its version, install the dependencies, and launch the app:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv\Scripts\python.exe backend\app.py
```

The first command creates `.venv` with Python 3.11; the remaining commands explicitly use that environment's Python, so they do not depend on which `python` or `pip` is first on PATH. Check the environment with:

```powershell
.\.venv\Scripts\python.exe --version
```

It should report Python 3.11.x. If `py -3.11` reports that no matching runtime is installed, install Python 3.11 and reopen PowerShell. If you already created `.venv` with a different Python version, remove that environment and run the setup commands again. Virtual environments are local and should not be uploaded to GitHub.

The dataset must be present at `backend/dataset/symbipredict_2022.csv`. The app uses the saved model when its metadata matches the dataset; otherwise, it trains a replacement model and saves it under `backend/models/`.

## Use the app

1. Accept the terms and choose whether to view the first-time tutorial.
2. Search or browse symptoms, add them to the selection, and review the selection.
3. Choose **Get estimate** to review the entries and view the model's three closest patterns.

The dataset uses its own symptom names. For example, it includes `fever`, not separate high- and mild-fever entries.
