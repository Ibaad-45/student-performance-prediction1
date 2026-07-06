# Installation Guide

This guide walks through setting up the Student Performance Prediction System from scratch on Windows, macOS, or Linux.

## Prerequisites

- **Python 3.10+** (developed and tested on Python 3.12) — [download here](https://www.python.org/downloads/)
- **pip** (bundled with Python)
- **git** (to clone the repository) — [download here](https://git-scm.com/downloads)

Check your Python version:
```bash
python3 --version
```

---

## Step 1 — Get the Code

If you're cloning from GitHub:
```bash
git clone https://github.com/<your-username>/student-performance-prediction.git
cd student-performance-prediction
```

Or if you already have the project folder locally, just `cd` into it.

---

## Step 2 — Create a Virtual Environment

Keeping dependencies isolated avoids conflicts with other Python projects.

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

You'll know it worked when you see `(venv)` at the start of your terminal prompt.

---

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs Flask, pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, gunicorn, and pytest.

If you hit a permissions error on Linux, do **not** use `sudo pip install`. Instead confirm your virtual environment is activated (Step 2).

---

## Step 4 — Generate the Dataset

```bash
python src/generate_dataset.py
```

Expected output:
```
[OK] Generated 2500 student records -> .../data/raw/student_performance_raw.csv
[OK] Missing values injected: ...
```

This creates `data/raw/student_performance_raw.csv`.

---

## Step 5 — Train the Models

```bash
python src/train_model.py
```

This will:
- Run EDA and save charts to `static/plots/`
- Train and compare 3 regression models + 1 classification model
- Save the winning models, scaler, and encoders to `models/`
- Save `models/metrics.json` for the dashboard

Expect this to take **10–30 seconds** on a typical laptop.

---

## Step 6 — (Optional) Verify the Models

```bash
python src/evaluate_model.py
```

This reloads the saved models and reprints the evaluation metrics — useful to confirm everything saved correctly.

---

## Step 7 — Run the Web App

```bash
python app.py
```

You should see:
```
 * Running on http://127.0.0.1:5000
```

Open your browser to **http://127.0.0.1:5000** — you'll see the prediction form. Visit **http://127.0.0.1:5000/dashboard** for the analytics dashboard.

---

## Step 8 — Run the Tests (Optional but Recommended)

```bash
python -m unittest tests/test_app.py -v
```

All 14 tests should pass (`OK` at the end of the output).

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'flask'` | Your virtual environment isn't activated, or `pip install -r requirements.txt` failed — re-run Step 2 and 3. |
| `FileNotFoundError: Raw dataset not found` | Run Step 4 (`generate_dataset.py`) before training. |
| `FileNotFoundError: Model artifact '...' not found` | Run Step 5 (`train_model.py`) before starting the app. |
| Port 5000 already in use | Run `PORT=5001 python app.py` (macOS/Linux) or set the `PORT` environment variable on Windows, then visit that port instead. |
| Charts look broken/missing on the dashboard | Re-run `python src/train_model.py` — the dashboard reads PNGs from `static/plots/`. |

---

## Next Steps

- See [`DEPLOYMENT.md`](DEPLOYMENT.md) to put this online.
- See [`GITHUB_UPLOAD.md`](GITHUB_UPLOAD.md) to publish your code.
