# Deployment Guide

This guide covers three free/low-cost ways to put the Student Performance
Prediction System online: **Render**, **Railway**, and **PythonAnywhere**.
All three work with the `Procfile` and `requirements.txt` already included
in this repository.

> **Before deploying:** make sure you've run `python src/generate_dataset.py`
> and `python src/train_model.py` at least once locally and committed the
> resulting `models/*.pkl` and `static/plots/*.png` files to your repo (they
> are *not* excluded by `.gitignore`), OR configure a build-time step on your
> host to run these scripts automatically (shown below for each platform).

---

## Option A — Render (recommended, free tier available)

1. Push your project to GitHub (see `GITHUB_UPLOAD.md`).
2. Go to [render.com](https://render.com) and sign in with GitHub.
3. Click **New +** → **Web Service** → select your repository.
4. Configure the service:
   - **Environment**: `Python 3`
   - **Build Command**:
     ```
     pip install -r requirements.txt && python src/generate_dataset.py && python src/train_model.py
     ```
   - **Start Command**:
     ```
     gunicorn app:app --bind 0.0.0.0:$PORT
     ```
5. Add an environment variable: `FLASK_ENV = production`.
6. Click **Create Web Service**. Render will build and deploy automatically;
   subsequent pushes to your GitHub `main` branch redeploy automatically.
7. Your app will be live at `https://<your-service-name>.onrender.com`.

---

## Option B — Railway

1. Push your project to GitHub.
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**.
3. Railway auto-detects Python and reads the `Procfile` for the start command.
4. Under **Variables**, add: `FLASK_ENV = production`.
5. Under **Settings → Deploy**, set a custom build command so the model trains during deployment:
   ```
   pip install -r requirements.txt && python src/generate_dataset.py && python src/train_model.py
   ```
6. Deploy. Railway gives you a public URL under **Settings → Networking → Generate Domain**.

---

## Option C — PythonAnywhere (simplest for beginners, no credit card)

1. Create a free account at [pythonanywhere.com](https://www.pythonanywhere.com).
2. Open a **Bash console** from the Dashboard and clone your repo:
   ```bash
   git clone https://github.com/<your-username>/student-performance-prediction.git
   cd student-performance-prediction
   pip install --user -r requirements.txt
   python src/generate_dataset.py
   python src/train_model.py
   ```
3. Go to the **Web** tab → **Add a new web app** → choose **Flask** → Python 3.10+.
4. Set the **source code** path to your cloned repo folder.
5. Edit the generated WSGI file to import your app:
   ```python
   import sys
   path = '/home/<your-username>/student-performance-prediction'
   if path not in sys.path:
       sys.path.insert(0, path)
   from app import app as application
   ```
6. Click **Reload** on the Web tab. Your app is live at `https://<your-username>.pythonanywhere.com`.

---

## Capturing Screenshots for Your README

Once deployed (or running locally), capture these for your portfolio:

1. **Home page** — the prediction form with the gauge on the right (`/`)
2. **A completed prediction** — fill the form, click "Predict Performance", screenshot the animated gauge + grade badges
3. **Dashboard** — the metrics cards + chart grid (`/dashboard`)
4. **Prediction history table** — after making a few predictions, scroll to the "Live Prediction Activity" section on the dashboard

Save them into a `docs/screenshots/` folder and reference them in your `README.md`'s screenshots section.

---

## Environment Variables Reference

| Variable      | Purpose                                | Default (dev) |
|---------------|------------------------------------------|----------------|
| `SECRET_KEY`  | Flask session signing key                | insecure dev key — **always override in production** |
| `FLASK_ENV`   | `development` or `production`            | `development` |
| `FLASK_DEBUG` | Enables Flask's debug/reloader mode      | `False` |
| `PORT`        | Port the app listens on (set by most PaaS automatically) | `5000` |

Set `SECRET_KEY` to a long random string in production, e.g.:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## Running with Gunicorn Locally (production-style test)

```bash
pip install gunicorn   # already in requirements.txt
gunicorn app:app --bind 0.0.0.0:8000 --workers 2
```

Visit `http://127.0.0.1:8000`.
