# GitHub Upload Guide

Step-by-step instructions to publish this project to your own GitHub account.

## 1. Create a New Repository on GitHub

1. Go to [github.com/new](https://github.com/new).
2. **Repository name**: `student-performance-prediction`
3. **Description**: `Machine learning web app that predicts student exam performance using Flask, scikit-learn, and SQLite.`
4. Choose **Public** (so it's visible on your portfolio/resume).
5. **Do NOT** initialize with a README, .gitignore, or license — this project already has them.
6. Click **Create repository**. Keep the page open; you'll need the remote URL shown.

## 2. Initialize Git Locally and Push

From inside the project folder:

```bash
cd student-performance-prediction

# Initialize a git repository (skip if already a git repo)
git init

# Stage all files
git add .

# Create your first commit
git commit -m "Initial commit: Student Performance Prediction System"

# Rename the default branch to main (if not already)
git branch -M main

# Link your local repo to GitHub (replace with your actual URL)
git remote add origin https://github.com/<your-username>/student-performance-prediction.git

# Push
git push -u origin main
```

## 3. Verify

Refresh your GitHub repository page — you should see all files: `app.py`, `src/`, `templates/`, `static/`, `README.md`, etc.

## 4. Add Topics & Polish (optional but recommended for recruiters)

On your repo page, click the ⚙️ gear icon next to "About" and add:
- **Topics**: `machine-learning`, `flask`, `scikit-learn`, `python`, `data-science`, `rest-api`, `sqlite`
- **Website**: your deployed URL (see `DEPLOYMENT.md`)
- **Description**: a one-line summary (see above)

## 5. Future Updates

After making changes:
```bash
git add .
git commit -m "Describe what you changed"
git push
```

## 6. Common Issues

| Problem | Fix |
|---|---|
| `remote origin already exists` | Run `git remote remove origin` then re-add it. |
| Large `models/*.pkl` files rejected by GitHub (>100MB) | This project's models are small (a few MB), but if you retrain with many more estimators and hit GitHub's limit, use [Git LFS](https://git-lfs.github.com/) or exclude `models/*.pkl` from git and regenerate them via your deployment platform's build step instead (see `DEPLOYMENT.md`). |
| `Authentication failed` when pushing | GitHub no longer accepts passwords over HTTPS — create a [Personal Access Token](https://github.com/settings/tokens) and use it as your password, or set up [SSH keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh) instead. |

## 7. Cloning on Another Machine

```bash
git clone https://github.com/<your-username>/student-performance-prediction.git
cd student-performance-prediction
```
Then follow `INSTALLATION.md` from Step 2 onward.
