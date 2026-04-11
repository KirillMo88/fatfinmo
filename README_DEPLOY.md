# Deploy to Streamlit Community Cloud

## 1) Push this folder to GitHub repo `KirillMo88/fatfinmo`

From this directory (`SCREENER`):

```bash
git init
git branch -M main
git add app.py requirements.txt custom_universe_lists.json .streamlit/config.toml README_DEPLOY.md
git commit -m "Prepare Streamlit Cloud deployment"
git remote add origin https://github.com/KirillMo88/fatfinmo.git
git push -u origin main
```

If `origin` already exists:

```bash
git remote set-url origin https://github.com/KirillMo88/fatfinmo.git
git push -u origin main
```

## 2) Deploy on Streamlit Community Cloud

1. Open `https://share.streamlit.io`.
2. Click **Create app**.
3. Select repository: `KirillMo88/fatfinmo`.
4. Branch: `main`.
5. Main file path: `app.py`.
6. In **Advanced settings**, set Python to **3.12** (or 3.11 if needed).
7. Click **Deploy**.

## 3) After deployment

- App changes auto-redeploy on every push to `main`.
- If build fails, open app logs in Streamlit and fix missing dependencies in `requirements.txt`.

## Note about Inputs tab persistence

`custom_universe_lists.json` is file-based. On Streamlit Community Cloud, runtime disk is ephemeral.
- Inputs edits may reset after app restart/redeploy.
- To persist edits long-term, use external storage (GitHub API commit flow, database, or object storage).
