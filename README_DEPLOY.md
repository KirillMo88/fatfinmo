# Production deployment

GitHub `main` is the canonical production source. Never copy edited files directly
to the VPS. Every production update follows this order:

1. Create a branch from the latest `origin/main`.
2. Make the change and run the relevant tests.
3. Commit and push the branch to GitHub.
4. Merge the reviewed branch into GitHub `main`.
5. Update the local `main` and deploy that exact commit with:

```powershell
git switch main
git pull --ff-only origin main
.\deploy\deploy-vps.ps1 -SshKey "C:\path\to\fatfinmo_vps_ed25519"
```

The deploy script stops before touching the VPS when:

- the working tree has uncommitted changes;
- local `HEAD` is not the exact commit published as `origin/main`;
- the VPS configuration, persistent directory, or Compose configuration is missing.

Each deployment is extracted into `/opt/fatfinmo/releases/github-main-<UTC>-<SHA>`.
The release reuses `/opt/fatfinmo/.env` and `/opt/fatfinmo/persistent`, rebuilds only
`screener` and `screener-jobs`, checks the public health endpoint, and records the
successful release in `/opt/fatfinmo/current`.

## Scheduled refresh reliability

The jobs scheduler launches refreshes as supervised child processes (at most two at
once) so one slow source cannot block the scheduler or overload the server. A missed
latest daily or weekly run is started once after scheduler restart; failed or
lock-contended attempts retry with exponential backoff (starting at five minutes,
capped at one hour) until the next scheduled occurrence.
The default task timeouts and retry
interval can be overridden with the `SCREENER_*_TIMEOUT_SECONDS` and
`SCREENER_REFRESH_RETRY_SECONDS` environment variables.

Job locks use operating-system advisory locks. The persistent `.lock` files are
diagnostic metadata only and intentionally remain after a job exits; their existence
does not mean a job is running. A killed process releases its lock automatically.
Refresh outcomes and source errors are recorded in
`persistent/job_status/refresh_jobs.jsonl`. A successful scheduler attempt cannot
guarantee that an upstream provider has published new data; failed CFTC source
fetches are recorded as failures and retried while retaining the last saved data.

To roll back, run the previous release's Compose file against the same project:

```bash
docker compose -p fatfinmo \
  -f /opt/fatfinmo/releases/<previous-release>/docker-compose.yml \
  --env-file /opt/fatfinmo/.env \
  up -d --no-deps --build screener screener-jobs
```

# Streamlit Community Cloud

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
