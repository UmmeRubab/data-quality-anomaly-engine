# DataGuard— start here

## One-click local startup (Windows)

1. Extract the ZIP. **Copy its contents into your existing project folder**, replacing the source files, but **keep your existing `.git` and `.venv` folders**. Do not create a nested `data-quality-engine/data-quality-engine` folder.
2. Double-click `START_DATAGUARD.bat` in the project root.
3. Leave the two terminal windows open. Once both say `Ready` / `Application startup complete`, open http://localhost:3000.
4. The frontend now calls its own `/api/...` routes; Next.js forwards requests to the backend on port 8000. This avoids browser cross-origin restrictions and LAN-address mismatch.

If the backend window reports missing dependencies, from the project root run `python -m pip install -r requirements.txt`. If the frontend reports missing dependencies, run `npm install` in `phase2_frontend` once.

## UCI Bank Marketing

The benchmark is the UCI `bank-full.csv` dataset (45,211 rows, 17 columns including target `y`). On first use the Python backend downloads it from UCI and caches it in `data/bank-full.csv`; **internet access is required for the first load**. If UCI cannot be reached, download `bank-full.csv` from https://archive.ics.uci.edu/dataset/222/bank+marketing and put it in the project's `data` folder. This ZIP does not claim to contain that downloaded file.

CSV upload supports arbitrary header names and shows inferred types, missingness, duplicates, and column summaries. It is limited to 8 MB, 60,000 rows and 120 columns. Avoid uploading confidential or personal data to a public demo.

## Deploy publicly

1. Push the tested source to GitHub on your existing `feature/dataguard-research` branch; do not force-push or replace `.git`.
2. On Render, deploy the repository with the included `render.yaml`. Check `https://YOUR-RENDER-URL/api/health`.
3. On Vercel, import the same repository and choose `phase2_frontend` as the Root Directory. Set `DATAGUARD_API_URL` to your **actual** Render HTTPS URL (without `/api`), then deploy.
4. The frontend uses same-origin `/api` rewrites to Render. You do not need to expose your home PC or keep VS Code open after deployment. Render free instances can sleep and cold-start.

## Research caveats

This is an exploratory data-centric ML prototype. It evaluates controlled training corruption against an untouched test split. A single run is not evidence of statistical robustness; use repeated trials and confidence intervals before making research claims.

## Verification

Run `python -m pytest tests phase2_backend -q` from the root. The local web UI and public hosting still need to be checked on your Windows machine/accounts.
