# DataGuard AI — Bank Marketing upgrade

This package is a snapshot of the existing `feature/dataguard-research` project, extended with dataset profiling, CSV upload, and controlled classification experiments. The original synthetic label-noise endpoint and old benchmark scripts remain available.

## Install on Windows (existing clone)

1. Back up your existing `E:\PhD USA\Projects\data-quality-engine` folder. Do not replace its `.git` directory.
2. Extract this ZIP separately. Copy the contents into your existing project folder, allowing source files to be overwritten. Do **not** copy the extracted folder itself as a nested folder.
3. From the project root in PowerShell, with `.venv` active:

```powershell
python -m pip install -r requirements.txt
python -m pip install pytest httpx
python -m pytest tests phase2_backend -q
python -m uvicorn phase2_backend.main:app --reload --port 8000
```

4. In a separate terminal:

```powershell
cd phase2_frontend
npm install
npm run dev
```

5. Open http://localhost:3000. The first UCI dataset load needs internet access; later loads use `data/bank-full.csv`. You may also download `bank-full.csv` from UCI and place it in `data/` manually.

## Bank Marketing source and research cautions

Source: https://archive.ics.uci.edu/dataset/222/bank+marketing (Moro, Cortez, Rita; UCI Machine Learning Repository). The app uses the `bank-full.csv` variant (45,211 rows, 16 features and the `y` target). Its original values `unknown` are counted as missing in profiling and ML preparation. The `duration` column is excluded by default when predicting `y`, since call duration is not known before the call. This is an exploratory research prototype, not a validated predictive system. A single split does not establish robustness; add repeated stratified splits and uncertainty estimates before research claims.

## CSV upload behavior

Arbitrary header names are accepted and profiled without predefined schemas. Data types are inferred heuristically. The user chooses the target for classification; not every CSV is suitable. Uploads are processed in memory and not persisted by the API. CSV must be UTF-8, up to 8 MB, 60,000 rows, and 120 columns. No personal or confidential datasets should be uploaded to the public demo. This prototype has no authentication, rate limiting, or persistent experiment store. Before promoting beyond a limited demo, add request limits at the hosting edge, privacy review, monitoring, and a clear data policy.

## Publish (after you test locally)

1. Commit the changes to `feature/dataguard-research` (review `git status` first), then push that branch. No force push.
2. On Render, create a Blueprint from the GitHub repo using `render.yaml`, select the research branch. Render's free plan availability and limits may change. Set `DATAGUARD_ALLOWED_ORIGINS` to your **exact** deployed Vercel origin, e.g. `https://your-project.vercel.app`. If Render requires manual setup, use build `pip install -r requirements.txt` and start `uvicorn phase2_backend.main:app --host 0.0.0.0 --port $PORT`.
3. On Vercel, import the same repo, set Root Directory to `phase2_frontend`, select research branch for deployment or promote the tested branch as appropriate. Set `DATAGUARD_API_URL` to the Render URL, e.g. `https://your-api.onrender.com` (no trailing slash). Redeploy after changing this setting.
4. Update Render CORS origin if your final Vercel domain differs. Visit `/api/health` on Render and the public Vercel URL to verify.
5. The frontend and backend are separate services. The public URL does not work until you configure both accounts. Free Render services may cold-start and the UCI dataset is downloaded once per ephemeral instance.

## Further development

Edit `phase2_frontend/app/page.tsx` for UI, `phase2_frontend/app/globals.css` for design, `phase2_backend/main.py` for endpoints, and `phase2_backend/dataset_engine.py` for profiling and ML logic. Run tests and push to GitHub to trigger connected deployments.
