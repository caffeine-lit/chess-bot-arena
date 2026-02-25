# Streamlit Chess AI Demo

Standalone Streamlit app for playing chess against a pretrained next-move model (`.pt` artifact).

## What to copy into this folder
- A compatible model artifact in `models/` (example: `models/model.pt`)
- Optional training/eval outputs can stay under `artifacts/`, but the app auto-discovers models only from `models/`

The app is self-contained and does not import from the parent repo. You can move this folder and treat it as a new repo root.

## Run locally

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## Swapping models
- Put multiple `*.pt` files under `models/` and choose one in the sidebar.
- Or upload a `.pt` file with the sidebar uploader (overrides local selection when enabled).

## Streamlit Cloud deployment notes
- Add a compatible `.pt` model file to the deployed app repo (for example `models/model.pt`).
- `requirements.txt` installs `torch`, `python-chess`, and `streamlit`.
- The app runs inference on CPU only (no GPU required).
- `packages.txt` is included as an APT-package placeholder (not needed by default).

## Deploy on Streamlit Community Cloud
1. Put this folder in its own repo (or move it so it becomes the repo root).
2. Ensure the entrypoint is `app.py`.
3. Commit `runtime.txt` and `requirements.txt`.
4. Choose one model strategy:
   - Commit a small/medium `.pt` artifact into `models/` (recommended for auto-load), or
   - Deploy without a bundled model and upload a `.pt` file in the sidebar after launch.
5. In Streamlit Cloud, create the app and point it at this repo/root.
6. If you later need Linux system packages, add them to `packages.txt` (one per line).

If your model file is large, runtime upload is usually the simplest path for a demo.

## Compatibility
This demo expects the same artifact structure as the training output from this project:
- `state_dict`
- `vocab`
- `config` (`embed_dim`, `hidden_dim`, `use_winner`)
