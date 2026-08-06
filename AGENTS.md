# AGENTS.md

Personal AI/ML learning workspace. **Not a git repo.** No tests, no lint config — verification = running scripts and inspecting output.

## Running code

- Always use the project venv: `./venv/bin/python script.py` (or `source venv/bin/activate`). `python` on PATH is Anaconda — do **not** use it for scripts; notebooks historically ran under Anaconda Jupyter.
- Streamlit apps must be launched from the project root: `./venv/bin/streamlit run cnn/app.py`, `./venv/bin/streamlit run cnn/yolo/app.py`, `./venv/bin/streamlit run supervised/Scikit-learn/app.py`. Caching (`@st.cache_resource`) already handles the rerun-per-interaction model; don't add manual caching.
- All deps are pinned in `requirements.txt` and already installed in `venv/` (Python 3.13). If a package is missing, install into the venv, not Anaconda.

## Layout & conventions

- **`supervised/`** — `Scikit-learn/`, `pytorch/`, `tensorflow/`, each with `train.py` (trains → saves model), `predict.py` (loads model, runs sample), and a model artifact (`model.pkl`, `.keras`). Scikit-learn `train.py` re-creates `train_test_split(test_size=0.2, random_state=42)` — any app/metrics code must reuse that exact split.
- **`unsuperviced/`** (note the misspelling) — `k-Means Clustering/`, `Hierarchical Clustering/`.
- **`scikit-learn-course/`** — numbered `01`–`07` lessons meant to be run **in order**; see its `README.md`.
- **`cnn/`** — CIFAR-10 CNN + `yolo/` subproject. YOLO/SAM weights auto-download on first run into `cnn/yolo/weights/` (already populated, `.pt` files).
- **`nlp/`** — notebooks only (`langchain/`, `nltk/`, `spaCy/`, `transformers/`).
- **`computer vision/OpenCV`** — note the space in the directory name; quote paths.

## Gotchas

- `.env` contains a real `GOOGLE_API_KEY` (currently unused by code). Never log, print, or commit it.
- Model artifacts (`*.pkl`, `*.keras`, `*.joblib`) are loaded relative to each script's directory — keep them co-located with their `predict.py`.
- Top-level scratch: `calculator.py`, `mcp-guide.html`, `openclaw.ipynb` (OpenRouter/LLM experiments), `word_analyzer_combined.json`.
