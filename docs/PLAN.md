# Flight prices — master plan (execution roadmap)

This document is the single source of truth for turning the notebook study into a **recruiter-grade, production-shaped** artifact without losing the original analysis.

## North star

A reviewer should be able to: **clone → `pip install -e .` → `make test` → (optional data) `flight-train` → `docker compose up` → hit `/predict`** and see **clear docs, clean boundaries, and honest evaluation**.

## Principles

1. **Notebook stays** as narrative EDA; **library code** is the source of truth for training and serving.
2. **No silent leakage** in the library path: encoders and `PowerTransformer` fit on **training rows only**; test and API use frozen transforms.
3. **One obvious command** per action (`train`, `serve`, `test`).
4. **CI must pass without Kaggle data** (synthetic fixture only).

---

## Phase A — Foundation (current sprint)

| Step | Deliverable | Done when |
|------|-------------|-----------|
| A1 | `src/flight_prices/` package | Importable; `ruff`/`compileall` clean |
| A2 | `preprocess.py` — `clean_common`, `FlightPreprocessor` (sklearn-compatible) | Train-only fit; `transform` works on holdout + single-row API |
| A3 | `train.py` + CLI `flight-train` | Writes `artifacts/model.joblib`, `artifacts/metrics.json`, `artifacts/feature_list.json` |
| A4 | Baselines | `DummyRegressor(mean)` + `Ridge` on same split for comparison table |
| A5 | `tests/fixtures/*.csv` + pytest | Full train smoke on fixture; CI green |

## Phase B — Rigor and story

| Step | Deliverable | Done when |
|------|-------------|-----------|
| B1 | **Time-aware split** (optional flag) — sort by `Date_of_Journey`, last 25% test | Documented in README; metrics differ honestly from random split |
| B2 | **SHAP** (or permutation importance) for primary model | Saved plot or JSON under `artifacts/` |
| B3 | **Error analysis** — deciles of absolute error by airline or stops | Short section in README or notebook link |

## Phase C — Serving and ops

| Step | Deliverable | Done when |
|------|-------------|-----------|
| C1 | FastAPI `/predict` + `/health` | OpenAPI works; example `curl` in README |
| C2 | Dockerfile + compose | API boots with mounted `artifacts/` |
| C3 | `Makefile` targets | `make test`, `make train`, `make up` |

## Phase D — Polish

| Step | Deliverable | Done when |
|------|-------------|-----------|
| D1 | `requirements-lock.txt` or `uv lock` from a clean machine | Reproducible pins documented |
| D2 | Optional `nbstripout` / pre-commit | Contributor hygiene documented |
| D3 | GitHub repo rename `regrression` → `regression` | Links updated |

---

## Definition of done (project-level)

- [ ] README architecture diagram mentions **library + API**, not only the notebook.
- [ ] CI: lint (optional) + pytest on fixture + no network.
- [ ] `artifacts/` documented and gitignored; sample `metrics.json` structure in README.
- [ ] Clear **limitations** paragraph (static historical fares, geography, no live inventory).

---

## Execution order (slow and proper)

Work strictly **A → C** first (shipping skeleton + honesty), then **B** (depth), then **D** (polish). Each phase ends with a **small commit** and a green CI run.
