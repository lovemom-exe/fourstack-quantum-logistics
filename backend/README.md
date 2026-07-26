# Perishable Goods Forecast API

FastAPI is the application boundary between the React frontend, Supabase, and exported prediction artifacts:

```text
React -> FastAPI -> Supabase Storage/PostgreSQL
                 -> exported VQR artifacts
```

The browser must never receive the Supabase service-role key and must not load the VQR directly. FastAPI authenticates the Supabase access token, enforces organization ownership, reads operational records, builds features from the saved schema, invokes an adapter, and persists forecast jobs/results.

## Environment setup

With `uv`:

```bash
cd backend
uv sync --dev
cp .env.example .env
uv run uvicorn app.main:app --reload
```

Or with `venv` and pip:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Set these values in the uncommitted `backend/.env`:

- `SUPABASE_URL`
- `SUPABASE_ANON_KEY` for future browser-safe integration only
- `SUPABASE_SERVICE_ROLE_KEY` for backend persistence
- `SUPABASE_STORAGE_BUCKET` (default `dataset-files`)
- `FRONTEND_ORIGIN` (default `http://localhost:5173`)
- `MODEL_ARTIFACT_DIR` (default `../ml/models/perishable_vqr`)

Development and test mode can start without Supabase so `/health`, `/docs`, `/redoc`, and model status remain inspectable. Supabase-backed routes return a structured configuration error until credentials are set. Production mode fails at startup when service credentials are absent. Settings and health responses never serialize secret values.

## Supabase setup and migration

Create a Supabase project, configure the environment values, and create a private Storage bucket named `dataset-files`. Apply:

```text
backend/supabase/migrations/001_initial_schema.sql
```

With a linked Supabase CLI project, this is typically:

```bash
supabase db push
```

The migration creates organization-owned tables, indexes, checks, timestamp triggers, and RLS templates. Review the comments at the bottom before enabling any browser-direct database access. The FastAPI server uses the service role only after authenticating a bearer token and adds `organization_id` filters in repositories.

## Authentication

Protected endpoints expect:

```http
Authorization: Bearer <supabase-access-token>
```

The token is verified through Supabase Auth, then `user_profiles` supplies the organization and role. Tests override this dependency and do not require a live Supabase account.

## CSV onboarding flow

1. `POST /api/v1/datasets/upload` accepts `multipart/form-data`.
2. FastAPI validates the extension/size, sanitizes the name, creates a collision-safe private Storage path, and uploads the original bytes.
3. pandas preserves original column names while generating normalized names only for matching.
4. Row/column counts, inferred types, missing/unique counts, duplicate count, samples, and initial mapping suggestions are stored.
5. No product, supplier, sales, or inventory rows are inserted during upload.
6. The user reviews `auto-map`, `mappings`, preview, and validation endpoints.

Unknown columns default to `ignored` as model inputs. During ingestion, non-canonical customer fields are retained in `metadata` or `extra_features` JSONB, so uploads do not require schema changes.

## Mapping, validation, and ingestion

Column normalization makes `Units Sold`, `units-sold`, `units sold`, and `UNITS_SOLD` equivalent. Exact and alias matches are suggested but remain unconfirmed. Validation returns structured errors/warnings for required mappings, missing/invalid values, dates, identifiers, ranges, duplicates, references, and model feature availability; it never silently fixes rows.

`POST /api/v1/datasets/{dataset_id}/ingest`:

```text
Storage download -> confirmed mappings -> validation -> relationship resolution
-> stable master upserts -> bounded transaction batches -> JSONB extras
```

Source row numbers and `(dataset_id, source_row_number)` uniqueness make transaction ingestion idempotent. A completed dataset is skipped on subsequent ingestion calls.

## Model artifact contract

Place the trained files in:

```text
ml/models/perishable_vqr/
├── vqr_model.dill
├── x_scaler.joblib
├── y_scaler.joblib
├── feature_selector.joblib
├── feature_schema.json
└── model_metadata.json
```

`GET /api/v1/models/status` reads only lightweight JSON and file presence. It does not load the VQR.

The VQR adapter:

- uses `VQR.from_dill`;
- loads and caches the selector/scalers/model once;
- never calls `fit` or `fit_transform`;
- reads candidate and selected order from `feature_schema.json`;
- validates feature counts dynamically;
- applies selector, X scaling, prediction, and Y inverse scaling in training order;
- clips negatives only after inverse target scaling.

No route contains a fixed feature count or guessed feature list. Replacing a 14-feature artifact with an 8-feature artifact only requires replacing the artifact directory.

## Prediction flow

```text
Validate request and organization access
-> verify dataset ingestion/mappings when dataset_id is supplied
-> create prediction job
-> select real or explicitly labeled mock adapter
-> query product/supplier/location and records dated before the forecast
-> construct schema-ordered rows
-> predict
-> save one result per product/location/date
-> complete job
```

Supported horizons are 14, 30, 60, and 90 days. The current VQR training design uses mostly static and scenario inputs. If those inputs are unchanged, the backend intentionally produces the same prediction for each horizon date; it does not fabricate daily trends. Date-specific values should only be introduced after the saved training schema includes matching time features.

When `ALLOW_MOCK_PREDICTIONS=true` and VQR artifacts are absent, a deterministic placeholder can run and every response/result is marked `is_mock: true`. It must not be presented as a real forecast. With mock mode disabled, the API returns HTTP 503:

```json
{
  "error": {
    "code": "MODEL_NOT_READY",
    "message": "The trained prediction model is not available yet.",
    "details": {}
  }
}
```

## Endpoint summary

OpenAPI is available at `/docs` and `/redoc`.

- Health: `GET /health`, `GET /api/v1/health`
- Warehouses: `POST/GET /api/v1/warehouses`, `GET/PUT/DELETE /api/v1/warehouses/{id}`
- Datasets: upload/list/get/delete, preview, columns, validate, readiness
- Mappings: auto-map/list/update under `/api/v1/datasets/{id}`
- Ingestion: `POST /api/v1/datasets/{id}/ingest`
- Models: `GET /api/v1/models/status`
- Predictions: `POST /api/v1/predictions`, job and result `GET` routes

## Tests

Tests use fake services/repositories and dependency overrides:

```bash
cd backend
uv run pytest
```

No test requires a Supabase account or loads/trains the real VQR.

## Frontend integration remaining

The existing Data and Forecast pages still use local demonstration state. A frontend API client must:

- attach the Supabase access token to FastAPI requests;
- replace local warehouse/upload/mapping/readiness state with the endpoints above;
- send UUID product/warehouse/store IDs;
- convert displayed discount percentages (for example `10`) to the backend fraction (`0.10`);
- show `is_mock` prominently and handle `MODEL_NOT_READY`;
- render persisted forecast results without calling Supabase or the model directly.
