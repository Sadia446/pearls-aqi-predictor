# AeroSense — Streamlit dashboard

A lightweight read-only view of the same forecasts the Next.js app serves. It
queries the tables our pipelines write (`predictions`, `aqi_features_latest`,
`forecast_drivers`, `alerts`, `model_registry`) and computes nothing itself.

## Run locally

```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

It picks up `SUPABASE_DB_URL` from the project's `.env` automatically.

## Deploy (Streamlit Community Cloud, free)

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **New app** → pick this repo, branch `main`.
3. Set **Main file path** to `streamlit_app/app.py`.
4. Under **Advanced settings → Secrets**, paste:

   ```toml
   SUPABASE_DB_URL = "postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres"
   ```

5. Deploy.

> Secrets live only in Streamlit's settings — never commit them. `.streamlit/secrets.toml`
> is gitignored.
