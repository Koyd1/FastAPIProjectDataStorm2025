-- Supabase schema for anomaly analytics application.
-- The schema avoids storing sensitive client data (transaction ids, PII).

create schema if not exists anomaly;
set search_path to anomaly, public;

-- Table: upload_batches
-- Stores metadata about each dataset uploaded through the web UI.
create table upload_batches_table (
    id uuid primary key default gen_random_uuid(),
    filename text not null,
    object_key text not null unique,
    row_count integer not null check (row_count >= 0),
    uploaded_at timestamptz not null default now(),
    source varchar(32) not null default 'csv',
    notes text
);

comment on table upload_batches_table is
    'Metadata for sanitized CSV datasets saved to Supabase Storage.';

comment on column upload_batches_table.filename is
    'Original filename provided by the user (sanitized, without PII).';

comment on column upload_batches_table.object_key is
    'Path to the CSV object stored in Supabase Storage bucket.';

comment on column upload_batches_table.source is
    'Source of the upload: csv (default), single_csv_prediction, form_submission.';


-- Table: rf_snapshots
-- Captures global statistics and model artifacts derived from all uploads.
create table if not exists rf_snapshots (
    id uuid primary key default gen_random_uuid(),
    computed_at timestamptz not null default now(),
    sample_size integer not null check (sample_size >= 0),
    class_counts jsonb not null default '{}'::jsonb,
    feature_importance jsonb not null default '[]'::jsonb,
    notes text
);

comment on table rf_snapshots is
    'Stores periodic summaries from Random Forest clustering (class distribution and aggregated feature importance).';

comment on column rf_snapshots.class_counts is
    'JSON object with predicted class names as keys and counts as values.';

comment on column rf_snapshots.feature_importance is
    'JSON array describing feature importance rankings produced by Random Forest.';


-- Optional helper view combining latest stats.
create or replace view rf_snapshots.latest_rf_snapshot as
select *
from rf_snapshots
order by computed_at desc
limit 1;

CREATE TABLE anomaly.requests (
    id SERIAL PRIMARY KEY,
    issuer_bank TEXT,
    account_type TEXT,
    card_type TEXT,
    product_tier TEXT,
    region TEXT,
    urban TEXT,
    age INTEGER,
    tenure_months INTEGER,
    cluster_id_expected INTEGER,
    cluster_name_expected TEXT,
    channel TEXT,
    merchant_category TEXT,
    merchant_country TEXT,
    currency TEXT,
    fx_rate_to_mdl NUMERIC,
    amount_txn_ccy NUMERIC,
    amount_mdl NUMERIC,
    card_present INTEGER,
    auth_method TEXT,
    is_3ds INTEGER,
    three_ds_result TEXT,
    device_type TEXT,
    device_trust_score NUMERIC,
    ip_risk_score NUMERIC,
    geo_distance_km NUMERIC,
    hour_of_day INTEGER,
    day_of_week INTEGER,
    is_night INTEGER,
    is_weekend INTEGER,
    txn_count_1h INTEGER,
    txn_amount_1h_mdl NUMERIC,
    txn_count_24h INTEGER,
    txn_amount_24h_mdl NUMERIC,
    merchant_risk_score NUMERIC,
    velocity_risk_score NUMERIC,
    new_device_flag INTEGER,
    cross_border INTEGER,
    campaign_q2_2025 INTEGER,
    amount_log_z NUMERIC,
    predicted_class TEXT,
    class_probability INTEGER
);
CREATE VIEW public.requests AS SELECT * FROM anomaly.requests;
