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

