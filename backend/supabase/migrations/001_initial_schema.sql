-- Perishable Goods Forecast API initial schema.
-- Apply through the Supabase CLI after reviewing the RLS policy assumptions.

create extension if not exists pgcrypto;

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create table public.organizations (
  id uuid primary key default gen_random_uuid(),
  name text not null check (length(trim(name)) > 0),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.user_profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  organization_id uuid not null references public.organizations(id) on delete restrict,
  email text not null,
  full_name text,
  role text not null default 'member',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (organization_id, email)
);

create table public.warehouses (
  id uuid primary key default gen_random_uuid(),
  organization_id uuid not null references public.organizations(id) on delete cascade,
  warehouse_code text not null,
  warehouse_name text not null check (length(trim(warehouse_name)) > 0),
  warehouse_type text not null check (warehouse_type in ('ambient', 'chilled', 'frozen', 'mixed')),
  country text,
  region text,
  city text,
  storage_capacity numeric check (storage_capacity is null or storage_capacity >= 0),
  capacity_unit text,
  timezone text not null default 'UTC',
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (organization_id, warehouse_code)
);

create table public.stores (
  id uuid primary key default gen_random_uuid(),
  organization_id uuid not null references public.organizations(id) on delete cascade,
  warehouse_id uuid references public.warehouses(id) on delete set null,
  external_store_id text not null,
  store_name text,
  region text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (organization_id, external_store_id)
);

create table public.suppliers (
  id uuid primary key default gen_random_uuid(),
  organization_id uuid not null references public.organizations(id) on delete cascade,
  external_supplier_id text not null,
  supplier_name text not null,
  supplier_score numeric,
  lead_time_days numeric check (lead_time_days is null or lead_time_days >= 0),
  minimum_order_quantity numeric check (
    minimum_order_quantity is null or minimum_order_quantity >= 0
  ),
  supplier_country text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (organization_id, external_supplier_id)
);

create table public.products (
  id uuid primary key default gen_random_uuid(),
  organization_id uuid not null references public.organizations(id) on delete cascade,
  external_product_id text not null,
  product_name text not null,
  category text,
  supplier_id uuid references public.suppliers(id) on delete set null,
  shelf_life_days numeric check (shelf_life_days is null or shelf_life_days >= 0),
  storage_temp numeric,
  spoilage_sensitivity numeric check (
    spoilage_sensitivity is null or spoilage_sensitivity >= 0
  ),
  base_price numeric check (base_price is null or base_price >= 0),
  cost_price numeric check (cost_price is null or cost_price >= 0),
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (organization_id, external_product_id)
);

create table public.datasets (
  id uuid primary key default gen_random_uuid(),
  organization_id uuid not null references public.organizations(id) on delete cascade,
  dataset_name text not null check (length(trim(dataset_name)) > 0),
  dataset_type text not null check (
    dataset_type in ('warehouse', 'products', 'suppliers', 'sales', 'inventory', 'custom')
  ),
  status text not null default 'uploaded' check (
    status in (
      'uploaded', 'mapping_required', 'validated', 'ready',
      'ingesting', 'ingested', 'failed'
    )
  ),
  original_filename text not null,
  storage_path text not null unique,
  file_size_bytes bigint not null check (file_size_bytes >= 0),
  row_count bigint not null default 0 check (row_count >= 0),
  column_count integer not null default 0 check (column_count >= 0),
  date_min date,
  date_max date,
  uploaded_by uuid not null references public.user_profiles(id) on delete restrict,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  check (date_min is null or date_max is null or date_min <= date_max)
);

create table public.dataset_columns (
  id uuid primary key default gen_random_uuid(),
  dataset_id uuid not null references public.datasets(id) on delete cascade,
  source_column text not null,
  normalized_column text not null,
  detected_type text not null,
  sample_values jsonb not null default '[]'::jsonb,
  missing_count bigint not null default 0 check (missing_count >= 0),
  unique_count bigint not null default 0 check (unique_count >= 0),
  created_at timestamptz not null default now(),
  unique (dataset_id, source_column)
);

create table public.column_mappings (
  id uuid primary key default gen_random_uuid(),
  organization_id uuid not null references public.organizations(id) on delete cascade,
  dataset_id uuid not null references public.datasets(id) on delete cascade,
  source_column text not null,
  target_field text,
  mapping_type text not null check (
    mapping_type in (
      'exact', 'alias', 'manual', 'generated', 'default',
      'ignored', 'identifier', 'custom_feature'
    )
  ),
  confidence numeric not null default 0 check (confidence between 0 and 1),
  is_confirmed boolean not null default false,
  default_value jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (dataset_id, source_column)
);

create table public.sales_records (
  id uuid primary key default gen_random_uuid(),
  organization_id uuid not null references public.organizations(id) on delete cascade,
  dataset_id uuid not null references public.datasets(id) on delete cascade,
  product_id uuid not null references public.products(id) on delete restrict,
  store_id uuid references public.stores(id) on delete set null,
  warehouse_id uuid references public.warehouses(id) on delete set null,
  transaction_date date not null,
  units_sold numeric not null check (units_sold >= 0),
  selling_price numeric check (selling_price is null or selling_price >= 0),
  discount_pct numeric check (discount_pct is null or discount_pct between 0 and 1),
  is_promoted boolean,
  spoilage_risk numeric check (spoilage_risk is null or spoilage_risk >= 0),
  source_row_number bigint not null check (source_row_number >= 2),
  extra_features jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  unique (dataset_id, source_row_number)
);

create table public.inventory_snapshots (
  id uuid primary key default gen_random_uuid(),
  organization_id uuid not null references public.organizations(id) on delete cascade,
  dataset_id uuid not null references public.datasets(id) on delete cascade,
  product_id uuid not null references public.products(id) on delete restrict,
  warehouse_id uuid not null references public.warehouses(id) on delete restrict,
  snapshot_date date not null,
  current_inventory numeric not null check (current_inventory >= 0),
  reserved_inventory numeric check (reserved_inventory is null or reserved_inventory >= 0),
  incoming_inventory numeric check (incoming_inventory is null or incoming_inventory >= 0),
  expiration_date date,
  batch_id text,
  spoilage_risk numeric check (spoilage_risk is null or spoilage_risk >= 0),
  source_row_number bigint not null check (source_row_number >= 2),
  extra_features jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  check (expiration_date is null or expiration_date >= snapshot_date),
  unique (dataset_id, source_row_number)
);

create table public.model_registry (
  id uuid primary key default gen_random_uuid(),
  organization_id uuid references public.organizations(id) on delete cascade,
  model_name text not null,
  model_version text not null,
  model_type text not null check (model_type in ('vqr', 'sklearn', 'mock')),
  artifact_directory text not null,
  target_name text not null,
  feature_count integer not null check (feature_count >= 0),
  is_active boolean not null default false,
  metrics jsonb not null default '{}'::jsonb,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create unique index model_registry_org_name_version_uidx
  on public.model_registry (coalesce(organization_id, '00000000-0000-0000-0000-000000000000'::uuid), model_name, model_version);
create unique index model_registry_one_active_per_org_idx
  on public.model_registry (coalesce(organization_id, '00000000-0000-0000-0000-000000000000'::uuid))
  where is_active;

create table public.prediction_jobs (
  id uuid primary key default gen_random_uuid(),
  organization_id uuid not null references public.organizations(id) on delete cascade,
  requested_by uuid not null references public.user_profiles(id) on delete restrict,
  model_id uuid references public.model_registry(id) on delete set null,
  dataset_id uuid references public.datasets(id) on delete set null,
  warehouse_id uuid references public.warehouses(id) on delete set null,
  store_id uuid references public.stores(id) on delete set null,
  product_id uuid references public.products(id) on delete set null,
  forecast_start_date date not null,
  forecast_horizon_days integer not null check (forecast_horizon_days in (14, 30, 60, 90)),
  status text not null default 'pending' check (
    status in ('pending', 'running', 'completed', 'failed', 'model_not_ready')
  ),
  request_payload jsonb not null default '{}'::jsonb,
  is_mock boolean not null default false,
  error_code text,
  error_message text,
  created_at timestamptz not null default now(),
  started_at timestamptz,
  completed_at timestamptz
);

create table public.forecast_results (
  id uuid primary key default gen_random_uuid(),
  organization_id uuid not null references public.organizations(id) on delete cascade,
  prediction_job_id uuid not null references public.prediction_jobs(id) on delete cascade,
  product_id uuid references public.products(id) on delete set null,
  warehouse_id uuid references public.warehouses(id) on delete set null,
  store_id uuid references public.stores(id) on delete set null,
  forecast_date date not null,
  predicted_units_sold numeric not null check (predicted_units_sold >= 0),
  lower_bound numeric check (lower_bound is null or lower_bound >= 0),
  upper_bound numeric check (upper_bound is null or upper_bound >= 0),
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  check (lower_bound is null or upper_bound is null or lower_bound <= upper_bound),
  unique (prediction_job_id, product_id, warehouse_id, store_id, forecast_date)
);

create index user_profiles_organization_idx on public.user_profiles (organization_id);
create index warehouses_organization_idx on public.warehouses (organization_id);
create index stores_organization_warehouse_idx on public.stores (organization_id, warehouse_id);
create index suppliers_organization_idx on public.suppliers (organization_id);
create index products_organization_supplier_idx on public.products (organization_id, supplier_id);
create index datasets_organization_type_status_idx on public.datasets (organization_id, dataset_type, status);
create index dataset_columns_dataset_idx on public.dataset_columns (dataset_id);
create index column_mappings_dataset_confirmed_idx on public.column_mappings (dataset_id, is_confirmed);
create index sales_lookup_idx on public.sales_records (organization_id, product_id, transaction_date desc);
create index sales_location_idx on public.sales_records (warehouse_id, store_id, transaction_date desc);
create index inventory_lookup_idx on public.inventory_snapshots (organization_id, product_id, warehouse_id, snapshot_date desc);
create index prediction_jobs_organization_created_idx on public.prediction_jobs (organization_id, created_at desc);
create index forecast_results_job_date_idx on public.forecast_results (prediction_job_id, forecast_date);

create trigger organizations_set_updated_at before update on public.organizations
for each row execute function public.set_updated_at();
create trigger user_profiles_set_updated_at before update on public.user_profiles
for each row execute function public.set_updated_at();
create trigger warehouses_set_updated_at before update on public.warehouses
for each row execute function public.set_updated_at();
create trigger stores_set_updated_at before update on public.stores
for each row execute function public.set_updated_at();
create trigger suppliers_set_updated_at before update on public.suppliers
for each row execute function public.set_updated_at();
create trigger products_set_updated_at before update on public.products
for each row execute function public.set_updated_at();
create trigger datasets_set_updated_at before update on public.datasets
for each row execute function public.set_updated_at();
create trigger column_mappings_set_updated_at before update on public.column_mappings
for each row execute function public.set_updated_at();
create trigger model_registry_set_updated_at before update on public.model_registry
for each row execute function public.set_updated_at();

-- This helper assumes every authenticated user has one user_profiles row.
-- Bootstrap organizations/profiles through a trusted admin or service-role flow.
create or replace function public.current_organization_id()
returns uuid
language sql
stable
security definer
set search_path = public
as $$
  select organization_id
  from public.user_profiles
  where id = auth.uid()
  limit 1
$$;

revoke all on function public.current_organization_id() from public;
grant execute on function public.current_organization_id() to authenticated;

alter table public.organizations enable row level security;
alter table public.user_profiles enable row level security;
alter table public.warehouses enable row level security;
alter table public.stores enable row level security;
alter table public.suppliers enable row level security;
alter table public.products enable row level security;
alter table public.datasets enable row level security;
alter table public.dataset_columns enable row level security;
alter table public.column_mappings enable row level security;
alter table public.sales_records enable row level security;
alter table public.inventory_snapshots enable row level security;
alter table public.model_registry enable row level security;
alter table public.prediction_jobs enable row level security;
alter table public.forecast_results enable row level security;

create policy organizations_same_org_select on public.organizations
for select to authenticated
using (id = public.current_organization_id());

create policy user_profiles_same_org_select on public.user_profiles
for select to authenticated
using (organization_id = public.current_organization_id());

create policy warehouses_same_org_all on public.warehouses
for all to authenticated
using (organization_id = public.current_organization_id())
with check (organization_id = public.current_organization_id());

create policy stores_same_org_all on public.stores
for all to authenticated
using (organization_id = public.current_organization_id())
with check (organization_id = public.current_organization_id());

create policy suppliers_same_org_all on public.suppliers
for all to authenticated
using (organization_id = public.current_organization_id())
with check (organization_id = public.current_organization_id());

create policy products_same_org_all on public.products
for all to authenticated
using (organization_id = public.current_organization_id())
with check (organization_id = public.current_organization_id());

create policy datasets_same_org_all on public.datasets
for all to authenticated
using (organization_id = public.current_organization_id())
with check (organization_id = public.current_organization_id());

create policy dataset_columns_same_org_all on public.dataset_columns
for all to authenticated
using (
  exists (
    select 1 from public.datasets d
    where d.id = dataset_id
      and d.organization_id = public.current_organization_id()
  )
)
with check (
  exists (
    select 1 from public.datasets d
    where d.id = dataset_id
      and d.organization_id = public.current_organization_id()
  )
);

create policy column_mappings_same_org_all on public.column_mappings
for all to authenticated
using (organization_id = public.current_organization_id())
with check (organization_id = public.current_organization_id());

create policy sales_records_same_org_all on public.sales_records
for all to authenticated
using (organization_id = public.current_organization_id())
with check (organization_id = public.current_organization_id());

create policy inventory_snapshots_same_org_all on public.inventory_snapshots
for all to authenticated
using (organization_id = public.current_organization_id())
with check (organization_id = public.current_organization_id());

create policy model_registry_same_org_select on public.model_registry
for select to authenticated
using (
  organization_id is null
  or organization_id = public.current_organization_id()
);

create policy prediction_jobs_same_org_all on public.prediction_jobs
for all to authenticated
using (organization_id = public.current_organization_id())
with check (organization_id = public.current_organization_id());

create policy forecast_results_same_org_all on public.forecast_results
for all to authenticated
using (organization_id = public.current_organization_id())
with check (organization_id = public.current_organization_id());

-- The FastAPI service uses the service-role key after authenticating each request
-- and always applies organization filters in repositories. Service-role bypasses RLS.
-- If browser-direct access is ever enabled, review every policy above first.
--
-- Supabase Storage policies are project-specific because bucket visibility and
-- upload ownership conventions vary. Create the private `dataset-files` bucket,
-- then restrict objects to a first path segment equal to the user's organization.
-- Never expose the service-role key to the browser.
