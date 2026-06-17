"""
generate_dashboard_json.py
==========================
Generates combined_dashboard_data_periods.json from raw source data.

Sources used:
  - master_stop_search_with_neighbourhood.csv  (stop-and-search records)
  - MEP_oslaua_2025.csv                        (ethnicity population proportions per LAD)
  - LAD_MAY_2025_UK_BGC_V2_*.geojson          (LAD boundaries for spatial join)
  - Police_Force_Areas.geojson                 (PFA boundaries for spatial join)

Output structure:
  {
    metadata: { ... },
    administrative: [ { code, name, groups, totalStops, totalHits } ... ],  // 361 LADs
    pfa:            [ { code, name, groups, totalStops, totalHits } ... ],  // 43 PFAs
    nationwide:     [ { code, name, groups, totalStops, totalHits } ]       // 1 entry
  }

Each group contains:
  { id, name, population, stops, hits, periods: [
      { months: 1,  stops, hits },
      { months: 3,  stops, hits },
      { months: 6,  stops, hits },
      { months: 12, stops, hits },
      { months: 24, stops, hits },
      { months: 36, stops, hits },
  ]}

Population figures are static (MEP_oslaua_2025.csv).
stops/hits at the group level are grand totals across all months in the dataset.
"""

import json
from pathlib import Path
from collections import defaultdict

import pandas as pd
import geopandas as gpd


SS_CSV  = Path(r"C:\God have mercy on me your unfaithfull slave\master_stop_search_with_neighbourhood.csv")
MEP_CSV = Path(r"C:\God have mercy on me your unfaithfull slave\MEP_oslaua_2025.csv")
LAD_GEO = Path(r"C:\God have mercy on me your unfaithfull slave\LAD_MAY_2025_UK_BGC_V2_7479229380691107175.geojson")
PFA_GEO = Path(r"C:\God have mercy on me your unfaithfull slave\Police_Force_Areas.geojson")

OUTPUT = Path(r"C:\Users\marti\Desktop\RESULTS\combined_dashboard_data_periods.json")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
print(f"Output folder created: {OUTPUT.parent.exists()}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Time windows (in calendar months, counting backwards from the latest month)
# ─────────────────────────────────────────────────────────────────────────────
PERIODS = [1, 3, 6, 12, 24, 36]

# ─────────────────────────────────────────────────────────────────────────────
# Ethnicity mapping  (Officer-defined → group id / name)
# ─────────────────────────────────────────────────────────────────────────────
ETH_MAP = {
    "White":   {"id": 1, "name": "White"},
    "Black":   {"id": 2, "name": "Black"},
    "Asian":   {"id": 3, "name": "Asian"},
    "Mixed":   {"id": 4, "name": "Other"},   # Mixed → Other (no MEP column)
    "Other":   {"id": 4, "name": "Other"},
    None:      {"id": 5, "name": "Unknown"},
}

# MEP column groups  →  (group_id, group_name, [mep_cols])
# Columns in MEP_oslaua_2025: aao abd acn ain apk baf bca oxx unknown wao wbr wir
MEP_GROUPS = [
    (1, "White",   ["wao", "wbr", "wir"]),
    (2, "Black",   ["baf", "bca"]),
    (3, "Asian",   ["aao", "abd", "acn", "ain", "apk"]),
    (4, "Other",   ["oxx"]),
    (5, "Unknown", ["unknown"]),
]
# Total population scale: MEP values are proportions → multiply by 100,000
MEP_SCALE = 100_000

print("Loading stop-search CSV …", flush=True)

usecols = [
    "Date", "month", "Officer-defined ethnicity",
    "Outcome linked to object of search",
    "Outcome",                               # add this
    "Latitude", "Longitude", "police_force",
]
df = pd.read_csv(SS_CSV, usecols=usecols, low_memory=False)
print(df["Outcome"].value_counts(dropna=False).head(10))
# ─── Parse dates ───────────────────────────────────────────────────────────
df["month_dt"] = pd.to_datetime(df["month"], format="%Y-%m", errors="coerce")
df = df.dropna(subset=["month_dt"])

latest_month = df["month_dt"].max()           # e.g. 2026-03-01
print(f"  Latest month in data: {latest_month.strftime('%Y-%m')}", flush=True)
print(f"  Total rows: {len(df):,}", flush=True)

# ─── Hits = Outcome linked to object of search == True ─────────────────────
df["hit"] = df["Outcome"].str.strip().str.lower() != "a no further action disposal"

# ─── Normalise ethnicity ────────────────────────────────────────────────────
def norm_eth(val):
    if pd.isna(val):
        return None
    v = str(val).strip()
    for k in ETH_MAP:
        if k and v.startswith(k):
            return k
    return None

df["eth_norm"] = df["Officer-defined ethnicity"].apply(norm_eth)
df["group_id"]   = df["eth_norm"].map(lambda x: ETH_MAP.get(x, {"id":5})["id"])
df["group_name"] = df["eth_norm"].map(lambda x: ETH_MAP.get(x, {"name":"Unknown"})["name"])

# ─── Normalise police_force (strip " - Copy" suffix) ───────────────────────
df["pf_clean"] = df["police_force"].str.replace(r"\s*-\s*Copy$", "", regex=True).str.strip()

# ─────────────────────────────────────────────────────────────────────────────
# Spatial join: assign each stop to a LAD code
# ─────────────────────────────────────────────────────────────────────────────
print("Loading LAD boundaries for spatial join …", flush=True)
lad_gdf = gpd.read_file(LAD_GEO)[["LAD25CD", "LAD25NM", "geometry"]]
lad_gdf = lad_gdf.to_crs("EPSG:4326")

# Only keep rows with valid coordinates
has_ll = df["Latitude"].notna() & df["Longitude"].notna()
print(f"  Rows with lat/lon: {has_ll.sum():,} / {len(df):,}", flush=True)

df_geo = df[has_ll].copy()
geometry = gpd.points_from_xy(df_geo["Longitude"], df_geo["Latitude"])
gdf = gpd.GeoDataFrame(df_geo, geometry=geometry, crs="EPSG:4326")

print("  Running point-in-polygon join …", flush=True)
joined = gpd.sjoin(gdf, lad_gdf[["LAD25CD", "LAD25NM", "geometry"]],
                   how="left", predicate="within")
df.loc[has_ll, "LAD25CD"] = joined["LAD25CD"].values
df.loc[has_ll, "LAD25NM"] = joined["LAD25NM"].values

# ─────────────────────────────────────────────────────────────────────────────
# PFA join
# ─────────────────────────────────────────────────────────────────────────────
print("Loading PFA boundaries …", flush=True)
pfa_gdf = gpd.read_file(PFA_GEO)[["PFA23CD", "PFA23NM", "geometry"]]
pfa_gdf = pfa_gdf.to_crs("EPSG:4326")

pfa_joined = gpd.sjoin(gdf, pfa_gdf[["PFA23CD", "PFA23NM", "geometry"]],
                        how="left", predicate="within")
df.loc[has_ll, "PFA23CD"] = pfa_joined["PFA23CD"].values
df.loc[has_ll, "PFA23NM"] = pfa_joined["PFA23NM"].values

# ─────────────────────────────────────────────────────────────────────────────
# Population data from MEP CSV
# ─────────────────────────────────────────────────────────────────────────────
print("Loading MEP population data …", flush=True)
mep = pd.read_csv(MEP_CSV)
mep = mep.set_index("oslaua")

def lad_populations(lad_code):
    """Return dict {group_id: population} for a LAD code."""
    pops = {}
    if lad_code not in mep.index:
        for gid, gname, _ in MEP_GROUPS:
            pops[gid] = 1
        return pops
    row = mep.loc[lad_code]
    for gid, gname, cols in MEP_GROUPS:
        prop = sum(row.get(c, 0) for c in cols if c in row.index)
        pops[gid] = max(1, round(prop * MEP_SCALE))
    return pops

# ─────────────────────────────────────────────────────────────────────────────
# Helper: build period cutoffs
# "last N months" = the N most recent calendar months up to and including
# latest_month. E.g. latest=2026-03, N=3 → cutoff=2026-01 (Jan, Feb, Mar).
# ─────────────────────────────────────────────────────────────────────────────
def month_cutoffs(latest, periods):
    cutoffs = {}
    for n in periods:
        cutoff = (latest - pd.DateOffset(months=n)).replace(day=1)
        cutoffs[n] = cutoff
    return cutoffs

cutoffs = month_cutoffs(latest_month, PERIODS)
print("  Period cutoffs:")
for n, c in cutoffs.items():
    print(f"    last {n:2d} months: from {c.strftime('%Y-%m')}")

# ─────────────────────────────────────────────────────────────────────────────
# Aggregate function
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_periods(sub_df):
    """
    Given a subset DataFrame (already filtered to one area), return a dict
    keyed by group_id with period stops/hits for each time window.
    """
    period_data = {}
    for n, cutoff in cutoffs.items():
        period_data[n] = sub_df[sub_df["month_dt"] > cutoff]

    result = {}
    for gid, gname, _ in MEP_GROUPS:
        group_periods = []
        for n in PERIODS:
            g_eth = period_data[n][period_data[n]["group_id"] == gid]
            group_periods.append({
                "months": n,
                "stops":  int(len(g_eth)),
                "hits":   int(g_eth["hit"].sum()),
            })
        result[gid] = {"id": gid, "name": gname, "periods": group_periods}

    return result

# ─────────────────────────────────────────────────────────────────────────────
# Build ADMINISTRATIVE (LAD) entries
# ─────────────────────────────────────────────────────────────────────────────
print("\nBuilding ADMINISTRATIVE (LAD) data …", flush=True)
df_lad = df.dropna(subset=["LAD25CD"])
###
lad_meta = dict(zip(lad_gdf["LAD25CD"], lad_gdf["LAD25NM"]))

administrative = []
lad_codes = sorted(lad_meta.keys())

for i, lad_code in enumerate(lad_codes):
    if i % 50 == 0:
        print(f"  LAD {i+1}/{len(lad_codes)} …", flush=True)

    sub  = df_lad[df_lad["LAD25CD"] == lad_code]
    pops = lad_populations(lad_code)
    agg  = aggregate_periods(sub)

    groups = []
    for gid, gname, _ in MEP_GROUPS:
        g     = agg[gid]
        g_all = sub[sub["group_id"] == gid]       # all-time rows for this group
        groups.append({
            "id":         gid,
            "name":       gname,
            "population": pops[gid],
            "stops":      int(len(g_all)),         # true all-time total
            "hits":       int(g_all["hit"].sum()),
            "periods":    g["periods"],
        })

    administrative.append({
        "code":       lad_code,
        "name":       lad_meta.get(lad_code, lad_code),
        "groups":     groups,
        "totalStops": int(len(sub)),
        "totalHits":  int(sub["hit"].sum()),
    })

# ─────────────────────────────────────────────────────────────────────────────
# Build PFA entries
# ─────────────────────────────────────────────────────────────────────────────
print("\nBuilding PFA data …", flush=True)
df_pfa = df.dropna(subset=["PFA23CD"])
####
pfa_meta = dict(zip(pfa_gdf["PFA23CD"], pfa_gdf["PFA23NM"]))

pfa_list = []
for pfa_code in sorted(pfa_meta.keys()):
    sub = df_pfa[df_pfa["PFA23CD"] == pfa_code]

    lad_in_pfa   = sub["LAD25CD"].dropna().unique()
    pop_by_group = defaultdict(int)
    for lad in lad_in_pfa:
        for gid, pop in lad_populations(lad).items():
            pop_by_group[gid] += pop
    for gid, _, _ in MEP_GROUPS:
        if pop_by_group[gid] == 0:
            pop_by_group[gid] = 1

    agg = aggregate_periods(sub)

    groups = []
    for gid, gname, _ in MEP_GROUPS:
        g     = agg[gid]
        g_all = sub[sub["group_id"] == gid]       # all-time rows for this group
        groups.append({
            "id":         gid,
            "name":       gname,
            "population": int(pop_by_group[gid]),
            "stops":      int(len(g_all)),         # true all-time total
            "hits":       int(g_all["hit"].sum()),
            "periods":    g["periods"],
        })

    pfa_list.append({
        "code":       pfa_code,
        "name":       pfa_meta.get(pfa_code, pfa_code),
        "groups":     groups,
        "totalStops": int(len(sub)),
        "totalHits":  int(sub["hit"].sum()),
    })

# ─────────────────────────────────────────────────────────────────────────────
# Build NATIONWIDE entry
# ─────────────────────────────────────────────────────────────────────────────
print("\nBuilding NATIONWIDE data …", flush=True)

nationwide_pop = defaultdict(int)
for lad_code in lad_codes:
    for gid, pop in lad_populations(lad_code).items():
        nationwide_pop[gid] += pop

agg_nat = aggregate_periods(df)

nationwide_groups = []
for gid, gname, _ in MEP_GROUPS:
    g     = agg_nat[gid]
    g_all = df[df["group_id"] == gid]             # all-time rows for this group
    nationwide_groups.append({
        "id":         gid,
        "name":       gname,
        "population": int(nationwide_pop[gid]),
        "stops":      int(len(g_all)),             # true all-time total
        "hits":       int(g_all["hit"].sum()),
        "periods":    g["periods"],
    })

nationwide = [{
    "code":       "NATIONWIDE",
    "name":       "Nationwide",
    "groups":     nationwide_groups,
    "totalStops": int(len(df)),
    "totalHits":  int(df["hit"].sum()),
}]

# ─────────────────────────────────────────────────────────────────────────────
# Assemble and write output
# ─────────────────────────────────────────────────────────────────────────────
output = {
    "metadata": {
        "generated_from": "generate_dashboard_json.py",
        "latest_month": latest_month.strftime("%Y-%m"),
        "periods_months": PERIODS,
        "period_note": (
            "Each group contains a 'periods' array with stops/hits for the "
            "last 1, 3, 6, 12, 24, and 36 calendar months ending at (and "
            "including) the latest month in the dataset. "
            "The top-level stops/hits on each group are grand totals across "
            "all months present in the dataset."
        ),
        "ethnicity_method": (
            "Officer-defined ethnicity. 'Mixed' stops are merged into 'Other' "
            "because MEP_oslaua_2025.csv has no Mixed population column."
        ),
        "population_source": "MEP_oslaua_2025.csv — proportions × 100,000",
        "ss_method": (
            "Stop-and-search rows are assigned to LAD / PFA boundaries using "
            "point-in-polygon (lat/lon → LAD25CD / PFA23CD)."
        ),
    },
    "administrative": administrative,
    "pfa":            pfa_list,
    "nationwide":     nationwide,
}

print(f"\nWriting output to {OUTPUT} …", flush=True)
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(output, f, separators=(",", ":"))

size_mb = OUTPUT.stat().st_size / 1024 / 1024
print(f"Done! File size: {size_mb:.1f} MB")
print(f"  administrative entries: {len(administrative)}")
print(f"  pfa entries:            {len(pfa_list)}")
print(f"  nationwide entries:     {len(nationwide)}")

# Quick sanity check
sample_lad = administrative[0]
print(f"\nSample LAD: {sample_lad['code']} – {sample_lad['name']}")
print(f"  totalStops={sample_lad['totalStops']}  totalHits={sample_lad['totalHits']}")
g = sample_lad['groups'][0]
print(f"  {g['name']} group — all-time stops={g['stops']} hits={g['hits']}")
print(f"  periods:")
for p in g['periods']:
    print(f"    last {p['months']:2d}m: {p['stops']} stops, {p['hits']} hits")
    

