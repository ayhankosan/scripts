"""
Anti-join evaluation: FFA v8 served/unserved layer vs. consultant (NTIA primary) layer.

Answers:
  1. Which location_ids are missing from each dataset
  2. Download/upload speed and technology of locations missing from the consultant data
  3. Duplicate location_ids in either dataset
  4. Status agreement on shared locations, and indicators of why records are missing

Outputs (OUT_FOLDER):
  antijoin_evaluation.xlsx        one sheet per matrix (CSV per matrix if openpyxl is missing)
  ids_missing_from_consultant.csv my IDs absent from the consultant layer, with speed and technology
  ids_missing_from_mine.csv       consultant IDs absent from my layer, with consultant status
  duplicates_mine.csv             every duplicated location_id in MY_LAYER
  duplicates_consultant.csv       every duplicated location_id in CONS_LAYER
  antijoin_eval.gdb
    mine_not_in_consultant        full records (geometry + attributes)
    consultant_not_in_mine        full records (geometry + attributes)

Runs in an ArcGIS Pro notebook or the Python window. Layer names from the
Contents pane work, as do full paths (r"C:\\...\\data.gdb\\feature_class").
"""

import os

import numpy as np
import pandas as pd
import arcpy

# =============================================================== Config
MY_LAYER = "FFA_v8_served_unserved_ffa_elig_XYTableToPoint"
CONS_LAYER = "ntia_primary_12312025_wStatus"

MY_KEY = "location_id"
CONS_KEY = "location_id"

MY_STATUS = "status"              # values: served / unserved
CONS_STATUS = "serv_stat_D25"     # values: 0 / 1 / 2
CONS_STATUS_LABELS = {"0": "unserved", "1": "undetermined", "2": "served"}

MY_DOWN = "max_cf_down_mbps"
MY_UP = "max_cf_up_mbps"
MY_HAS_CABLE = "has_cable"
MY_HAS_FIBER = "has_fiber"

# Extra fields profiled to explain missing records (skipped when absent)
MY_PROFILE_FIELDS = ["bsl_flag", "building_type_code", "status_reason", "county_geoid"]
CONS_PROFILE_FIELDS = []          # field names of the consultant layer print at start

# Optional: full v8 Fabric CSV, used to check whether consultant-only IDs exist in v8
FABRIC_REFERENCE_CSV = None       # r"C:\data\fabric\FCC_Active_BSL_v8.csv"
FABRIC_KEY = "location_id"
FABRIC_BSL_FIELD = "bsl_flag"

OUT_FOLDER = r"C:\data\ffa\antijoin_eval"
EXPORT_FEATURES = True            # False skips the feature class exports
TOP_N = 15                        # categories shown per high-cardinality field

arcpy.env.overwriteOutput = True

# =============================================================== Helpers
NUMERIC_TYPES = {"Double", "Single", "Integer", "SmallInteger", "BigInteger", "OID"}
NULL_NUM = -999999
TRUE_VALUES = {"true", "t", "1", "1.0", "yes", "y"}
TIER_ORDER = ["No cable/fiber record", "Below 25/3", "25/3 to <100/20", "100/20 and above"]
TECH_ORDER = ["Fiber + Cable", "Fiber only", "Cable only", "No cable/fiber"]


def find_field(dataset, name, label, required=True):
    """Returns the actual field name (case-insensitive match) or None for missing optional fields."""
    names = {f.name.lower(): f.name for f in arcpy.ListFields(dataset)}
    actual = names.get(name.lower()) if name else None
    if actual is None and required:
        raise ValueError(f"{label}: field '{name}' not found. Available: {sorted(names.values())}")
    if actual is None and name:
        print(f"  {label}: optional field '{name}' not found (skipped)")
    return actual


def load(dataset, field_map):
    """Loads the mapped fields into a DataFrame with canonical column names; nulls become NaN."""
    types = {f.name: f.type for f in arcpy.ListFields(dataset)}
    fields = list(field_map)
    nulls = {f: (NULL_NUM if types[f] in NUMERIC_TYPES else "") for f in fields}
    arr = arcpy.da.TableToNumPyArray(dataset, fields, skip_nulls=False, null_value=nulls)
    df = pd.DataFrame({f: arr[f] for f in fields})
    for f in fields:
        if types[f] in NUMERIC_TYPES:
            col = df[f].astype("float64")
            df[f] = col.mask(col == NULL_NUM)
        else:
            df[f] = df[f].astype(object).where(df[f] != "", np.nan)
    return df.rename(columns=field_map)


def numeric_keys(series):
    """Returns integer keys when every non-null ID is a whole number, otherwise None."""
    num = pd.to_numeric(series, errors="coerce")
    valid = series.notna()
    if num[valid].notna().all() and (num[valid] % 1 == 0).all():
        return num.astype("Int64")
    return None


def text_keys(series):
    """Normalizes IDs to trimmed text without a trailing '.0'."""
    s = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return s.mask(s == "")


def code_text(series):
    """Normalizes codes such as 1, 1.0 and '1 ' to '1'."""
    s = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return s.mask(s == "")


def to_bool(series):
    """Interprets true/false, 1/0 and yes/no style values as booleans."""
    return series.astype("string").str.strip().str.lower().isin(TRUE_VALUES)


def dup_report(df, status_col, out_csv):
    """Writes every duplicated key to CSV; returns (dup keys, extra rows, keys with conflicting status)."""
    counts = df["key"].value_counts()
    dup_keys = counts[counts > 1]
    if dup_keys.empty:
        pd.DataFrame(columns=["location_id", "rows", "statuses"]).to_csv(out_csv, index=False)
        return 0, 0, 0
    dups = df[df["key"].isin(dup_keys.index)]
    detail = dups.groupby("key").agg(
        rows=(status_col, "size"),
        distinct_statuses=(status_col, lambda s: s.astype(str).nunique()),
        statuses=(status_col, lambda s: " | ".join(sorted(s.astype(str).unique()))),
    ).sort_values("rows", ascending=False)
    detail.index.name = "location_id"
    detail.to_csv(out_csv)
    return len(dup_keys), int(dup_keys.sum() - len(dup_keys)), int((detail["distinct_statuses"] > 1).sum())


def speed_tier(down, up):
    """Classifies max cable/fiber speeds into BEAD-style tiers."""
    tier = np.select(
        [down.isna() & up.isna(), (down >= 100) & (up >= 20), (down >= 25) & (up >= 3)],
        [TIER_ORDER[0], TIER_ORDER[3], TIER_ORDER[2]],
        default=TIER_ORDER[1],
    )
    return pd.Categorical(tier, categories=TIER_ORDER, ordered=True)


def speed_stats(series):
    """Returns descriptive statistics for a speed column."""
    s = series.dropna()
    q = s.quantile([0.1, 0.25, 0.5, 0.75, 0.9]) if len(s) else pd.Series([np.nan] * 5)
    return {
        "records": len(series), "with_speed": len(s), "no_speed": int(series.isna().sum()),
        "mean": s.mean(), "min": s.min(), "p10": q.iloc[0], "p25": q.iloc[1],
        "median": q.iloc[2], "p75": q.iloc[3], "p90": q.iloc[4], "max": s.max(),
    }


def representation(missing, base, dimension, top_n=None):
    """Compares category shares of missing records against matched records.

    ratio > 1: category is over-represented among missing records, pointing to a cause.
    """
    a = missing.astype(object).fillna("(null)").value_counts()
    b = base.astype(object).fillna("(null)").value_counts()
    t = pd.DataFrame({"missing_n": a, "matched_n": b}).fillna(0).astype("int64")
    t["missing_pct"] = (100 * t["missing_n"] / max(t["missing_n"].sum(), 1)).round(2)
    t["matched_pct"] = (100 * t["matched_n"] / max(t["matched_n"].sum(), 1)).round(2)
    t["ratio_missing_vs_matched"] = (t["missing_pct"] / t["matched_pct"].replace(0, np.nan)).round(2)
    t = t.sort_values("missing_n", ascending=False)
    if top_n:
        t = t.head(top_n)
    t.index.name = "category"
    t = t.reset_index()
    t.insert(0, "dimension", dimension)
    return t


def crosstab(rows, cols, row_name, col_name):
    """Count matrix with totals."""
    ct = pd.crosstab(rows, cols, margins=True, margins_name="Total", dropna=False)
    ct.index.name, ct.columns.name = row_name, col_name
    return ct


def row_pct(ct):
    """Row percentages of a count matrix with totals."""
    return (ct.div(ct["Total"], axis=0) * 100).round(2)


def export_subset(dataset, key_field, keyset, normalize, out_gdb, out_name):
    """Copies records whose normalized key is in keyset to a new feature class or table."""
    out = os.path.join(out_gdb, out_name)
    if arcpy.Exists(out):
        arcpy.management.Delete(out)
    desc = arcpy.Describe(dataset)
    fields = [f.name for f in arcpy.ListFields(dataset)
              if f.type not in ("OID", "Geometry") and f.name.lower() not in ("shape_length", "shape_area")]
    if hasattr(desc, "shapeType"):
        arcpy.management.CreateFeatureclass(out_gdb, out_name, desc.shapeType.upper(),
                                            template=desc.catalogPath,
                                            spatial_reference=desc.spatialReference)
        cols = ["SHAPE@"] + fields
    else:
        arcpy.management.CreateTable(out_gdb, out_name, template=desc.catalogPath)
        cols = fields
    ki = cols.index(key_field)
    n = 0
    with arcpy.da.SearchCursor(dataset, cols) as src, arcpy.da.InsertCursor(out, cols) as dst:
        for row in src:
            if normalize(row[ki]) in keyset:
                dst.insertRow(row)
                n += 1
    print(f"  {out_name}: {n:,} records -> {out}")


# =============================================================== Load
os.makedirs(OUT_FOLDER, exist_ok=True)
OUT_GDB = os.path.join(OUT_FOLDER, "antijoin_eval.gdb")

print("Consultant layer fields:", [f.name for f in arcpy.ListFields(CONS_LAYER)])

my_map = {
    find_field(MY_LAYER, MY_KEY, "Mine"): "key",
    find_field(MY_LAYER, MY_STATUS, "Mine"): "status_raw",
    find_field(MY_LAYER, MY_DOWN, "Mine"): "down",
    find_field(MY_LAYER, MY_UP, "Mine"): "up",
}
cable_f = find_field(MY_LAYER, MY_HAS_CABLE, "Mine", required=False)
fiber_f = find_field(MY_LAYER, MY_HAS_FIBER, "Mine", required=False)
if cable_f:
    my_map[cable_f] = "has_cable"
if fiber_f:
    my_map[fiber_f] = "has_fiber"
my_profile = [f for f in (find_field(MY_LAYER, p, "Mine", required=False) for p in MY_PROFILE_FIELDS) if f]
my_map.update({f: f for f in my_profile if f not in my_map})

cons_map = {
    find_field(CONS_LAYER, CONS_KEY, "Consultant"): "key",
    find_field(CONS_LAYER, CONS_STATUS, "Consultant"): "status_raw",
}
cons_profile = [f for f in (find_field(CONS_LAYER, p, "Consultant", required=False)
                            for p in CONS_PROFILE_FIELDS) if f]
cons_map.update({f: f for f in cons_profile if f not in cons_map})

print("Loading my dataset ...")
mine = load(MY_LAYER, my_map)
print(f"  {len(mine):,} records")
print("Loading consultant dataset ...")
cons = load(CONS_LAYER, cons_map)
print(f"  {len(cons):,} records")

# Keys: integers when both sides are numeric, otherwise normalized text
mk, ck = numeric_keys(mine["key"]), numeric_keys(cons["key"])
NUMERIC_IDS = mk is not None and ck is not None
if NUMERIC_IDS:
    mine["key"], cons["key"] = mk, ck
else:
    mine["key"], cons["key"] = text_keys(mine["key"]), text_keys(cons["key"])
print(f"Key type: {'integer' if NUMERIC_IDS else 'text'}")


def normalize_scalar(v):
    """Normalizes one raw key value the same way as the loaded keys."""
    if v is None:
        return None
    if NUMERIC_IDS:
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None
    s = str(v).strip()
    return s[:-2] if s.endswith(".0") else s


# Derived attributes, own dataset
mine["status"] = mine["status_raw"].astype("string").str.strip().str.lower().fillna("(null)")
cable = to_bool(mine["has_cable"]) if "has_cable" in mine else pd.Series(False, index=mine.index)
fiber = to_bool(mine["has_fiber"]) if "has_fiber" in mine else pd.Series(False, index=mine.index)
mine["technology"] = pd.Categorical(
    np.select([cable & fiber, fiber, cable], TECH_ORDER[:3], default=TECH_ORDER[3]),
    categories=TECH_ORDER, ordered=True)
mine["speed_tier"] = speed_tier(mine["down"], mine["up"])

# Derived attributes, consultant dataset
code = code_text(cons["status_raw"])
cons["status"] = code.map(CONS_STATUS_LABELS).astype(object)
cons.loc[cons["status"].isna() & code.notna(), "status"] = "unmapped (" + code + ")"
cons["status"] = cons["status"].fillna("(null)")
cons["status_code"] = code.fillna("(null)")

# =============================================================== Duplicates and empty keys
empty_mine, empty_cons = int(mine["key"].isna().sum()), int(cons["key"].isna().sum())
mine, cons = mine[mine["key"].notna()], cons[cons["key"].notna()]

dup_mine = dup_report(mine, "status", os.path.join(OUT_FOLDER, "duplicates_mine.csv"))
dup_cons = dup_report(cons, "status", os.path.join(OUT_FOLDER, "duplicates_consultant.csv"))

mine_u = mine.drop_duplicates("key").copy()
cons_u = cons.drop_duplicates("key").copy()

# =============================================================== Anti-join
mine_u["in_consultant"] = mine_u["key"].isin(cons_u["key"])
cons_u["in_mine"] = cons_u["key"].isin(mine_u["key"])

miss_cons = mine_u[~mine_u["in_consultant"]]     # in mine, missing from consultant
miss_mine = cons_u[~cons_u["in_mine"]]           # in consultant, missing from mine
matched_mine = mine_u[mine_u["in_consultant"]]
matched_cons = cons_u[cons_u["in_mine"]]

max_mine = mine_u["key"].max() if NUMERIC_IDS else None
max_cons = cons_u["key"].max() if NUMERIC_IDS else None
min_mine = mine_u["key"].min() if NUMERIC_IDS else None
min_cons = cons_u["key"].min() if NUMERIC_IDS else None
if NUMERIC_IDS:
    # Fabric IDs grow with each release: IDs above the other max are newer locations,
    # IDs below the other min are older (likely retired) locations
    mine_u["id_position"] = np.select(
        [mine_u["key"] > max_cons, mine_u["key"] < min_cons],
        ["Above consultant max ID (newer)", "Below consultant min ID (older)"],
        default="Within consultant ID range")
    cons_u["id_position"] = np.select(
        [cons_u["key"] > max_mine, cons_u["key"] < min_mine],
        ["Above my max ID (newer)", "Below my min ID (older)"],
        default="Within my ID range")
    miss_cons, matched_mine = mine_u[~mine_u["in_consultant"]], mine_u[mine_u["in_consultant"]]
    miss_mine, matched_cons = cons_u[~cons_u["in_mine"]], cons_u[cons_u["in_mine"]]


def pct(a, b):
    return round(100 * a / b, 2) if b else np.nan


# =============================================================== Matrix 1: overview
overview = pd.DataFrame({
    "My dataset": [
        len(mine) + empty_mine, empty_mine, len(mine_u), dup_mine[0], dup_mine[1], dup_mine[2],
        len(matched_mine), len(miss_cons), pct(len(miss_cons), len(mine_u)),
        min_mine, max_mine,
        int((miss_cons["key"] > max_cons).sum()) if NUMERIC_IDS else None,
        int((miss_cons["key"] < min_cons).sum()) if NUMERIC_IDS else None,
    ],
    "Consultant": [
        len(cons) + empty_cons, empty_cons, len(cons_u), dup_cons[0], dup_cons[1], dup_cons[2],
        len(matched_cons), len(miss_mine), pct(len(miss_mine), len(cons_u)),
        min_cons, max_cons,
        int((miss_mine["key"] > max_mine).sum()) if NUMERIC_IDS else None,
        int((miss_mine["key"] < min_mine).sum()) if NUMERIC_IDS else None,
    ],
}, dtype=object, index=[
    "Total records", "Records with empty location_id", "Unique location_ids",
    "Duplicated location_ids", "Extra rows caused by duplicates",
    "Duplicated location_ids with conflicting status",
    "location_ids in both datasets", "location_ids only in this dataset",
    "% of unique IDs only in this dataset",
    "Min location_id", "Max location_id",
    "Only-in IDs above the other dataset's max ID",
    "Only-in IDs below the other dataset's min ID",
])
overview.index.name = "metric"

# =============================================================== Matrix 2: status agreement
matched = matched_mine[["key", "status", "technology", "speed_tier"]].merge(
    matched_cons[["key", "status"]].rename(columns={"status": "cons_status"}), on="key")
status_ct = crosstab(matched["status"], matched["cons_status"], "my status", "consultant status")
status_pct = row_pct(status_ct)

agree = int(((matched["status"] == "served") & (matched["cons_status"] == "served")).sum()
            + ((matched["status"] == "unserved") & (matched["cons_status"] == "unserved")).sum())
undet = int((matched["cons_status"] == "undetermined").sum())
comparable = len(matched) - undet
agreement = pd.DataFrame({
    "count": [len(matched), undet, comparable, agree, comparable - agree],
    "pct": [100.0, pct(undet, len(matched)), pct(comparable, len(matched)),
            pct(agree, comparable), pct(comparable - agree, comparable)],
}, index=["Matched locations", "Consultant 'undetermined'", "Comparable (excl. undetermined)",
          "Same status", "Different status"])
agreement.index.name = "metric"

# Disagreements by speed tier and technology (explains why statuses differ)
matched["comparison"] = matched["status"] + " -> consultant " + matched["cons_status"]
cmp_speed = crosstab(matched["comparison"], matched["speed_tier"], "my status -> consultant", "my speed tier")
cmp_tech = crosstab(matched["comparison"], matched["technology"], "my status -> consultant", "my technology")

# =============================================================== Matrix 3: missing from consultant
miss_tech_speed = crosstab(miss_cons["technology"], miss_cons["speed_tier"], "technology", "speed tier")
miss_status_tech = crosstab(miss_cons["status"], miss_cons["technology"], "my status", "technology")

stats_rows = {}
for grp, frame in [("Missing from consultant", miss_cons), ("In both datasets", matched_mine)]:
    for col, label in [("down", "download Mbps"), ("up", "upload Mbps")]:
        stats_rows[(grp, label)] = speed_stats(frame[col])
speed_tbl = pd.DataFrame(stats_rows).T.round(2)
speed_tbl.index.names = ["group", "measure"]

# =============================================================== Matrix 4: why missing from consultant
dims = [("my status", "status", None), ("technology", "technology", None),
        ("speed tier", "speed_tier", None)]
if NUMERIC_IDS:
    dims.append(("ID vs consultant ID range", "id_position", None))
dims += [(f, f, TOP_N) for f in my_profile]
why_cons = pd.concat([representation(miss_cons[c], matched_mine[c], d, n) for d, c, n in dims],
                     ignore_index=True)

# =============================================================== Matrix 5: missing from mine
cons_dims = [("consultant status", "status", None)]
if NUMERIC_IDS:
    cons_dims.append(("ID vs my ID range", "id_position", None))
cons_dims += [(f, f, TOP_N) for f in cons_profile]
why_mine = pd.concat([representation(miss_mine[c], matched_cons[c], d, n) for d, c, n in cons_dims],
                     ignore_index=True)

fabric_check = None
if FABRIC_REFERENCE_CSV and len(miss_mine):
    print("Checking consultant-only IDs against the full Fabric ...")
    want = set(miss_mine["key"].astype("int64").tolist()) if NUMERIC_IDS else set(miss_mine["key"].tolist())
    found = {}
    for chunk in pd.read_csv(FABRIC_REFERENCE_CSV, usecols=[FABRIC_KEY, FABRIC_BSL_FIELD],
                             dtype=str, chunksize=2_000_000):
        keys = chunk[FABRIC_KEY].map(normalize_scalar)
        hit = keys.isin(want)
        found.update(zip(keys[hit], chunk.loc[hit, FABRIC_BSL_FIELD]))
    cat = miss_mine["key"].map(lambda k: (
        "Not in Fabric reference (retired or other vintage)" if normalize_scalar(k) not in found
        else ("In Fabric, bsl_flag True (check pipeline)"
              if str(found[normalize_scalar(k)]).strip().lower() in TRUE_VALUES
              else "In Fabric, bsl_flag False (removed by mass-market filter)")))
    fabric_check = cat.value_counts().rename("count").to_frame()
    fabric_check["pct"] = (100 * fabric_check["count"] / len(miss_mine)).round(2)
    fabric_check.index.name = "consultant-only location_ids"

# =============================================================== Findings
findings = []
findings.append(f"{len(miss_cons):,} of my {len(mine_u):,} unique location_ids "
                f"({pct(len(miss_cons), len(mine_u))}%) are missing from the consultant dataset.")
findings.append(f"{len(miss_mine):,} of the consultant's {len(cons_u):,} unique location_ids "
                f"({pct(len(miss_mine), len(cons_u))}%) are missing from my dataset.")
for label, d in [("My dataset", dup_mine), ("Consultant", dup_cons)]:
    findings.append(f"{label}: {d[0]:,} duplicated location_ids ({d[1]:,} extra rows, "
                    f"{d[2]:,} with conflicting status)." if d[0] else f"{label}: no duplicated location_ids.")
if NUMERIC_IDS:
    for name, frame, lo, hi, other in [("Missing from consultant", miss_cons, min_cons, max_cons, "the consultant"),
                                       ("Missing from mine", miss_mine, min_mine, max_mine, "my")]:
        if len(frame):
            above, below = int((frame["key"] > hi).sum()), int((frame["key"] < lo).sum())
            findings.append(f"{name}: {above:,} ({pct(above, len(frame))}%) IDs above {other} max ID "
                            f"(newer Fabric locations), {below:,} ({pct(below, len(frame))}%) below "
                            f"{other} min ID (older, likely retired locations).")
for name, tbl in [("Missing from consultant", why_cons), ("Missing from mine", why_mine)]:
    strong = tbl[(tbl["ratio_missing_vs_matched"] >= 1.5) & (tbl["missing_pct"] >= 10)]
    for _, r in strong.iterrows():
        findings.append(f"{name}: '{r['category']}' ({r['dimension']}) makes up {r['missing_pct']}% of missing "
                        f"vs {r['matched_pct']}% of matched records ({r['ratio_missing_vs_matched']}x).")
    absent = tbl[(tbl["matched_n"] == 0) & (tbl["missing_n"] > 0)]
    for _, r in absent.iterrows():
        findings.append(f"{name}: '{r['category']}' ({r['dimension']}) occurs only among missing records "
                        f"({int(r['missing_n']):,}).")
if comparable:
    findings.append(f"Status agreement on comparable matched locations: {pct(agree, comparable)}% "
                    f"({undet:,} consultant 'undetermined' excluded).")
findings_df = pd.DataFrame({"finding": findings})

# =============================================================== Output
sheets = {
    "00_findings": findings_df,
    "01_overview": overview,
    "02_status_counts": status_ct,
    "03_status_row_pct": status_pct,
    "04_agreement": agreement,
    "05_mismatch_by_speed": cmp_speed,
    "06_mismatch_by_tech": cmp_tech,
    "07_missing_tech_x_speed": miss_tech_speed,
    "08_missing_status_x_tech": miss_status_tech,
    "09_missing_speed_stats": speed_tbl,
    "10_why_missing_from_cons": why_cons,
    "11_why_missing_from_mine": why_mine,
}
if fabric_check is not None:
    sheets["12_cons_only_fabric_check"] = fabric_check

with pd.option_context("display.width", 200, "display.max_columns", 30, "display.max_colwidth", 140):
    for name, tbl in sheets.items():
        print(f"\n=== {name} ===")
        print(tbl.to_string(index=name != "00_findings"))

# Lists of missing location_ids
list_cols = ["key", "status", "technology", "down", "up", "speed_tier"] + \
            (["id_position"] if NUMERIC_IDS else []) + my_profile
miss_cons[list_cols].rename(columns={"key": "location_id", "down": "max_cf_down_mbps", "up": "max_cf_up_mbps"}) \
    .sort_values("location_id").to_csv(os.path.join(OUT_FOLDER, "ids_missing_from_consultant.csv"), index=False)
cons_cols = ["key", "status_code", "status"] + (["id_position"] if NUMERIC_IDS else []) + cons_profile
miss_mine[cons_cols].rename(columns={"key": "location_id", "status": "status_label"}) \
    .sort_values("location_id").to_csv(os.path.join(OUT_FOLDER, "ids_missing_from_mine.csv"), index=False)
print(f"\nID lists -> {OUT_FOLDER}\\ids_missing_from_consultant.csv, ids_missing_from_mine.csv")

xlsx = os.path.join(OUT_FOLDER, "antijoin_evaluation.xlsx")
try:
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        for name, tbl in sheets.items():
            tbl.to_excel(xw, sheet_name=name[:31], index=name != "00_findings")
    print(f"\nMatrices -> {xlsx}")
except ImportError:
    for name, tbl in sheets.items():
        tbl.to_csv(os.path.join(OUT_FOLDER, f"{name}.csv"))
    print(f"\nopenpyxl not available; matrices written as CSV to {OUT_FOLDER}")

if EXPORT_FEATURES:
    print("\nExporting missing records ...")
    if not arcpy.Exists(OUT_GDB):
        arcpy.management.CreateFileGDB(OUT_FOLDER, os.path.basename(OUT_GDB))
    as_set = (lambda s: set(s.astype("int64").tolist())) if NUMERIC_IDS else (lambda s: set(s.tolist()))
    export_subset(MY_LAYER, find_field(MY_LAYER, MY_KEY, "Mine"), as_set(miss_cons["key"]),
                  normalize_scalar, OUT_GDB, "mine_not_in_consultant")
    export_subset(CONS_LAYER, find_field(CONS_LAYER, CONS_KEY, "Consultant"), as_set(miss_mine["key"]),
                  normalize_scalar, OUT_GDB, "consultant_not_in_mine")

print("\nDone.")
