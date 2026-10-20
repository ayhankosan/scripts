"""
Anti-join comparison of two served/unserved layers in ArcPy.

Produces three outputs in OUT_GDB:
  only_in_mine       - records in MY_LAYER whose key is missing from PARTNER_LAYER
  only_in_partner    - records in PARTNER_LAYER whose key is missing from MY_LAYER
  status_mismatch    - records in both layers whose status differs
                       (exported from MY_LAYER with the partner status added)

Runs in the ArcGIS Pro Python window, a Pro notebook, or the
ArcGIS Pro Python environment (propy.bat).
"""

import os
from collections import Counter

import arcpy

# ---------------------------------------------------------------- Config
# Feature classes, tables, or CSVs. Raw strings keep Windows backslashes intact.
MY_LAYER = r"C:\data\ffa\ffa_status.gdb\FFA_v8_served_unserved"
PARTNER_LAYER = r"C:\data\ffa\ffa_status.gdb\partner_v8_served_unserved"

# Key field in each dataset
MY_KEY = "location_id"
PARTNER_KEY = "location_id"

# Status field in each dataset (set to None to skip the status comparison)
MY_STATUS = "status"
# Field NAME, not alias (gdb field names have no spaces; check with arcpy.ListFields)
PARTNER_STATUS = "Served_Status_Dec_25"

# Translates partner codes to Served/Unserved before comparison.
# ASSUMPTION: 0 = unserved (<25/3), 1 = underserved (25/3 to <100/20), 2 = served (>=100/20).
# Under the 25/3 rule, underserved and served both count as Served.
# Verify against the partner's documentation and the cross-tab printed at the end.
# Set to None when the partner field already holds Served/Unserved text.
PARTNER_STATUS_MAP = {"0": "Unserved", "1": "Served", "2": "Served"}

# Output geodatabase (created if missing)
OUT_GDB = r"C:\data\ffa\antijoin_results.gdb"

# Number of ObjectIDs per selection query
CHUNK_SIZE = 1000

arcpy.env.overwriteOutput = True


# ---------------------------------------------------------------- Helpers
def norm_key(value):
    """Normalizes a key to text so 123, 123.0 and '123' match."""
    if value is None:
        return None
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s or None


def norm_status(value):
    """Normalizes a status value for comparison (case and whitespace insensitive)."""
    return None if value is None else str(value).strip().lower()


def read_keys(dataset, key_field, status_field=None):
    """Returns {key: (oid, status)} for a dataset; reports duplicate and empty keys."""
    fields = ["OID@", key_field] + ([status_field] if status_field else [])
    out, dupes, empty = {}, 0, 0
    with arcpy.da.SearchCursor(dataset, fields) as cur:
        for row in cur:
            k = norm_key(row[1])
            if k is None:
                empty += 1
                continue
            if k in out:
                dupes += 1
                continue
            out[k] = (row[0], row[2] if status_field else None)
    name = os.path.basename(dataset)
    print(f"{name}: {len(out):,} unique keys")
    if dupes:
        print(f"  WARNING: {dupes:,} duplicate keys (first occurrence kept)")
    if empty:
        print(f"  WARNING: {empty:,} records with an empty key (ignored)")
    return out


def export_oids(dataset, oids, out_name):
    """Exports the records with the given ObjectIDs to OUT_GDB; returns the output path."""
    out_path = os.path.join(OUT_GDB, out_name)
    if arcpy.Exists(out_path):
        arcpy.management.Delete(out_path)
    if not oids:
        print(f"{out_name}: 0 records (nothing exported)")
        return None

    desc = arcpy.Describe(dataset)
    is_fc = hasattr(desc, "shapeType")
    view = f"view_{out_name}"
    if is_fc:
        arcpy.management.MakeFeatureLayer(dataset, view)
    else:
        arcpy.management.MakeTableView(dataset, view)

    oid_field = arcpy.AddFieldDelimiters(view, desc.OIDFieldName)
    oids = sorted(oids)
    for i in range(0, len(oids), CHUNK_SIZE):
        chunk = ",".join(str(o) for o in oids[i:i + CHUNK_SIZE])
        arcpy.management.SelectLayerByAttribute(
            view, "ADD_TO_SELECTION" if i else "NEW_SELECTION", f"{oid_field} IN ({chunk})"
        )

    if is_fc:
        arcpy.conversion.ExportFeatures(view, out_path)
    else:
        arcpy.conversion.ExportTable(view, out_path)
    arcpy.management.Delete(view)

    print(f"{out_name}: {int(arcpy.management.GetCount(out_path)[0]):,} records -> {out_path}")
    return out_path


# ---------------------------------------------------------------- Run
if not arcpy.Exists(OUT_GDB):
    arcpy.management.CreateFileGDB(os.path.dirname(OUT_GDB), os.path.basename(OUT_GDB))

compare_status = bool(MY_STATUS and PARTNER_STATUS)
mine = read_keys(MY_LAYER, MY_KEY, MY_STATUS if compare_status else None)
partner = read_keys(PARTNER_LAYER, PARTNER_KEY, PARTNER_STATUS if compare_status else None)

# Anti-join in both directions
only_mine_keys = mine.keys() - partner.keys()
only_partner_keys = partner.keys() - mine.keys()
shared_keys = mine.keys() & partner.keys()

print("\nExporting results")
export_oids(MY_LAYER, [mine[k][0] for k in only_mine_keys], "only_in_mine")
export_oids(PARTNER_LAYER, [partner[k][0] for k in only_partner_keys], "only_in_partner")

# Status comparison on shared keys
def partner_label(value):
    """Translates a partner status value with PARTNER_STATUS_MAP; flags unmapped codes."""
    if PARTNER_STATUS_MAP is None:
        return value
    return PARTNER_STATUS_MAP.get(norm_key(value), f"UNMAPPED:{value}")


mismatch = []
crosstab = Counter()
if compare_status:
    for k in shared_keys:
        mine_st, partner_raw = mine[k][1], partner[k][1]
        crosstab[(mine_st, norm_key(partner_raw))] += 1
        if norm_status(mine_st) != norm_status(partner_label(partner_raw)):
            mismatch.append(mine[k][0])

    out = export_oids(MY_LAYER, mismatch, "status_mismatch")
    if out:
        # Adds the partner code and its translated status next to the original status
        arcpy.management.AddField(out, "partner_code", "TEXT", field_length=20)
        arcpy.management.AddField(out, "partner_status", "TEXT", field_length=50)
        with arcpy.da.UpdateCursor(out, [MY_KEY, "partner_code", "partner_status"]) as cur:
            for row in cur:
                raw = partner[norm_key(row[0])][1]
                row[1] = norm_key(raw)
                row[2] = partner_label(raw)
                cur.updateRow(row)

# ---------------------------------------------------------------- Summary
print("\nSummary")
print(f"  Keys in both layers:     {len(shared_keys):,}")
print(f"  Only in mine:            {len(only_mine_keys):,}")
print(f"  Only in partner:         {len(only_partner_keys):,}")
if compare_status:
    print(f"  Status mismatches:       {len(mismatch):,}")

    # Cross-tab of own status vs. raw partner code, for checking the code mapping
    my_vals = sorted({m for m, _ in crosstab}, key=str)
    codes = sorted({c for _, c in crosstab}, key=str)
    print("\nCross-tab (rows: my status, columns: partner code)")
    print(f"  {'':<12}" + "".join(f"{str(c):>14}" for c in codes))
    for m in my_vals:
        print(f"  {str(m):<12}" + "".join(f"{crosstab[(m, c)]:>14,}" for c in codes))
    if PARTNER_STATUS_MAP is not None:
        unmapped = sorted({c for c in codes if c not in PARTNER_STATUS_MAP}, key=str)
        if unmapped:
            print(f"  WARNING: partner codes missing from PARTNER_STATUS_MAP: {unmapped}")
