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
PARTNER_STATUS = "status"

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
mismatch = {}
if compare_status:
    mismatch = {
        mine[k][0]: partner[k][1]
        for k in shared_keys
        if norm_status(mine[k][1]) != norm_status(partner[k][1])
    }
    out = export_oids(MY_LAYER, list(mismatch), "status_mismatch")
    if out:
        # Adds the partner status next to the original status
        arcpy.management.AddField(out, "partner_status", "TEXT", field_length=50)
        key_to_partner = {k: partner[k][1] for k in shared_keys}
        with arcpy.da.UpdateCursor(out, [MY_KEY, "partner_status"]) as cur:
            for row in cur:
                row[1] = key_to_partner.get(norm_key(row[0]))
                cur.updateRow(row)

# ---------------------------------------------------------------- Summary
print("\nSummary")
print(f"  Keys in both layers:     {len(shared_keys):,}")
print(f"  Only in mine:            {len(only_mine_keys):,}")
print(f"  Only in partner:         {len(only_partner_keys):,}")
if compare_status:
    print(f"  Status mismatches:       {len(mismatch):,}")
