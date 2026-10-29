"""Builds the FFA Fabric V4 -> V8 map series in ArcGIS Pro 3.5.

Creates, for each basemap (Light Gray Canvas and Imagery), three maps and three
layouts, each a spatial map series with one page per project:

  A - Eligibility changes   : Fabric version 4 vs version 8 eligibility data inside the project area
  B - Grantee submission changes : initial grantee submission vs 2026 Q2 and Q3 data
  C - Follow-up             : listed locations the Fabric update does not explain

Pages are 14 x 8.5 in (legal, landscape) and match the per-project PDFs from the
notebook: map on the left (whole project visible), plain-English summary,
legend and sources in fixed panels on the right, so no text overlaps.

How to run (inside ArcGIS Pro):
  1. Run the analysis notebook first so the output geodatabase exists.
  2. In ArcGIS Pro, create or open the project that should hold the maps.
  3. View > Python window, then run:
       exec(open(r"C:\\Change_Analysis\\scripts\\build_map_series.py", encoding="utf-8").read())
  4. Read the printed log: any step that could not be automated prints its manual click-path.

Not executed outside ArcGIS Pro: arcpy is only available there.
"""
import datetime
import os
import time

import arcpy

# ------------------------------------------------------------------ settings
BASE_DIR = r"C:\Change_Analysis"
GDB = os.path.join(BASE_DIR, "outputs", "FFA_V4_V8_Change_Analysis.gdb")
PDF_DIR = os.path.join(BASE_DIR, "outputs", "map_series_pdf")
EXPORT_PDF = False                      # True = one multi-page PDF per layout (slow: every page downloads basemap tiles)
EXPORT_DPI = 150                        # PDF resolution; 300 roughly quadruples export time
PAGE_W, PAGE_H = 14.0, 8.5              # inches, legal landscape (same as the notebook PDFs)
MAP_X0, MAP_Y0, MAP_X1, MAP_Y1 = 0.35, 0.75, 9.35, 7.5     # 9.0 x 6.75 in map frame (matches page_extents)
PANEL_X0, PANEL_X1 = 9.7, 13.65
PROJECTED_SR = arcpy.SpatialReference(3857)                 # page_extents were built in Web Mercator
BASEMAPS = {"gray": "Light Gray Canvas", "imagery": "Imagery"}   # remove a key to skip that variant
OVERVIEW_BASEMAPS = {"gray": "Light Gray Canvas", "imagery": "Imagery Hybrid"}   # both carry place-name labels
OVERVIEW_Y1 = MAP_Y0 + 1.3              # overview map: full panel width, bottom of the right-hand panel
OVERVIEW_MARGIN_PCT = 500               # overview shows the page extent enlarged by this margin
NAVY_RGB = [31, 56, 100, 100]
DENSITY_LAYER = "map_A_no_change_density"          # shaded hexagons for "No change – not eligible"
DENSITY_NAME = "No change – not eligible (shaded, darker = more)"
SHADE_RGB = {1: [204, 204, 204, 75], 2: [170, 170, 170, 75], 3: [130, 130, 130, 75]}
POINT_SIZE = {"Needs follow-up": 5.5, "Explained by the Fabric update": 4.5, "No change": 3.5}
T_V4, T_V8 = "Fabric version 4 eligibility data", "Fabric version 8 eligibility data"
T_SUB4, T_SUB8 = "initial grantee submission", "2026 Q2 and Q3 data"

PAGES = {
    "A": {"title": "Eligibility changes in the project area",
          "subtitle": f"{T_V4[0].upper() + T_V4[1:]} compared with the {T_V8}",
          "points": "map_A_eligibility_changes", "story": "story_a", "legend": "What happened to each location"},
    "B": {"title": "Grantee submission changes",
          "subtitle": f"Initial grantee submission compared with the {T_SUB8}",
          "points": "map_B_list_changes", "story": "story_b", "legend": "What happened to each submitted location"},
    "C": {"title": "Locations that need follow-up",
          "subtitle": "Listed locations the Fabric update does not explain",
          "points": "map_C_follow_up", "story": "story_c", "legend": "Follow-up reason"},
}

FIELD_ALIASES = {
    "map_label": "What happened", "tier": "Tier", "what_it_means": "What it means", "why": "Why",
    "detail": "Detail", "project": "Project", "location_id": "FCC location ID", "page_title": "Project page",
    "eligible_fabric_v4": f"Eligible – {T_V4}", "eligible_fabric_v8": f"Eligible – {T_V8}",
    "needs_follow_up": "Needs follow-up",
    "in_initial_submission": "In initial grantee submission", "in_2026_q2_q3_data": f"In {T_SUB8}",
    "status_fabric_v4": f"Status – {T_V4}", "status_fabric_v8": f"Status – {T_V8}",
    "shift_m": "Point moved (m)", "locations": "Locations", "shade_label": "Shading",
    "point_source": "Point position from",
    "bottom_line": "Bottom line", "story_a": "Eligibility summary", "story_b": "Grantee submission summary",
    "story_c": "Follow-up summary",
}

manual_steps = []


def log(msg):
    """Prints one progress line."""
    print(msg)


def todo(msg):
    """Records a step that needs doing by hand and prints it."""
    manual_steps.append(msg)
    print("  ! MANUAL: " + msg)


# ------------------------------------------------------------------ checks
for name in ("project_areas", "page_extents", "map_A_eligibility_changes", "map_B_list_changes", "map_C_follow_up",
             "map_legend"):
    if not arcpy.Exists(os.path.join(GDB, name)):
        raise RuntimeError(f"{name} not found in {GDB}. Run the analysis notebook first.")
os.makedirs(PDF_DIR, exist_ok=True)

aprx = arcpy.mp.ArcGISProject("CURRENT")

legend = {}       # (page, class) -> (label, tier, [r, g, b, 100])
with arcpy.da.SearchCursor(os.path.join(GDB, "map_legend"),
                           ["page", "map_class", "map_label", "tier", "red", "green", "blue"]) as cur:
    for page, code, label, tier, r, g, b in cur:
        legend[(page, code)] = (label, tier, [int(r), int(g), int(b), 100])
log(f"Legend classes read: {len(legend)}")

for fc in ("project_areas", "page_extents", "map_A_eligibility_changes", "map_B_list_changes", "map_C_follow_up",
           DENSITY_LAYER):
    path = os.path.join(GDB, fc)
    if not arcpy.Exists(path):
        continue
    present = {f.name for f in arcpy.ListFields(path)}
    for field, alias in FIELD_ALIASES.items():
        if field in present:
            try:
                arcpy.management.AlterField(path, field, new_field_alias=alias)
            except Exception as e:                        # alias is cosmetic; keep going
                log(f"  alias not set on {fc}.{field}: {e}")


# Plain field names in the analyst geodatabase (FFA_V4_V8_Analyst.gdb)
ANALYST_GDB = os.path.join(BASE_DIR, "outputs", "FFA_V4_V8_Analyst.gdb")
ANALYST_ALIASES = {
    "page_title": "Project page", "location_id": "FCC location ID", "headline": "What happened (headline)",
    "needs_follow_up": "Needs follow-up", "follow_up_reason": "Follow-up reason",
    "fabric_change": f"Change, {T_V4} to {T_V8}", "fabric_change_tier": "Fabric change tier",
    "submission_change": "Grantee submission change", "submission_change_tier": "Grantee submission change tier",
    "submission_change_why": "Why the submission changed", "in_any_submission": "In a grantee submission",
    "in_initial_submission": "In initial grantee submission", "in_2026_q2_q3_data": f"In {T_SUB8}",
    "in_fabric_v4_data": f"In {T_V4}", "in_fabric_v8_data": f"In {T_V8}",
    "inside_area_fabric_v4": f"Inside area – {T_V4}", "inside_area_fabric_v8": f"Inside area – {T_V8}",
    "eligible_fabric_v4": f"Eligible – {T_V4}", "eligible_fabric_v8": f"Eligible – {T_V8}",
    "status_fabric_v4": f"Status – {T_V4}", "status_fabric_v8": f"Status – {T_V8}",
    "shift_m": "Point moved (m)", "detail": "Detail",
    "listed_more_than_once": "Listed more than once", "duplicate_note": "Duplicate note",
    "label": "Project", "pr_number": "PR number", "bottom_line": "Bottom line",
    "v4_total": f"Locations in area – {T_V4}", "v8_total": f"Locations in area – {T_V8}",
    "v4_elig": f"Eligible in area – {T_V4}", "v8_elig": f"Eligible in area – {T_V8}", "elig_chg": "Change in eligible",
    "sub4_n": "Locations in initial grantee submission", "sub8_n": f"Locations in {T_SUB8}",
    "sub8_elig": f"Of those in {T_SUB8}, eligible in {T_V8}", "fu_total": "Needs follow-up (total)",
    "story_a": "Eligibility summary", "story_b": "Grantee submission summary", "story_c": "Follow-up summary",
}
def read_field_guide(gdb):
    """Plain labels written by the notebook: {column: label} and {field_name: label}. Skipped if unreadable."""
    by_col, by_field = {}, {}
    path = os.path.join(gdb, "field_guide")
    if not arcpy.Exists(path):
        return by_col, by_field
    names = {f.name.lower(): f.name for f in arcpy.ListFields(path)}
    src = next((names[n] for n in ("source_column", "column", "column_", "column_1") if n in names), None)
    if src is None or "field_name" not in names or "label" not in names:
        log(f"  field_guide in {gdb} has unexpected fields {sorted(names)}; aliases skipped")
        return by_col, by_field
    try:
        with arcpy.da.SearchCursor(path, [src, names["field_name"], names["label"]]) as cur:
            for row in cur:
                col, fname, label = row[0], row[1], str(row[2])[:255]
                by_col[col], by_field[fname] = label, label
    except Exception as e:                                  # aliases are cosmetic; the maps do not depend on them
        log(f"  field_guide not read ({e}); aliases skipped")
    return by_col, by_field


guide_by_col, guide_by_field = read_field_guide(GDB)
ANALYST_ALIASES.update(guide_by_field)
for fc in ("project_areas", "summary_projects"):           # technical layers keep short names; aliases explain them
    path = os.path.join(GDB, fc)
    if not arcpy.Exists(path):
        continue
    present = {f.name for f in arcpy.ListFields(path)}
    for field, alias in guide_by_col.items():
        if field in present:
            try:
                arcpy.management.AlterField(path, field, new_field_alias=alias)
            except Exception as e:
                log(f"  alias not set on {fc}.{field}: {e}")
if arcpy.Exists(ANALYST_GDB):
    for fc in ("projects", "locations"):
        path = os.path.join(ANALYST_GDB, fc)
        present = {f.name for f in arcpy.ListFields(path)}
        for field, alias in ANALYST_ALIASES.items():
            if field in present:
                try:
                    arcpy.management.AlterField(path, field, new_field_alias=alias)
                except Exception as e:
                    log(f"  alias not set on analyst {fc}.{field}: {e}")
    log("Analyst geodatabase: plain field aliases set")


# ------------------------------------------------------------------ helpers
def remove_existing(name, kind):
    """Deletes an existing map or layout with the same name so reruns start clean."""
    items = aprx.listMaps(name) if kind == "map" else aprx.listLayouts(name)
    for item in items:
        aprx.deleteItem(item)
        log(f"  removed existing {kind}: {name}")


def style_item(kind, preferred):
    """First style item of a kind, preferring a name that contains `preferred`."""
    items = aprx.listStyleItems("ArcGIS 2D", kind)
    for it in items:
        if preferred.lower() in it.name.lower():
            return it
    return items[0] if items else None


def symbolize_points(lyr, page):
    """Unique values on map_class with the notebook's colours, plain labels and tier-based sizes."""
    sym = lyr.symbology
    sym.updateRenderer("UniqueValueRenderer")
    sym.renderer.fields = ["map_class"]
    for grp in sym.renderer.groups:
        for itm in grp.items:
            code = itm.values[0][0]
            label, tier, rgb = legend.get((page, code), (code, "No change", [128, 128, 128, 100]))
            itm.label = label
            itm.symbol.color = {"RGB": rgb}
            itm.symbol.outlineColor = {"RGB": [255, 255, 255, 100]}
            itm.symbol.outlineWidth = 0.3
            itm.symbol.size = POINT_SIZE.get(tier, 3.0)
    lyr.symbology = sym


def symbolize_density(lyr):
    """Unique values on shade (1-3): light to darker grey hexagons without outlines."""
    sym = lyr.symbology
    sym.updateRenderer("UniqueValueRenderer")
    sym.renderer.fields = ["shade"]
    labels = {1: "Fewer locations", 2: "More locations", 3: "Most locations"}
    for grp in sym.renderer.groups:
        for itm in grp.items:
            shade = int(float(itm.values[0][0]))
            itm.label = labels.get(shade, str(shade))
            itm.symbol.color = {"RGB": SHADE_RGB.get(shade, SHADE_RGB[2])}
            itm.symbol.outlineColor = {"RGB": [0, 0, 0, 0]}
    lyr.symbology = sym


def add_overview(lyt, main_mf, variant, map_name):
    """Overview map frame under the legend: labelled basemap, follows each page's extent, navy extent box."""
    name = f"Overview - {map_name}"
    remove_existing(name, "map")
    om = aprx.createMap(name, "MAP")
    om.spatialReference = PROJECTED_SR
    om.addBasemap(OVERVIEW_BASEMAPS[variant])
    omf = lyt.createMapFrame(arcpy.Extent(PANEL_X0, MAP_Y0, PANEL_X1, OVERVIEW_Y1), om, "Overview")
    lyt_cim = lyt.getDefinition("V3")
    for el in lyt_cim.elements:
        if getattr(el, "name", "") != "Overview":
            continue
        cam = arcpy.cim.CreateCIMObjectFromClassName("CIMAutoCamera", "V3")
        cam.source = main_mf.name
        cam.autoCameraType = "Extent"
        cam.marginType = "Percent"
        cam.margin = OVERVIEW_MARGIN_PCT
        el.autoCamera = cam
        colour = arcpy.cim.CreateCIMObjectFromClassName("CIMRGBColor", "V3")
        colour.values = NAVY_RGB
        stroke = arcpy.cim.CreateCIMObjectFromClassName("CIMSolidStroke", "V3")
        stroke.enable, stroke.width, stroke.color = True, 1.6, colour
        poly = arcpy.cim.CreateCIMObjectFromClassName("CIMPolygonSymbol", "V3")
        poly.symbolLayers = [stroke]
        ref = arcpy.cim.CreateCIMObjectFromClassName("CIMSymbolReference", "V3")
        ref.symbol = poly
        ind = arcpy.cim.CreateCIMObjectFromClassName("CIMExtentIndicator", "V3")
        ind.name = "Page extent"
        ind.sourceMapFrame = main_mf.name
        ind.symbol = ref
        el.extentIndicators = [ind]
    lyt.setDefinition(lyt_cim)
    return omf


def symbolize_outline(lyr, colour):
    """Hollow polygon with a 1.6 pt outline."""
    sym = lyr.symbology
    sym.renderer.symbol.color = {"RGB": [0, 0, 0, 0]}
    sym.renderer.symbol.outlineColor = {"RGB": colour}
    sym.renderer.symbol.outlineWidth = 1.6
    lyr.symbology = sym


def hide(lyr):
    """No fill and no outline (the page-extents index layer only drives the map series)."""
    sym = lyr.symbology
    sym.renderer.symbol.color = {"RGB": [0, 0, 0, 0]}
    sym.renderer.symbol.outlineColor = {"RGB": [0, 0, 0, 0]}
    lyr.symbology = sym


def page_query(lyr, field="page_title"):
    """Shows only the current page's features (Layer Properties > Page Query > Match)."""
    cim = lyr.getDefinition("V3")
    pdef = arcpy.cim.CreateCIMObjectFromClassName("CIMPageDefinition", "V3")
    pdef.fieldName = field
    pdef.type = "Match"
    cim.pageDefinition = pdef
    lyr.setDefinition(cim)


def text_box(lyt, text, x0, y0, x1, y1, size, name):
    """Paragraph text that wraps inside the rectangle (falls back to point text)."""
    poly = arcpy.Polygon(arcpy.Array([arcpy.Point(x0, y0), arcpy.Point(x0, y1), arcpy.Point(x1, y1),
                                      arcpy.Point(x1, y0), arcpy.Point(x0, y0)]))
    last = None
    for ttype, geom in (("POLYGON", poly), ("RECTANGLE", arcpy.Extent(x0, y0, x1, y1)), ("RECTANGLE", poly)):
        try:
            return aprx.createTextElement(lyt, geom, ttype, text, size, name=name)
        except Exception as e:
            last = e
    log(f"  paragraph text fell back to point text ({last})")
    todo(f"Layout '{lyt.name}': convert text '{name}' to paragraph text (right-click > Convert to Paragraph Text) "
         f"and fit it to x {x0}-{x1} in, y {y0}-{y1} in.")
    return aprx.createTextElement(lyt, arcpy.Point(x0, y1), "POINT", text, size, name=name)


def enable_map_series(lyt, mf, index_lyr):
    """Spatial map series on page_extents: name field page_title, best fit with no margin."""
    if hasattr(lyt, "createSpatialMapSeries"):
        lyt.createSpatialMapSeries(mf, index_lyr, "page_title", "project_key")
    else:                                               # Python CIM route
        cim = lyt.getDefinition("V3")
        ms = arcpy.cim.CreateCIMObjectFromClassName("CIMSpatialMapSeries", "V3")
        ms.enabled = True
        ms.mapFrameName = mf.name
        ms.indexLayerURI = index_lyr.getDefinition("V3").uRI
        ms.nameField = "page_title"
        ms.sortField = "project_key"
        ms.sortAscending = True
        ms.startingPageNumber = 1
        cim.mapSeries = ms
        lyt.setDefinition(cim)
    lyt = aprx.listLayouts(lyt.name)[0]
    try:                                                # page_extents are already frame-shaped
        cim = lyt.getDefinition("V3")
        cim.mapSeries.extentOptions = "BestFit"
        cim.mapSeries.marginType = "Percent"
        cim.mapSeries.margin = 0
        lyt.setDefinition(cim)
    except Exception as e:
        log(f"  extent option not set ({e})")
    lyt = aprx.listLayouts(lyt.name)[0]
    lyt.mapSeries.refresh()
    return lyt


# ------------------------------------------------------------------ build
today = datetime.date.today().strftime("%B %d, %Y")
footer = ("Sources: Fabric version 4 eligibility data (field FFAv4Dec23) and Fabric version 8 eligibility data "
          "(field status_code), 0 = unserved (eligible), 1 = served; "
          "CPUC FFA awarded project areas; initial grantee submission and 2026 Q2 and Q3 data. "
          f"Eligible = no cable or fiber at 25/3 Mbps or better. Created {today}.")
with arcpy.da.SearchCursor(os.path.join(GDB, "page_extents"), ["page_title"]) as cur:
    longest_title = max((len(row[0] or "") for row in cur), default=40)


def title_size(heading):
    """Largest font size (11-17 pt) that keeps the longest page title on one line inside the page."""
    chars = longest_title + 3 + len(heading)
    return max(11.0, min(17.0, round((PAGE_W - 0.7) * 72 / (0.6 * chars) * 2) / 2))


for variant, basemap in BASEMAPS.items():
    for key, cfg in PAGES.items():
        suffix = "" if variant == "gray" else " (imagery)"
        map_name = f"{key} - {cfg['title']}{suffix}"
        lyt_name = f"Map Series {key} - {cfg['title']}{suffix}"
        log(f"\n=== {lyt_name} === ({time.strftime('%H:%M:%S')})")
        remove_existing(map_name, "map")
        remove_existing(lyt_name, "layout")

        m = aprx.createMap(map_name, "MAP")
        m.spatialReference = PROJECTED_SR
        try:
            m.addBasemap(basemap)
        except Exception as e:
            log(f"  basemap not added ({e})")
            todo(f"Map '{map_name}': Map tab > Basemap > {basemap} (sign in to ArcGIS Online if prompted).")

        extents_lyr = m.addDataFromPath(os.path.join(GDB, "page_extents"))
        extents_lyr.name = "Page extent (map series index)"
        areas = m.addDataFromPath(os.path.join(GDB, "project_areas"))
        areas.name = "Project area outline"
        pts = m.addDataFromPath(os.path.join(GDB, cfg["points"]))
        pts.name = cfg["legend"]
        m.moveLayer(areas, pts, "BEFORE")
        dens = None
        if key == "A" and arcpy.Exists(os.path.join(GDB, DENSITY_LAYER)):
            dens = m.addDataFromPath(os.path.join(GDB, DENSITY_LAYER))
            dens.name = DENSITY_NAME
            m.moveLayer(pts, dens, "AFTER")                 # hexagons under the points
            try:
                symbolize_density(dens)
            except Exception as e:
                log(f"  density symbology failed ({e})")
                todo(f"Map '{map_name}': symbolize '{DENSITY_NAME}' by Unique Values on shade, light to darker grey, no outline.")

        for fn, args in ((hide, (extents_lyr,)),
                         (symbolize_outline, (areas, [255, 255, 255, 100] if variant == "imagery" else [31, 56, 100, 100])),
                         (symbolize_points, (pts, key))):
            try:
                fn(*args)
            except Exception as e:
                log(f"  {fn.__name__} failed ({e})")
                if fn is symbolize_points:
                    todo(f"Map '{map_name}': symbolize '{pts.name}' by Unique Values on map_class; colours and labels "
                         "are in the map_legend table.")

        # Layout: map left, summary + legend right, title band and footer
        lyt = aprx.createLayout(PAGE_W, PAGE_H, "INCH", lyt_name)
        mf = lyt.createMapFrame(arcpy.Extent(MAP_X0, MAP_Y0, MAP_X1, MAP_Y1), m, "Map Frame")
        aprx.createTextElement(lyt, arcpy.Point(MAP_X0, PAGE_H - 0.5), "POINT",
                               f'<dyn type="page" property="page_title"/> · {cfg["title"]}', title_size(cfg["title"]), name="Title")
        aprx.createTextElement(lyt, arcpy.Point(MAP_X0, PAGE_H - 0.8), "POINT", cfg["subtitle"], 10.5, name="Subtitle")
        aprx.createTextElement(lyt, arcpy.Point(PANEL_X0, MAP_Y1 - 0.05), "POINT", "Description", 11.5,
                               name="Summary heading")
        text_box(lyt, f'<dyn type="page" property="{cfg["story"]}"/>', PANEL_X0, MAP_Y1 - 2.65, PANEL_X1, MAP_Y1 - 0.3,
                 9.5, "Summary")
        try:
            leg = lyt.createMapSurroundElement(arcpy.Extent(PANEL_X0, OVERVIEW_Y1 + 0.2, PANEL_X1, MAP_Y1 - 2.85), "LEGEND",
                                               mf, style_item("LEGEND", "Legend 1"), "Legend")
            leg.title = cfg["legend"]
            try:
                leg.fittingStrategy = "AdjustColumns"      # same font size on every page
            except Exception as e:
                log(f"  legend fitting not set ({e})")
            for item in leg.items:
                try:
                    if item.name == extents_lyr.name:
                        leg.removeItem(item)
                    else:
                        item.showVisibleFeatures = True          # only categories on the current page
                except Exception as e:
                    log(f"  legend item option not set ({e})")
                    todo(f"Layout '{lyt_name}': Legend > each item > Show features visible in the map extent.")
                    break
        except Exception as e:
            log(f"  legend issue ({e})")
            todo(f"Layout '{lyt_name}': Insert > Legend in the right-hand panel; untick the page extent layer.")
        for kind, pref, pt in (("NORTH_ARROW", "ArcGIS North 1", arcpy.Point(MAP_X1 - 0.45, MAP_Y1 - 0.75)),
                               ("SCALE_BAR", "Scale Line 1", arcpy.Point(MAP_X0 + 0.2, MAP_Y0 + 0.15))):
            try:
                lyt.createMapSurroundElement(pt, kind, mf, style_item(kind, pref), kind.title().replace("_", " "))
            except Exception as e:
                log(f"  {kind.lower()} not added ({e})")
        text_box(lyt, footer, MAP_X0, 0.12, PAGE_W - 1.6, 0.55, 7, "Sources")
        try:
            add_overview(lyt, mf, variant, map_name)
        except Exception as e:
            log(f"  overview map not added ({e})")
            todo(f"Layout '{lyt_name}': Insert > Map Frame > a map with the {OVERVIEW_BASEMAPS[variant]} basemap, "
                 f"placed at x {PANEL_X0}-{PANEL_X1} in, y {MAP_Y0}-{OVERVIEW_Y1} in; Map Frame properties > Display "
                 f"options > Constraint: Linked map frame extent ('Map Frame'), margin {OVERVIEW_MARGIN_PCT}%; "
                 "then Insert > Extent Indicator from 'Map Frame', navy outline.")

        try:
            lyt = enable_map_series(lyt, mf, extents_lyr)
            log(f"  map series: {lyt.mapSeries.pageCount} pages")
        except Exception as e:
            log(f"  map series not created ({e})")
            todo(f"Layout '{lyt_name}': Layout tab > Map Series > Spatial. Map frame: 'Map Frame'; Layer: "
                 "'Page extent (map series index)'; Name field: page_title; Sort field: project_key; "
                 "Map extent: Best fit, margin 0%.")

        for lyr in [l for l in (pts, dens, areas, extents_lyr) if l is not None]:
            try:
                page_query(lyr)
            except Exception as e:
                log(f"  page query not set on '{lyr.name}' ({e})")
                todo(f"Map '{map_name}': layer '{lyr.name}' > Properties > Page Query > field page_title, Match.")

        if EXPORT_PDF and lyt.mapSeries is not None and getattr(lyt.mapSeries, "enabled", False):
            out = os.path.join(PDF_DIR, f"Map_Series_{key}_{variant}.pdf")
            try:
                lyt.mapSeries.exportToPDF(out, "ALL", resolution=EXPORT_DPI)
                log(f"  PDF: {out}")
            except Exception as e:
                log(f"  PDF export failed ({e})")

aprx.save()
log("\nProject saved.")
if manual_steps:
    log("\nFinish these by hand:")
    for i, s in enumerate(manual_steps, 1):
        log(f"  {i}. {s}")
else:
    log("Everything was automated. Open the layouts from the Catalog pane > Layouts.")
