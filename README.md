Wardriving Analyzer v4.4.1 — Quick README
What this build is

v4.4.1 is an incremental feature + stability build of the Wardriving Analyzer focused on:

Flexible CSV ingestion via persistent “profiles” (multiple CSV formats supported without hard-coded column names)

Improved map focus UX (SOI detail pane layout, focus-mode dimming, and ring overlays with hover tooltips)

Reporting configuration (fonts, company branding, watermark)

Playback + export pipelines (intended to support time-based replay and PDF/HTML reporting)

Note: v4.4.1 is meant to be the “testable checkpoint” after adding the left-side SOI info panel + focus ring hover tooltips and after trying to address report/playback regressions.

What it does (core features)
Live + Offline Data

Offline/local parsing: load CSV logs from disk

Optional live ingest: MQTT ingest path (if enabled/configured in your environment)

Merges detections into SOIs (Signals of Interest) and computes per-SOI aggregate statistics.

Views

Map View: renders SOIs as markers; supports selecting a SOI to focus it.

Table View: lists SOIs; supports selecting SOIs and switching to map focus.

Focus Mode / Selection

When a single SOI is selected, the map can render:

Strongest point region (solid circle)

Weighted “most likely location” region (semi-opaque ellipse)

First/last heard extent (dotted ring)

Non-selected SOIs can be dimmed via a focus-mode opacity control.

Whitelist / Hide

SOIs can be “hidden/whitelisted” so they do not appear in:

Map view

Table view

Playback

Reports/exports (depending on settings)

Exports

Export SOI data as:

CSV

GeoJSON

KML

Generate a Report (intended as PDF primarily; HTML may exist depending on settings).

What it’s supposed to do (intended behavior)
1) CSV Profiles (robust multi-format ingestion)

Instead of requiring one exact CSV format, the analyzer uses profiles that map arbitrary CSV columns into a canonical internal schema.

Two profiles are included/mapped by default:

A “legacy/canonical” CSV format

An “extended/example” CSV format (the richer scan export style)

You can add your own profile via Settings using a + upload (JSON profile).

Profiles are saved to disk so they persist across restarts.

2) SOI information pane layout

SOI details should be visible as a fixed panel (so it isn’t cut off). The intended layout is:

SOI detail panel on one side

Load/Stats/Filters panels on the other side

Table/Map sandwiched between, with collapsible headers/panels.

3) Playback (time-based replay)

Intended behavior:

Playback popup should provide:

play/pause, stop

start/end jump

forward/backward play

speed slider

During playback:

only SOIs currently detected in the active time frame are shown

SOIs disappear when no longer detected

4) Reports (PDF/HTML)

Intended behavior:

Generate report for:

selected SOI

multiple selected SOIs

Report includes:

strongest heard point

radius/extent

“most likely” physical location estimate

known characteristics (security, channel, etc.)

Report styling is configurable in Settings:

base font / size

company name styling

optional logo watermark with opacity (default ~8%)

High-level how it works (architecture overview)
Backend (Python)

Flask serves the UI and provides REST endpoints for:

uploading/loading CSV

exporting formats

generating reports

supplying detections for playback

A central analysis engine:

reads detections (CSV/MQTT)

normalizes them into a canonical detection shape

aggregates detections into SOIs

computes:

strongest point (max RSSI)

weighted centroid / covariance (for “most likely” region)

first/last seen extents

classification heuristics (fixed vs mobile)

Frontend (Browser UI)

Renders:

a map layer for markers + overlays (rings/ellipse)

a table for SOIs

settings modal for profiles + report options

Applies user preferences (opacity, panel collapse, whitelist) and stores some UI state locally.

Input format requirements
Minimum required per detection row (conceptual)

For best results:

identifier (MAC/device id/BSSID) → mapped to internal mac

signal type (wifi / bt / thread / halow / etc.)

timestamp (ISO8601 UTC recommended)

Strongly recommended for mapping/geo features:

RSSI (dBm)

Latitude / Longitude

Fix validity indicator (e.g., fix_valid, fix_ok, sats/hdop)

Worker / antenna ID

Channel / frequency (especially for WiFi)

CSV Profiles determine “which columns mean what”

A CSV profile JSON provides:

id, name

mapping dict mapping canonical keys to CSV column names (or a list of candidates)

Example mapping keys commonly supported:

mac, name, signal_type, timestamp

rssi, lat, lon, fix_ok

worker, channel

security (WiFi encryption/security)

WiFi Security filtering (if present)

If your CSV contains security fields, map them to security:

Open / Unsecured

WEP

WPA1

WPA2

WPA3

WPA2/3 mixed

OWE

Unknown

If not present, WiFi security will be treated as Unknown and security filtering won’t be meaningful.

Persistent configuration locations (v4.4.1)

Paths may vary depending on your packaging, but the intent is:

CSV Profiles stored under something like:
backend/user_data/profiles/

App settings (active profile, report options, watermark):
backend/user_data/settings.json

Known issues (v4.4.1)

Based on observed behavior during testing of this build line:

PDF report export may fail due to style/font attribute resolution issues (e.g., missing/undefined font variables in the generator).

Playback UI may not appear in some cases (event binding / popup rendering dependency issues).

(These are addressed in later hotfix builds, but this README is specifically describing v4.4.1.)

Quick “what to validate” checklist for v4.4.1

Load CSV using the correct profile

Map view:

select a SOI

verify rings/ellipse show and hover tooltips appear

Table view:

select SOIs and “View” in map

Report generation:

(may fail in v4.4.1; see known issues)
