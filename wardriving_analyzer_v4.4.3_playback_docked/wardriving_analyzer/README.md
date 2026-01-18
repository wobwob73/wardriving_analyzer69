# Wardriving Analyzer v4.4.3

Multi-spectrum RF survey / wardriving analysis tool for Ubuntu 24.

It supports:
- **Offline/local analysis** of one or more CSV runs
- **Live ingest** from an MQTT broker (optional)
- Map + table UI with filtering, selection, focus mode, and playback
- Exports: **KML**, **GeoJSON**, **CSV**, and **PDF/HTML reports**

---

## What it does

### Core workflow
1. **Ingest** one or more CSV files ("runs") and/or stream observations from MQTT.
2. **Normalize** incoming rows into a consistent internal detection schema using a **CSV Profile**.
3. **Aggregate** detections into SOIs (signals of interest) by ID (MAC/device identifier).
4. Compute per-SOI stats (detections, RSSI stats, channels, security modes, strongest point, observed radius, etc.).
5. **Classify** likely static vs mobile using heuristics (RSSI variance, geographic spread, clustering, multi-run consistency).
6. **Visualize** in:
   - **Map View** (markers + selection overlays)
   - **Table View** (searchable/sortable-ish list with selection)
7. **Export** GIS formats and/or **generate a report**.

---

## UI highlights

### Filters and visibility
- Filter by **device type** (WiFi/BLE/Thread/HaLow), **classification**, and **WiFi security**
- Per-worker **enable/disable** + **opacity sliders**
- **Whitelist/Hide** SOIs so they are ignored by static display and playback
- **Focus mode**: dim non-selected SOIs by a configurable opacity

### Selection details
When selecting a single SOI, the UI can show:
- **Strongest heard point** (centered on strongest RSSI)
- **Observed radius** (coverage boundary from captured detections)
- **Estimated physical location** (weighted center from detections)
- Additional rings/overlays in map view (first/last seen boundary, weighted area, etc.)

### Playback
Playback replays detections over time:
- Shows only SOIs present in the current playback frame
- Removes SOIs when no longer detected in subsequent frames
- Obeys worker filters, WiFi security filter, and whitelist

In v4.4.3 the playback controls are **docked in the bottom section of the SOI Info panel** (left side), so they can’t “fail to appear” as a popup.

---

## CSV Profiles (important)

Wardriving data in the wild comes in many shapes. This project solves that by using **Profiles**: small JSON mapping files that describe how to interpret a CSV format.

### Built-in profiles
- `default_profiles/v42_default.json`
- `default_profiles/scan_extended_v1.json`

### Add your own profile
1. Open **Settings → Profiles**
2. Click the **+** button and upload a `*.json` profile
3. The profile is persisted and will survive restarts (saved under the app’s local config directory).

A profile typically maps:
- unique ID field (MAC/device id)
- name/SSID field
- timestamp field (ISO8601 or epoch)
- lat/lon fields
- RSSI field
- type field (wifi/ble/thread/halow)
- optional: channel, security

---

## Input requirements (minimum)

### For map view & playback
To render in **Map View** and to participate in **Playback**, detections should include:
- `timestamp` (ISO8601 or epoch seconds/ms)
- `lat`, `lon` (non-zero)
- `rssi` (dBm)
- unique `id` (MAC/device identifier)

### For table-only analysis
If GPS is missing, SOIs still appear in **Table View** and are included in exports/reports where relevant.

---

## Reports

### Report formats
- **PDF** (ReportLab)
- **HTML**

### Report options
In **Settings → Report Options**:
- global font family/size/bold
- company name + font settings
- optional watermark logo with opacity control (default ~8%)

Reports include (for each SOI):
- strongest heard point
- estimated physical location
- observed radius
- security/channels/workers and other derived stats

---

## Install / Run

```bash
cd wardriving_analyzer
./install.sh
python3 backend/app.py
```

Then open:
- http://localhost:5000

---

## Exports
- **GeoJSON**: `/api/export/geojson`
- **KML**: `/api/export/kml`
- **CSV**: `/api/export/csv`
- **Report**: generated from the UI (selected SOIs or current filtered set)

---

## Troubleshooting

### Playback shows “no GPS detections”
- Ensure your CSV profile correctly maps **timestamp**, **lat**, **lon**, and **RSSI**.
- Playback only frames detections with GPS.

### MQTT ingest
- Confirm broker host/port/credentials
- Ensure your MQTT payload schema matches the configured handler

